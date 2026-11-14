// Shared pieces of the RelayVSR port: torch's RMSNorm, the ×4 resamplers torch's `F.interpolate` computes, and the
// window masks both networks attend under.

import Foundation
import MLX
import MLXNN

/// `torch.nn.RMSNorm(dim, eps)`: x · rsqrt(mean(x², −1) + eps) · weight.
///
/// Upstream evaluates it in the input's own dtype (bf16 under `--dtype bf16`). Here the reduction runs in fp32 and
/// the result is cast back: identical in the fp32 lane, and in a half-precision lane it removes the one place MLX
/// loses more than torch does — a mean of squares over 512–3072 values accumulated in 8 mantissa bits.
final class TorchRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    init(_ dim: Int, eps: Float) {
        weight = MLXArray.ones([dim])
        self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        return (xf * rsqrt((xf * xf).mean(axis: -1, keepDims: true) + eps) * weight.asType(.float32)).asType(x.dtype)
    }
}

/// `nn.Sequential` kept as a layer array so `layers.0` / `layers.2` … line up with upstream's keys.
@inline(__always) func seq(_ layers: [UnaryLayer], _ x: MLXArray) -> MLXArray { layers.reduce(x) { $1($0) } }

/// ×4 upsampling exactly as `F.interpolate(x, scale_factor=4, mode=…, align_corners=False)` computes it.
///
/// Output pixel o = 4·i + p samples the source at i + (p + 0.5)/4 − 0.5, so the four phases have FIXED taps:
///   p = 0, 1 → floor = i − 1, t = 0.625, 0.875        p = 2, 3 → floor = i, t = 0.125, 0.375
/// bilinear: taps floor, floor + 1 with (1 − t, t); torch clamps the source coordinate at 0, which equals edge
/// replication. bicubic (A = −0.75, no antialias): taps floor − 1 … floor + 2, indices clamped → edge replication.
public enum RelayVSRResample {
    /// Per phase: (offset from i, weight).
    static func taps(cubic: Bool) -> [[(Int, Float)]] {
        let ts: [(Int, Double)] = [(-1, 0.625), (-1, 0.875), (0, 0.125), (0, 0.375)]
        return ts.map { fl, t in
            if !cubic { return [(fl, Float(1 - t)), (fl + 1, Float(t))] }
            let a = -0.75
            func outer(_ x: Double) -> Double { ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }      // 1 ≤ |x| < 2
            func inner(_ x: Double) -> Double { ((a + 2) * x - (a + 3)) * x * x + 1 }              // |x| ≤ 1
            return [(fl - 1, Float(outer(t + 1))), (fl, Float(inner(t))), (fl + 1, Float(inner(1 - t))),
                    (fl + 2, Float(outer(2 - t)))]
        }
    }

    /// x: (B, H, W, C) → (B, 4H, 4W, C). Computed in fp32, returned in x's dtype.
    public static func upsample4(_ x: MLXArray, cubic: Bool) -> MLXArray {
        let t = taps(cubic: cubic)
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let xf = x.asType(.float32)
        // rows
        let pr = padded(xf, widths: [0, .init((2, 2)), 0, 0], mode: .edge)
        let rows = t.map { ph in
            ph.map { off, wt in pr[0..., (off + 2) ..< (off + 2 + h)] * wt }.reduce(MLXArray(Float(0))) { $0 + $1 }
        }
        let xr = stacked(rows, axis: 2).reshaped([b, h * 4, w, c])
        // columns
        let pc = padded(xr, widths: [0, 0, .init((2, 2)), 0], mode: .edge)
        let cols = t.map { ph in
            ph.map { off, wt in pc[0..., 0..., (off + 2) ..< (off + 2 + w)] * wt }.reduce(MLXArray(Float(0))) { $0 + $1 }
        }
        return stacked(cols, axis: 3).reshaped([b, h * 4, w * 4, c]).asType(x.dtype)
    }
}

/// A boolean attention mask over an (h, w) query grid and one (h, w) key plane, separable in rows and columns:
/// visible iff `rowVisible(qy, ky) && colVisible(qx, kx)`. Shape (h·w, h·w), row-major tokens.
func separableMask(h: Int, w: Int, rowVisible: (Int, Int) -> Bool, colVisible: (Int, Int) -> Bool) -> MLXArray {
    var ry = [Bool](repeating: false, count: h * h), cx = [Bool](repeating: false, count: w * w)
    for q in 0 ..< h { for k in 0 ..< h { ry[q * h + k] = rowVisible(q, k) } }
    for q in 0 ..< w { for k in 0 ..< w { cx[q * w + k] = colVisible(q, k) } }
    let y = MLXArray(ry, [h, 1, h, 1]), x = MLXArray(cx, [1, w, 1, w])
    return (y .&& x).reshaped([h * w, h * w])
}

/// Named intermediate tensors a parity gate can ask a network for.
public typealias RelayVSRTaps = [String: MLXArray]
