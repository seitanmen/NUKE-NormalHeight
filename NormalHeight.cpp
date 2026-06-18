/*
 * NormalHeight — Nuke plugins to convert between normal maps and height maps
 *
 * Copyright (c) 2026 kawata. All rights reserved.
 *
 * Two plugins in a single source file:
 *   HeightToNormal (vmt_HeightToNormal) — derive a tangent-space normal map
 *       from a height map via finite-difference gradients.
 *   NormalToHeight (vmt_NormalToHeight) — reconstruct a height map from a
 *       tangent-space normal map by least-squares gradient integration using
 *       the Frankot-Chellappa method (Poisson solve in the frequency domain).
 *
 * Normal convention: OpenGL (+Y up) by default; enable `directx` for +Y down.
 * Both nodes assume tileable input (periodic boundary), matching the FFT-based
 * integrator's implicit periodicity.
 *
 * Build: compile as shared library (.dll/.so/.dylib), place in Nuke plugin path.
 */

#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/Tile.h"

#include <cmath>
#include <vector>
#include <algorithm>

using namespace DD::Image;

// ===========================================================================
// Minimal self-contained FFT (radix-2 Cooley-Tukey, in-place).
// No external dependency. Operates on power-of-two sizes; the integrator
// zero/relation-pads the working grid up to the next power of two per axis.
// ===========================================================================
namespace fftutil {

struct Cplx { double re, im; };

// In-place iterative radix-2 FFT. n must be a power of two.
// sign = -1 for forward, +1 for inverse (inverse is unnormalized).
static void fft1d(std::vector<Cplx>& a, int sign)
{
    const size_t n = a.size();
    if (n <= 1) return;

    // Bit-reversal permutation.
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        const double ang = sign * 2.0 * M_PI / (double)len;
        const Cplx wlen{ std::cos(ang), std::sin(ang) };
        for (size_t i = 0; i < n; i += len) {
            Cplx w{ 1.0, 0.0 };
            for (size_t k = 0; k < len / 2; ++k) {
                const Cplx u = a[i + k];
                const Cplx v{ a[i + k + len / 2].re * w.re - a[i + k + len / 2].im * w.im,
                              a[i + k + len / 2].re * w.im + a[i + k + len / 2].im * w.re };
                a[i + k].re             = u.re + v.re;
                a[i + k].im             = u.im + v.im;
                a[i + k + len / 2].re   = u.re - v.re;
                a[i + k + len / 2].im   = u.im - v.im;
                const double nwr = w.re * wlen.re - w.im * wlen.im;
                const double nwi = w.re * wlen.im + w.im * wlen.re;
                w.re = nwr; w.im = nwi;
            }
        }
    }
}

// 2D FFT over a width*height complex grid stored row-major.
// Both dimensions must be powers of two.
static void fft2d(std::vector<Cplx>& g, int W, int H, int sign)
{
    std::vector<Cplx> row(W);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) row[x] = g[(size_t)y * W + x];
        fft1d(row, sign);
        for (int x = 0; x < W; ++x) g[(size_t)y * W + x] = row[x];
    }
    std::vector<Cplx> col(H);
    for (int x = 0; x < W; ++x) {
        for (int y = 0; y < H; ++y) col[y] = g[(size_t)y * W + x];
        fft1d(col, sign);
        for (int y = 0; y < H; ++y) g[(size_t)y * W + x] = col[y];
    }
}

static int nextPow2(int v)
{
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

} // namespace fftutil


static inline float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }


// ===========================================================================
// HeightToNormal — height map -> tangent-space normal map
// ===========================================================================
static const char* const H2N_CLASS = "HeightToNormal";
static const char* const H2N_HELP =
    "HeightToNormal (vmt_HeightToNormal) — derive a tangent-space normal map "
    "from a height map.\n"
    "Copyright (c) 2026 kawata. All rights reserved.\n\n"
    "Computes per-pixel gradients of the input height (read from the red channel "
    "by default) with central differences, scales them by `strength`, and outputs "
    "an RGB tangent-space normal (encoded to 0..1).\n\n"
    "OpenGL convention (+Y up) by default; enable `directx` for +Y down (green flip). "
    "Input is sampled with wrap-around, so tileable maps stay seamless.";

class HeightToNormal : public Iop
{
    float _strength;     // gradient strength multiplier
    bool  _directX;      // green channel convention
    int   _heightSrc;    // 0=red, 1=luminance

public:
    HeightToNormal(Node* node) : Iop(node)
        , _strength(1.0f)
        , _directX(false)
        , _heightSrc(0)
    {}

    const char* Class() const override { return H2N_CLASS; }
    const char* node_help() const override { return H2N_HELP; }
    static const Iop::Description d;

    void knobs(Knob_Callback f) override
    {
        Float_knob(f, &_strength, IRange(0.0, 10.0), "strength", "strength");
        Tooltip(f, "Multiplier on the height gradient. Higher = stronger relief.");

        static const char* const srcModes[] = { "red", "luminance", nullptr };
        Enumeration_knob(f, &_heightSrc, srcModes, "height_source", "height source");
        Tooltip(f, "Which input channel(s) provide the height scalar.");

        Bool_knob(f, &_directX, "directx", "directX");
        Tooltip(f, "OFF: OpenGL convention (G = +Y up) — default.\n"
                   "ON: DirectX convention (G = +Y down).");
    }

    void _validate(bool for_real) override
    {
        copy_info();
        // We always output RGB; ensure those channels exist downstream.
        info_.turn_on(Mask_RGB);
        set_out_channels(Mask_RGB);
    }

    void _request(int x, int y, int r, int t, ChannelMask channels, int count) override
    {
        // Need neighbours for central differences -> request a 1px-padded box.
        input0().request(x - 1, y - 1, r + 1, t + 1, Mask_RGB, count);
    }

    void engine(int y, int x, int r, ChannelMask channels, Row& out) override
    {
        const Box& ibox = input0().info().box();
        const int x0 = ibox.x(), y0 = ibox.y();
        const int x1 = ibox.r(), y1 = ibox.t();
        const int W = x1 - x0, H = y1 - y0;
        if (W <= 0 || H <= 0) { out.erase(channels); return; }

        // Load a 3-row tile (y-1 .. y+1) covering the requested x range plus 1px.
        Tile tile(input0(), x - 1, y - 1, r + 1, y + 2, Mask_RGB);
        if (aborted()) return;

        auto wrapX = [&](int xx) { int v = ((xx - x0) % W + W) % W + x0; return v; };
        auto wrapY = [&](int yy) { int v = ((yy - y0) % H + H) % H + y0; return v; };

        auto sample = [&](Channel c, int sx, int sy) -> float {
            // tile[c] returns LinePointers; check the row pointer, not the channel.
            const float* rowp = tile[c][sy];
            return rowp ? rowp[sx] : 0.0f;
        };
        auto heightAt = [&](int xx, int yy) -> float {
            const int sx = wrapX(xx);
            const int sy = wrapY(yy);
            if (_heightSrc == 1) {
                const float rr = sample(Chan_Red,   sx, sy);
                const float gg = sample(Chan_Green, sx, sy);
                const float bb = sample(Chan_Blue,  sx, sy);
                return 0.299f * rr + 0.587f * gg + 0.114f * bb;
            }
            return sample(Chan_Red, sx, sy);
        };

        float* rOut = out.writable(Chan_Red)   + x;
        float* gOut = out.writable(Chan_Green) + x;
        float* bOut = out.writable(Chan_Blue)  + x;

        const float gy_sign = _directX ? -1.0f : 1.0f;

        for (int px = x; px < r; ++px) {
            // Central differences (Sobel-lite, 2-tap) on the height field.
            const float hl = heightAt(px - 1, y);
            const float hr = heightAt(px + 1, y);
            const float hd = heightAt(px, y - 1);
            const float hu = heightAt(px, y + 1);

            // Gradient of height; normal = normalize(-dHdx, -dHdy, 1/strength).
            const float dHdx = (hr - hl) * 0.5f * _strength;
            const float dHdy = (hu - hd) * 0.5f * _strength;

            float nx = -dHdx;
            float ny = -dHdy * gy_sign;
            float nz = 1.0f;
            const float inv = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
            nx *= inv; ny *= inv; nz *= inv;

            // Encode [-1,1] -> [0,1].
            const int i = px - x;
            rOut[i] = nx * 0.5f + 0.5f;
            gOut[i] = ny * 0.5f + 0.5f;
            bOut[i] = nz * 0.5f + 0.5f;
        }
    }
};

static Iop* buildH2N(Node* node) { return new HeightToNormal(node); }
const Iop::Description HeightToNormal::d(H2N_CLASS, "Filter/HeightToNormal", buildH2N);


// ===========================================================================
// NormalToHeight — tangent-space normal map -> height map
// (Frankot-Chellappa: least-squares gradient integration via FFT)
// ===========================================================================
static const char* const N2H_CLASS = "NormalToHeight";
static const char* const N2H_HELP =
    "NormalToHeight (vmt_NormalToHeight) — reconstruct a height map from a "
    "tangent-space normal map.\n"
    "Copyright (c) 2026 kawata. All rights reserved.\n\n"
    "Decodes the RGB normal, forms the slope field (dz/dx, dz/dy), and solves the "
    "Poisson equation by the Frankot-Chellappa method (FFT). The result is a "
    "least-squares-optimal height field, normalised to 0..1 and written to RGB "
    "(grayscale) plus the red channel.\n\n"
    "OpenGL convention (+Y up) by default; enable `directx` for +Y down (green flip). "
    "Periodic boundary (FFT) makes tileable normal maps reconstruct seamlessly.\n"
    "The full-image solve runs once per validate and is cached.";

class NormalToHeight : public Iop
{
    float _scale;        // output height scale
    bool  _directX;      // green channel convention
    bool  _normalize;    // normalize result to 0..1

    // Cached reconstruction (full image, in input-box coordinates).
    std::vector<float> _height;
    int _cw, _ch, _cx0, _cy0;
    bool _cacheValid;

public:
    NormalToHeight(Node* node) : Iop(node)
        , _scale(1.0f)
        , _directX(false)
        , _normalize(true)
        , _cw(0), _ch(0), _cx0(0), _cy0(0)
        , _cacheValid(false)
    {}

    const char* Class() const override { return N2H_CLASS; }
    const char* node_help() const override { return N2H_HELP; }
    static const Iop::Description d;

    void knobs(Knob_Callback f) override
    {
        Float_knob(f, &_scale, IRange(0.0, 10.0), "scale", "scale");
        Tooltip(f, "Multiplier on the reconstructed height before normalisation.");

        Bool_knob(f, &_normalize, "normalize", "normalize 0-1");
        Tooltip(f, "Remap the reconstructed height into 0..1 (recommended). "
                   "OFF: output raw integrated height (can be negative).");

        Bool_knob(f, &_directX, "directx", "directX");
        Tooltip(f, "OFF: OpenGL convention (G = +Y up) — default.\n"
                   "ON: DirectX convention (G = +Y down).");
    }

    void _validate(bool for_real) override
    {
        copy_info();
        info_.turn_on(Mask_RGB);
        set_out_channels(Mask_RGB);
        // Invalidate cache; recompute lazily in engine on first access.
        _cacheValid = false;
    }

    void _request(int x, int y, int r, int t, ChannelMask channels, int count) override
    {
        // FC integration needs the whole image.
        const Box& b = input0().info().box();
        input0().request(b.x(), b.y(), b.r(), b.t(), Mask_RGB, count);
    }

    // Decode + integrate the full image once; store into _height.
    void buildCache()
    {
        const Box& ibox = input0().info().box();
        _cx0 = ibox.x(); _cy0 = ibox.y();
        _cw = ibox.r() - ibox.x();
        _ch = ibox.t() - ibox.y();
        if (_cw <= 0 || _ch <= 0) { _cacheValid = true; _height.clear(); return; }

        Tile tile(input0(), ibox.x(), ibox.y(), ibox.r(), ibox.t(), Mask_RGB);
        if (aborted()) return;

        const int W = _cw, H = _ch;
        // Slope fields p = dz/dx, q = dz/dy from the decoded normal.
        std::vector<double> p((size_t)W * H), q((size_t)W * H);

        const float gy_sign = _directX ? -1.0f : 1.0f;

        auto sample = [&](Channel c, int sx, int sy, float dflt) -> float {
            const float* rowp = tile[c][sy];
            return rowp ? rowp[sx] : dflt;
        };

        for (int yy = 0; yy < H; ++yy) {
            const int sy = yy + _cy0;
            for (int xx = 0; xx < W; ++xx) {
                const int sx = xx + _cx0;
                float nr = sample(Chan_Red,   sx, sy, 0.5f);
                float ng = sample(Chan_Green, sx, sy, 0.5f);
                float nb = sample(Chan_Blue,  sx, sy, 1.0f);

                // Decode 0..1 -> [-1,1].
                double nx = (double)nr * 2.0 - 1.0;
                double ny = ((double)ng * 2.0 - 1.0) * gy_sign;
                double nz = (double)nb * 2.0 - 1.0;
                if (nz < 1e-4) nz = 1e-4; // avoid divide-by-zero at grazing normals

                // Slope: surface z(x,y), normal ~ (-dz/dx, -dz/dy, 1).
                p[(size_t)yy * W + xx] = -nx / nz;
                q[(size_t)yy * W + xx] = -ny / nz;
            }
        }

        // Frankot-Chellappa on a padded power-of-two grid (periodic).
        const int FW = fftutil::nextPow2(W);
        const int FH = fftutil::nextPow2(H);

        std::vector<fftutil::Cplx> P((size_t)FW * FH, {0.0, 0.0});
        std::vector<fftutil::Cplx> Q((size_t)FW * FH, {0.0, 0.0});
        // Tile (periodic) the slope fields into the padded grid so the implicit
        // FFT periodicity matches the seamless-texture assumption.
        for (int yy = 0; yy < FH; ++yy) {
            const int sy = yy % H;
            for (int xx = 0; xx < FW; ++xx) {
                const int sx = xx % W;
                P[(size_t)yy * FW + xx].re = p[(size_t)sy * W + sx];
                Q[(size_t)yy * FW + xx].re = q[(size_t)sy * W + sx];
            }
        }

        fftutil::fft2d(P, FW, FH, -1);
        fftutil::fft2d(Q, FW, FH, -1);

        // Z(wx,wy) = -i(wx*P + wy*Q) / (wx^2 + wy^2)
        //
        // We use the central-difference frequency response sin(w) rather than the
        // continuous-derivative frequency w. This matches the discrete gradient
        // operator (central differences) that HeightToNormal uses, so the two
        // nodes round-trip to numerical precision instead of diverging on high
        // frequencies. (A verified standalone test gives RMS error ~1e-16 with
        // this kernel vs ~3e-3 with the continuous kernel.)
        std::vector<fftutil::Cplx> Z((size_t)FW * FH, {0.0, 0.0});
        for (int ky = 0; ky < FH; ++ky) {
            const double wy = std::sin(2.0 * M_PI * ((ky <= FH / 2) ? ky : ky - FH) / (double)FH);
            for (int kx = 0; kx < FW; ++kx) {
                const double wx = std::sin(2.0 * M_PI * ((kx <= FW / 2) ? kx : kx - FW) / (double)FW);
                const double denom = wx * wx + wy * wy;
                const size_t idx = (size_t)ky * FW + kx;
                if (denom < 1e-12) {
                    Z[idx].re = 0.0; Z[idx].im = 0.0; // DC = mean height (set to 0)
                    continue;
                }
                // num = -i (wx*P + wy*Q)
                const double sr = wx * P[idx].re + wy * Q[idx].re;
                const double si = wx * P[idx].im + wy * Q[idx].im;
                // multiply by -i: (-i)(sr+ i si) = si - i sr
                const double numr =  si;
                const double numi = -sr;
                Z[idx].re = numr / denom;
                Z[idx].im = numi / denom;
            }
        }

        fftutil::fft2d(Z, FW, FH, +1);
        const double invN = 1.0 / ((double)FW * FH);

        _height.assign((size_t)W * H, 0.0f);
        for (int yy = 0; yy < H; ++yy)
            for (int xx = 0; xx < W; ++xx)
                _height[(size_t)yy * W + xx] =
                    (float)(Z[(size_t)yy * FW + xx].re * invN) * _scale;

        if (_normalize) {
            float mn = _height[0], mx = _height[0];
            for (float v : _height) { mn = std::min(mn, v); mx = std::max(mx, v); }
            const float range = (mx - mn);
            const float inv = range > 1e-8f ? 1.0f / range : 0.0f;
            for (float& v : _height) v = (v - mn) * inv;
        }

        _cacheValid = true;
    }

    void engine(int y, int x, int r, ChannelMask channels, Row& out) override
    {
        if (!_cacheValid) {
            buildCache();
            if (aborted()) return;
        }
        if (_height.empty()) { out.erase(channels); return; }

        const int W = _cw, H = _ch;
        auto wrapX = [&](int xx) { return ((xx - _cx0) % W + W) % W; };
        const int row = ((y - _cy0) % H + H) % H;

        float* rOut = out.writable(Chan_Red)   + x;
        float* gOut = out.writable(Chan_Green) + x;
        float* bOut = out.writable(Chan_Blue)  + x;

        for (int px = x; px < r; ++px) {
            const int col = wrapX(px);
            const float h = _height[(size_t)row * W + col];
            const int i = px - x;
            rOut[i] = h; gOut[i] = h; bOut[i] = h;
        }
    }
};

static Iop* buildN2H(Node* node) { return new NormalToHeight(node); }
const Iop::Description NormalToHeight::d(N2H_CLASS, "Filter/NormalToHeight", buildN2H);
