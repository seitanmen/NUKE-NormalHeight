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

#include "DDImage/PlanarIop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/Tile.h"
#include "DDImage/Interest.h"
#include "DDImage/Blink.h"
#include "Blink/Blink.h"

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

// Blink kernel: height -> tangent-space normal via central differences.
// Ranged 2D access (±1 neighbours), edge-clamped. Params carry strength/sign/src.
static const char* const kH2NKernel =
"kernel HeightToNormalK : ImageComputationKernel<ePixelWise>\n"
"{\n"
"  Image<eRead, eAccessRanged2D, eEdgeClamped> src;\n"
"  Image<eWrite> dst;\n"
"  param:\n"
"    float k;          // gradient multiplier (strength * 200)\n"
"    float gySign;     // +1 OpenGL, -1 DirectX\n"
"    int   heightSrc;  // 0 = red, 1 = luminance\n"
"  void define() {\n"
"    defineParam(k, \"K\", 200.0f);\n"
"    defineParam(gySign, \"GySign\", 1.0f);\n"
"    defineParam(heightSrc, \"HeightSrc\", 0);\n"
"  }\n"
"  void init() {\n"
"    src.setRange(-1, -1, 1, 1);\n"
"  }\n"
"  float hAt(int dx, int dy) {\n"
"    if (heightSrc == 1)\n"
"      return 0.299f*src(dx,dy,0) + 0.587f*src(dx,dy,1) + 0.114f*src(dx,dy,2);\n"
"    return src(dx,dy,0);\n"
"  }\n"
"  void process() {\n"
"    float hl = hAt(-1,0), hr = hAt(1,0), hd = hAt(0,-1), hu = hAt(0,1);\n"
"    float dHdx = (hr - hl) * 0.5f * k;\n"
"    float dHdy = (hu - hd) * 0.5f * k;\n"
"    float nx = -dHdx;\n"
"    float ny = -dHdy * gySign;\n"
"    float nz = 1.0f;\n"
"    float inv = 1.0f / sqrt(nx*nx + ny*ny + nz*nz);\n"
"    nx *= inv; ny *= inv; nz *= inv;\n"
"    dst(0) = nx*0.5f + 0.5f;\n"
"    dst(1) = ny*0.5f + 0.5f;\n"
"    dst(2) = nz*0.5f + 0.5f;\n"
"    if (dst.kComps > 3) dst(3) = 1.0f;\n"
"  }\n"
"};\n";

class HeightToNormal : public PlanarIop
{
    float _strength;     // gradient strength multiplier
    bool  _directX;      // green channel convention
    int   _heightSrc;    // 0=red, 1=luminance
    bool  _useGPUIfAvailable;
    ::Blink::ComputeDevice _gpuDevice;
    ::Blink::ProgramSource _program;

public:
    HeightToNormal(Node* node) : PlanarIop(node)
        , _strength(1.0f)
        , _directX(false)
        , _heightSrc(0)
        , _useGPUIfAvailable(true)
        , _gpuDevice(::Blink::ComputeDevice::CurrentGPUDevice())
        , _program(kH2NKernel)
    {}

    const char* Class() const override { return H2N_CLASS; }
    const char* node_help() const override { return H2N_HELP; }
    static const Iop::Description d;

    void knobs(Knob_Callback f) override
    {
        Named_Text_knob(f, "info_desc", "",
                  "HeightToNormal — builds a tangent-space normal map from a "
                  "height map.");

        Float_knob(f, &_strength, IRange(-1.0, 1.0), "strength", "strength");
        Tooltip(f, "Relief strength. The slider runs -1..1 and maps linearly to a "
                   "gradient multiplier of -200..200 (slider 1.0 = x200, the "
                   "default). Negative values invert the relief (concave <-> convex).");
        Named_Text_knob(f, "h_strength", "",
                  "strength: relief amount, -1..1 -> gradient x(-200..200). "
                  "1 = default, 0 = flat, negative = inverted.");

        static const char* const srcModes[] = { "red", "luminance", nullptr };
        Enumeration_knob(f, &_heightSrc, srcModes, "height_source", "height source");
        Tooltip(f, "Which input channel(s) provide the height scalar.");
        Named_Text_knob(f, "h_source", "",
                  "height source: 'red' = R channel only, "
                  "'luminance' = Rec.601 weighted RGB.");

        Bool_knob(f, &_directX, "directx", "directX");
        Tooltip(f, "OFF: OpenGL convention (G = +Y up) — default.\n"
                   "ON: DirectX convention (G = +Y down).");
        Named_Text_knob(f, "h_directx", "",
                  "directX: green channel convention. OFF = OpenGL (+Y up), "
                  "ON = DirectX (+Y down).");

        Bool_knob(f, &_useGPUIfAvailable, "use_gpu", "use GPU if available");
        Tooltip(f, "Run the per-pixel normal computation on the GPU via Blink "
                   "when a GPU is available; otherwise it runs on the CPU. The "
                   "result is identical either way.");
        Named_Text_knob(f, "h_gpu", "",
                  "use GPU: GPU (Blink) when available, else CPU. Same result.");
    }

    void _validate(bool for_real) override
    {
        if (input0().node() == nullptr) { set_out_channels(Mask_None); return; }
        copy_info();
        info_.turn_on(Mask_RGB);
        set_out_channels(Mask_RGB);
    }

    void getRequests(const Box& box, const ChannelSet& channels, int count,
                     RequestOutput& reqData) const override
    {
        Box inBox = box;
        inBox.expand(1);                         // central differences need ±1
        inBox.intersect(input0().info());
        reqData.request(&input0(), inBox, Mask_RGB, count);
    }

    bool useStripes() const override { return false; }
    bool renderFullPlanes() const override { return true; }

    // CPU fallback (and reference). Central differences with edge clamp.
    void renderCPU(ImagePlane& out, const ImagePlane& in, const Box& b)
    {
        const int x0 = b.x(), x1 = b.r(), y0 = b.y(), y1 = b.t();
        const int nComp = out.channels().size();
        auto clampX = [&](int x){ return x < x0 ? x0 : (x >= x1 ? x1 - 1 : x); };
        auto clampY = [&](int y){ return y < y0 ? y0 : (y >= y1 ? y1 - 1 : y); };
        auto hAt = [&](int x, int y) -> float {
            const int sx = clampX(x), sy = clampY(y);
            if (_heightSrc == 1)
                return 0.299f*in.at(sx,sy,0) + 0.587f*in.at(sx,sy,1) + 0.114f*in.at(sx,sy,2);
            return in.at(sx, sy, 0);
        };
        const float gy = _directX ? -1.0f : 1.0f;
        const float k = _strength * 200.0f;
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                float dHdx = (hAt(x+1,y) - hAt(x-1,y)) * 0.5f * k;
                float dHdy = (hAt(x,y+1) - hAt(x,y-1)) * 0.5f * k;
                float nx = -dHdx, ny = -dHdy * gy, nz = 1.0f;
                float inv = 1.0f / std::sqrt(nx*nx + ny*ny + nz*nz);
                nx *= inv; ny *= inv; nz *= inv;
                out.writableAt(x, y, 0) = nx*0.5f + 0.5f;
                if (nComp > 1) out.writableAt(x, y, 1) = ny*0.5f + 0.5f;
                if (nComp > 2) out.writableAt(x, y, 2) = nz*0.5f + 0.5f;
                if (nComp > 3) out.writableAt(x, y, 3) = 1.0f;
            }
    }

    void renderStripe(ImagePlane& outputPlane) override
    {
        if (input0().node() == nullptr) { outputPlane.makeWritable(); return; }

        Box inputBox = outputPlane.bounds();
        inputBox.expand(1);
        inputBox.intersect(input0().info());
        const int W = inputBox.r() - inputBox.x();
        const int H = inputBox.t() - inputBox.y();

        ImagePlane inputPlane(inputBox, outputPlane.packed(),
                              outputPlane.channels(), outputPlane.nComps());
        input0().fetchPlane(inputPlane);
        if (aborted()) return;

        outputPlane.makeWritable();

        // Degenerate input (e.g. 1x1 black from an unconnected upstream): black out.
        if (W < 2 || H < 2) {
            const Box& b = outputPlane.bounds();
            const int nc = outputPlane.channels().size();
            for (int y = b.y(); y < b.t(); ++y)
                for (int x = b.x(); x < b.r(); ++x)
                    for (int c = 0; c < nc; ++c) outputPlane.writableAt(x, y, c) = 0.0f;
            return;
        }

        bool done = false;
        try {
            ::Blink::Image outImg, inImg;
            if (DD::Image::Blink::ImagePlaneAsBlinkImage(outputPlane, outImg) &&
                DD::Image::Blink::ImagePlaneAsBlinkImage(inputPlane, inImg)) {
                bool usingGPU = _useGPUIfAvailable && _gpuDevice.available();
                ::Blink::ComputeDevice dev =
                    usingGPU ? _gpuDevice : ::Blink::ComputeDevice::CurrentCPUDevice();

                ::Blink::Image inOnDev  = inImg.distributeTo(dev);
                ::Blink::Image outOnDev = usingGPU ? outImg.makeLike(_gpuDevice) : outImg;

                ::Blink::ComputeDeviceBinder binder(dev);
                std::vector< ::Blink::Image > images;
                images.push_back(inOnDev);
                images.push_back(outOnDev);
                ::Blink::Kernel kernel(_program, dev, images, kBlinkCodegenDefault);
                kernel.setParamValue("K", _strength * 200.0f);
                kernel.setParamValue("GySign", _directX ? -1.0f : 1.0f);
                kernel.setParamValue("HeightSrc", _heightSrc);
                kernel.iterate();
                if (usingGPU) outImg.copyFrom(outOnDev);
                done = true;
            }
        }
        catch (::Blink::Exception&) {
            done = false;   // fall through to CPU
        }
        if (!done) renderCPU(outputPlane, inputPlane, inputBox);
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

class NormalToHeight : public PlanarIop
{
    float _strength;     // scales the input normal's tangent (slope) -1..1
    float _maxSlope;     // clip |p|,|q| to this to kill grazing-normal spikes
    float _clip;         // percentile trimmed from each end when normalizing (fixed)
    bool  _directX;      // green channel convention

    // Cached reconstruction (full image, in input-box coordinates).
    std::vector<float> _height;
    int _cw, _ch, _cx0, _cy0;
    bool _cacheValid;

public:
    NormalToHeight(Node* node) : PlanarIop(node)
        , _strength(1.0f)
        , _maxSlope(5.0f)
        , _clip(0.00001f)
        , _directX(false)
        , _cw(0), _ch(0), _cx0(0), _cy0(0)
        , _cacheValid(false)
    {}

    const char* Class() const override { return N2H_CLASS; }
    const char* node_help() const override { return N2H_HELP; }
    static const Iop::Description d;

    void knobs(Knob_Callback f) override
    {
        Named_Text_knob(f, "info_desc", "",
                  "NormalToHeight — reconstructs a height map from a tangent-space "
                  "normal map (Frankot-Chellappa FFT integration).");

        Float_knob(f, &_strength, IRange(-1.0, 1.0), "strength", "strength");
        Tooltip(f, "Scales the input normal's tangent (X/Y) components in normal "
                   "space before integration, i.e. how strongly the surface tilts. "
                   "1.0 = use the normal as-is (default). Lower values flatten the "
                   "relief; negative values invert it. Range -1..1.");
        Named_Text_knob(f, "h_strength", "",
                  "strength: tilt scale on the input normal (-1..1). 1 = as-is, "
                  "0 = flat, negative = inverted. Output is renormalised, so a "
                  "uniform positive value mainly affects sign/balance.");

        Float_knob(f, &_maxSlope, IRange(1.0, 50.0), "max_slope", "max slope");
        Tooltip(f, "Clips the per-pixel slope |dz/dx|,|dz/dy| before integration. "
                   "Stray near-grazing normals (nz~0, e.g. a lone RGB=255/0 dot) "
                   "make the slope explode and the FFT turns it into a sharp "
                   "white/black peak. Lower this if you see such spikes (3-10 is a "
                   "good range); raise it to allow steeper walls. Default 5.");
        Named_Text_knob(f, "h_maxslope", "",
                  "max slope: clips runaway slopes from near-grazing normals "
                  "(nz~0) that would become white/black spikes. Lower (3-10) to "
                  "suppress spikes, raise for steeper walls. Default 5.");

        Bool_knob(f, &_directX, "directx", "directX");
        Tooltip(f, "OFF: OpenGL convention (G = +Y up) — default.\n"
                   "ON: DirectX convention (G = +Y down).");
        Named_Text_knob(f, "h_directx", "",
                  "directX: green channel convention. OFF = OpenGL (+Y up), "
                  "ON = DirectX (+Y down). Match your input normal map.");
    }

    void _validate(bool for_real) override
    {
        // Guard against an unconnected input (copy_info() would crash on a
        // null source Op). Leave the base Iop's default (black) info untouched.
        if (input0().node() == nullptr) {
            set_out_channels(Mask_None);
            _height.clear();
            return;
        }
        copy_info();
        info_.turn_on(Mask_RGB);
        set_out_channels(Mask_RGB);
    }

    void getRequests(const Box& box, const ChannelSet& channels, int count,
                     RequestOutput& reqData) const override
    {
        // FC integration needs the whole image.
        const Box& b = input0().info().box();
        reqData.request(&input0(), b, Mask_RGB, count);
    }

    bool useStripes() const override { return false; }
    bool renderFullPlanes() const override { return true; }

    void renderStripe(ImagePlane& outputPlane) override
    {
        if (input0().node() == nullptr) { outputPlane.makeWritable(); return; }

        const Box& ibox = input0().info().box();
        ImagePlane inputPlane(ibox, outputPlane.packed(),
                              outputPlane.channels(), outputPlane.nComps());
        input0().fetchPlane(inputPlane);
        if (aborted()) return;

        buildHeight(inputPlane, ibox);

        outputPlane.makeWritable();
        const Box& ob = outputPlane.bounds();
        const int nc = outputPlane.channels().size();
        if (_height.empty()) {
            for (int y = ob.y(); y < ob.t(); ++y)
                for (int x = ob.x(); x < ob.r(); ++x)
                    for (int c = 0; c < nc; ++c) outputPlane.writableAt(x, y, c) = 0.0f;
            return;
        }
        const int W = _cw, H = _ch;
        for (int y = ob.y(); y < ob.t(); ++y) {
            const int row = ((y - _cy0) % H + H) % H;
            for (int x = ob.x(); x < ob.r(); ++x) {
                const int col = ((x - _cx0) % W + W) % W;
                const float h = _height[(size_t)row * W + col];
                for (int c = 0; c < nc; ++c) outputPlane.writableAt(x, y, c) = (c < 3) ? h : 1.0f;
            }
        }
    }

    // Decode + integrate the full image once; store into _height.
    void buildHeight(const ImagePlane& in, const Box& ibox)
    {
        _cx0 = ibox.x(); _cy0 = ibox.y();
        _cw = ibox.r() - ibox.x();
        _ch = ibox.t() - ibox.y();
        if (_cw < 2 || _ch < 2) { _height.clear(); return; }

        const int W = _cw, H = _ch;
        const int nComp = in.channels().size();

        // Slope fields p = dz/dx, q = dz/dy from the decoded normal.
        std::vector<double> p((size_t)W * H), q((size_t)W * H);

        const float gy_sign = _directX ? -1.0f : 1.0f;

        for (int yy = 0; yy < H; ++yy) {
            const int sy = yy + _cy0;
            for (int xx = 0; xx < W; ++xx) {
                const int sx = xx + _cx0;
                float nr = in.at(sx, sy, 0);
                float ng = nComp > 1 ? in.at(sx, sy, 1) : 0.5f;
                float nb = nComp > 2 ? in.at(sx, sy, 2) : 1.0f;

                // Decode 0..1 -> [-1,1]. The tangent (X/Y) components are scaled
                // by 'strength' in normal space (sign inverts, 0 flattens; a
                // uniform positive scale is absorbed by the 0..1 normalisation).
                double nx = ((double)nr * 2.0 - 1.0) * (double)_strength;
                double ny = ((double)ng * 2.0 - 1.0) * gy_sign * (double)_strength;
                double nz = (double)nb * 2.0 - 1.0;
                if (nz < 1e-4) nz = 1e-4; // avoid divide-by-zero at grazing normals

                // Slope: surface z(x,y), normal ~ (-dz/dx, -dz/dy, 1).
                double pv = -nx / nz;
                double qv = -ny / nz;

                // Clip the slope. A near-grazing normal (nz ~ 0, e.g. a stray
                // RGB=255/0 dot in the source) makes the slope blow up toward
                // infinity, which the FFT integrator turns into a sharp white/black
                // impulse (the "peak artifact"). Limiting |p|,|q| to max_slope
                // suppresses those spikes; max_slope is also the tallest wall the
                // height field can represent (a height map cannot encode an
                // overhang / true vertical face).
                const double ms = (double)_maxSlope;
                if (pv >  ms) pv =  ms; else if (pv < -ms) pv = -ms;
                if (qv >  ms) qv =  ms; else if (qv < -ms) qv = -ms;

                p[(size_t)yy * W + xx] = pv;
                q[(size_t)yy * W + xx] = qv;
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

        // Solve the Poisson equation in the frequency domain using the FORWARD
        // difference operator, to exactly match HeightToNormal's forward-difference
        // gradient. For forward differences the per-axis frequency response is
        //     D(w) = e^{i w} - 1                      (w = 2*pi*k/N)
        // and the least-squares height is
        //     Z = (conj(Dx)*P + conj(Dy)*Q) / (|Dx|^2 + |Dy|^2),
        // with |D(w)|^2 = 2(1 - cos w). This kernel has NO zero at the Nyquist
        // frequency (unlike the central-difference sin(w) kernel), so the even/odd
        // sub-grids stay coupled and the 1-pixel grid grain disappears.
        std::vector<fftutil::Cplx> Z((size_t)FW * FH, {0.0, 0.0});
        for (int ky = 0; ky < FH; ++ky) {
            const double ay = 2.0 * M_PI * ((ky <= FH / 2) ? ky : ky - FH) / (double)FH;
            // conj(Dy) = e^{-i ay} - 1 = (cos ay - 1) - i sin ay
            const double dyr = std::cos(ay) - 1.0;
            const double dyi = -std::sin(ay);
            for (int kx = 0; kx < FW; ++kx) {
                const double ax = 2.0 * M_PI * ((kx <= FW / 2) ? kx : kx - FW) / (double)FW;
                const double dxr = std::cos(ax) - 1.0;
                const double dxi = -std::sin(ax);

                // |Dx|^2 + |Dy|^2 = 2(1-cos ax) + 2(1-cos ay)
                const double denom = (dxr * dxr + dxi * dxi) + (dyr * dyr + dyi * dyi);
                const size_t idx = (size_t)ky * FW + kx;
                if (denom < 1e-12) {
                    Z[idx].re = 0.0; Z[idx].im = 0.0; // DC = mean height (set to 0)
                    continue;
                }
                // num = conj(Dx)*P + conj(Dy)*Q   (complex multiply-add)
                const double numr = (dxr * P[idx].re - dxi * P[idx].im)
                                  + (dyr * Q[idx].re - dyi * Q[idx].im);
                const double numi = (dxr * P[idx].im + dxi * P[idx].re)
                                  + (dyr * Q[idx].im + dyi * Q[idx].re);
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
                    (float)(Z[(size_t)yy * FW + xx].re * invN);

        {
            // Always normalise to 0..1 with robust (percentile) min/max. The
            // Frankot-Chellappa solve can produce a few overshoot/ringing pixels
            // at sharp edges; using the absolute min/max would let those outliers
            // dominate the range and crush the mid-tones. Map the low/high
            // percentile points to 0/1 so the bulk of the image uses full contrast.
            const size_t n = _height.size();
            std::vector<float> sorted(_height);
            std::sort(sorted.begin(), sorted.end());

            // _clip is the fraction trimmed from EACH end (e.g. 0.001 = 0.1%).
            float clip = _clip;
            if (clip < 0.0f) clip = 0.0f;
            if (clip > 0.49f) clip = 0.49f;

            size_t loIdx = (size_t)(clip * (double)(n - 1));
            size_t hiIdx = (size_t)((1.0 - clip) * (double)(n - 1));
            if (hiIdx <= loIdx) { loIdx = 0; hiIdx = n - 1; }

            const float mn = sorted[loIdx];
            const float mx = sorted[hiIdx];
            const float range = (mx - mn);
            const float inv = range > 1e-8f ? 1.0f / range : 0.0f;
            for (float& v : _height) {
                float nv = (v - mn) * inv;     // map [lo,hi] percentile -> [0,1]
                if (nv < 0.0f) nv = 0.0f;       // clamp the trimmed outliers
                if (nv > 1.0f) nv = 1.0f;
                v = nv;
            }
        }

    }
};

static Iop* buildN2H(Node* node) { return new NormalToHeight(node); }
const Iop::Description NormalToHeight::d(N2H_CLASS, "Filter/NormalToHeight", buildN2H);
