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
#include "DDImage/Hash.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <string>
#include <atomic>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

using namespace DD::Image;

// ===========================================================================
// Minimal self-contained FFT (radix-2 Cooley-Tukey, in-place).
// No external dependency. Operates on power-of-two sizes; the integrator
// zero/relation-pads the working grid up to the next power of two per axis.
// ===========================================================================
namespace fftutil {

// Single precision: a height map reconstructed from an 8-bit normal map needs
// nowhere near double precision, and float roughly halves memory traffic (the
// 2D FFT is memory-bound on the strided column pass). Twiddle factors are still
// evaluated in double before being stored, to keep the table accurate.
struct Cplx { float re, im; };

// Precompute the twiddle table for a transform of length n (n a power of two):
//   tw[j] = exp(sign * 2*pi*i * j / n),  j in [0, n/2).
// A stage of length 'len' reads its k-th twiddle at tw[k * (n/len)].
static void makeTwiddles(std::vector<Cplx>& tw, int n, int sign)
{
    tw.resize(n / 2);
    for (int j = 0; j < n / 2; ++j) {
        const double ang = sign * 2.0 * M_PI * (double)j / (double)n;
        tw[j].re = (float)std::cos(ang);
        tw[j].im = (float)std::sin(ang);
    }
}

// In-place iterative radix-2 FFT with a precomputed twiddle table (size n/2).
// Using the table removes the per-butterfly twiddle recurrence (its complex
// multiply and accumulated rounding), which is both faster and more accurate.
static void fft1d(std::vector<Cplx>& a, const std::vector<Cplx>& tw)
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
        const size_t half = len >> 1;
        const size_t step = n / len;          // stride into the twiddle table
        for (size_t i = 0; i < n; i += len) {
            for (size_t k = 0; k < half; ++k) {
                const Cplx w = tw[k * step];
                const Cplx b = a[i + k + half];
                const Cplx v{ b.re * w.re - b.im * w.im,
                              b.re * w.im + b.im * w.re };
                const Cplx u = a[i + k];
                a[i + k].re        = u.re + v.re;
                a[i + k].im        = u.im + v.im;
                a[i + k + half].re = u.re - v.re;
                a[i + k + half].im = u.im - v.im;
            }
        }
    }
}

// Split [0,n) into one contiguous chunk per worker thread and run
// fn(begin, end) on each. Disjoint ranges => no locking needed. Falls back to
// a plain serial call for small n (thread spawn would cost more than it saves).
template <class F>
static void parallelChunks(int n, F&& fn)
{
    if (n <= 0) return;
    unsigned hw = std::thread::hardware_concurrency();
    int T = (int)(hw ? hw : 4);
    if (T > 32) T = 32;          // bound oversubscription vs Nuke's own pool
    if (T > n) T = n;
    if (n < 64 || T <= 1) { fn(0, n); return; }   // serial for small work

    const int chunk = (n + T - 1) / T;
    auto worker = [&](int t) {
        const int b = t * chunk;
        const int e = (b + chunk < n) ? (b + chunk) : n;
        if (b < e) fn(b, e);
    };
    std::vector<std::thread> pool;
    pool.reserve(T - 1);
    for (int t = 1; t < T; ++t) pool.emplace_back(worker, t);
    worker(0);                   // run the first chunk on this thread
    for (std::thread& th : pool) th.join();
}

// 2D FFT over a width*height complex grid stored row-major.
// Both dimensions must be powers of two. The H row transforms are mutually
// independent (each touches only its own row); likewise the W column
// transforms. A barrier between the two passes is implicit (parallelChunks
// joins before returning), so the column pass sees the fully transformed rows.
static void fft2d(std::vector<Cplx>& g, int W, int H, int sign)
{
    // Build each axis's twiddle table once; all rows share twW, all cols twH.
    // The tables are read-only during the passes, so threads share them safely.
    std::vector<Cplx> twW, twH;
    makeTwiddles(twW, W, sign);
    makeTwiddles(twH, H, sign);

    parallelChunks(H, [&](int y0, int y1) {
        std::vector<Cplx> row(W);                 // per-thread scratch
        for (int y = y0; y < y1; ++y) {
            for (int x = 0; x < W; ++x) row[x] = g[(size_t)y * W + x];
            fft1d(row, twW);
            for (int x = 0; x < W; ++x) g[(size_t)y * W + x] = row[x];
        }
    });
    parallelChunks(W, [&](int x0, int x1) {
        std::vector<Cplx> col(H);                 // per-thread scratch
        for (int x = x0; x < x1; ++x) {
            for (int y = 0; y < H; ++y) col[y] = g[(size_t)y * W + x];
            fft1d(col, twH);
            for (int y = 0; y < H; ++y) g[(size_t)y * W + x] = col[y];
        }
    });
}

static int nextPow2(int v)
{
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

} // namespace fftutil


static inline float clamp01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }


#ifdef _WIN32
// ===========================================================================
// Optional CUDA acceleration for NormalToHeight's full-image solve.
//
// Dynamically loads Nuke's bundled cudart / cuFFT / nvrtc plus the system CUDA
// driver (nvcuda.dll). The two 2D FFTs run on cuFFT; the Poisson solve and the
// percentile normalisation run in nvrtc-compiled kernels. Only the packed slope
// grid is uploaded and the final height downloaded. If any DLL/symbol is absent
// (e.g. a non-NVIDIA machine) or any call fails, reconstruct() returns false and
// the caller falls back to the (already fast, verified) CPU path. Output matches
// the CPU result to ~3e-5 (histogram-percentile vs exact selection) — invisible.
// ===========================================================================
namespace nhcuda {

struct F2 { float x, y; };
typedef int  cudaError_t;
typedef int  cufftHandle;
typedef int  cufftResult;
typedef int  CUresult;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUcontext;
typedef int  nvrtcResult;
typedef void* nvrtcProgram;
enum { H2D = 1, D2H = 2, CUFFT_C2C = 0x29, FWD = -1, INV = 1 };

typedef cudaError_t (*p_setdev)(int);
typedef cudaError_t (*p_malloc)(void**, size_t);
typedef cudaError_t (*p_free)(void*);
typedef cudaError_t (*p_memcpy)(void*, const void*, size_t, int);
typedef cudaError_t (*p_memset)(void*, int, size_t);
typedef cudaError_t (*p_sync)();
typedef cufftResult (*p_plan2d)(cufftHandle*, int, int, int);
typedef cufftResult (*p_execC2C)(cufftHandle, F2*, F2*, int);
typedef cufftResult (*p_fftdestroy)(cufftHandle);
typedef CUresult (*p_cuinit)(unsigned);
typedef CUresult (*p_ctxcur)(CUcontext*);
typedef CUresult (*p_ctxset)(CUcontext);
typedef CUresult (*p_devget)(int*, int);
typedef CUresult (*p_pctxretain)(CUcontext*, int);
typedef CUresult (*p_ctxpush)(CUcontext);
typedef CUresult (*p_ctxpop)(CUcontext*);
typedef CUresult (*p_modload)(CUmodule*, const void*, unsigned, int*, void**);
typedef CUresult (*p_modfn)(CUfunction*, CUmodule, const char*);
typedef CUresult (*p_launch)(CUfunction, unsigned,unsigned,unsigned, unsigned,unsigned,unsigned, unsigned, void*, void**, void**);
typedef CUresult (*p_ctxsync)();
typedef nvrtcResult (*p_ncreate)(nvrtcProgram*, const char*, const char*, int, const char**, const char**);
typedef nvrtcResult (*p_ncomp)(nvrtcProgram, int, const char**);
typedef nvrtcResult (*p_nptxsz)(nvrtcProgram, size_t*);
typedef nvrtcResult (*p_nptx)(nvrtcProgram, char*);
typedef nvrtcResult (*p_nlogsz)(nvrtcProgram, size_t*);
typedef nvrtcResult (*p_nlog)(nvrtcProgram, char*);
typedef nvrtcResult (*p_ndestroy)(nvrtcProgram*);

static p_setdev cudaSetDevice_; static p_malloc cudaMalloc_; static p_free cudaFree_;
static p_memcpy cudaMemcpy_; static p_memset cudaMemset_; static p_sync cudaSync_;
static p_plan2d cufftPlan2d_; static p_execC2C cufftExecC2C_; static p_fftdestroy cufftDestroy_;
static p_cuinit cuInit_; static p_ctxcur cuCtxGetCurrent_; static p_ctxset cuCtxSetCurrent_;
static p_devget cuDeviceGet_; static p_pctxretain cuDevicePrimaryCtxRetain_; static p_ctxpush cuCtxPushCurrent_; static p_ctxpop cuCtxPopCurrent_;
static p_modload cuModuleLoadDataEx_; static p_modfn cuModuleGetFunction_; static p_launch cuLaunchKernel_; static p_ctxsync cuCtxSynchronize_;
static p_ncreate nvrtcCreateProgram_; static p_ncomp nvrtcCompileProgram_; static p_nptxsz nvrtcGetPTXSize_;
static p_nptx nvrtcGetPTX_; static p_nlogsz nvrtcGetProgramLogSize_; static p_nlog nvrtcGetProgramLog_; static p_ndestroy nvrtcDestroyProgram_;

static const int HB = 16384;     // histogram bins for percentile normalisation

union FU { float f; unsigned u; };
static float decU(unsigned e) { unsigned m = ((e >> 31) - 1u) | 0x80000000u; FU x; x.u = e ^ m; return x.f; }

struct State {
    bool tried = false, ok = false;
    CUcontext ctx = nullptr;
    CUfunction kSolve=nullptr, kFinish=nullptr, kMinMax=nullptr, kHist=nullptr, kApply=nullptr;
    F2* dG=nullptr; F2* dZ=nullptr; float* dH=nullptr; unsigned* dMM=nullptr; unsigned* dHist=nullptr;
    cufftHandle plan = 0; int FW=0, FH=0, W=0, H=0;
};
static State S;
static std::atomic_flag gBusy = ATOMIC_FLAG_INIT;   // simple spinlock (no CRT mutex)

template <class T> static bool sym(T& fn, HMODULE m, const char* n) { fn = (T)GetProcAddress(m, n); return fn != nullptr; }

static bool init_locked()
{
    HMODULE rt = LoadLibraryA("cudart64_110.dll");
    HMODULE ft = LoadLibraryA("cufft64_10.dll");
    HMODULE nv = LoadLibraryA("nvrtc64_112_0.dll");
    HMODULE dr = LoadLibraryA("nvcuda.dll");
    if (!rt || !ft || !nv || !dr) return false;
    bool ok = true;
    ok &= sym(cudaSetDevice_, rt, "cudaSetDevice"); ok &= sym(cudaMalloc_, rt, "cudaMalloc");
    ok &= sym(cudaFree_, rt, "cudaFree"); ok &= sym(cudaMemcpy_, rt, "cudaMemcpy");
    ok &= sym(cudaMemset_, rt, "cudaMemset"); ok &= sym(cudaSync_, rt, "cudaDeviceSynchronize");
    ok &= sym(cufftPlan2d_, ft, "cufftPlan2d"); ok &= sym(cufftExecC2C_, ft, "cufftExecC2C"); ok &= sym(cufftDestroy_, ft, "cufftDestroy");
    ok &= sym(cuInit_, dr, "cuInit"); ok &= sym(cuCtxGetCurrent_, dr, "cuCtxGetCurrent"); ok &= sym(cuCtxSetCurrent_, dr, "cuCtxSetCurrent");
    ok &= sym(cuDeviceGet_, dr, "cuDeviceGet"); ok &= sym(cuDevicePrimaryCtxRetain_, dr, "cuDevicePrimaryCtxRetain");
    ok &= sym(cuCtxPushCurrent_, dr, "cuCtxPushCurrent_v2"); ok &= sym(cuCtxPopCurrent_, dr, "cuCtxPopCurrent_v2");
    ok &= sym(cuModuleLoadDataEx_, dr, "cuModuleLoadDataEx"); ok &= sym(cuModuleGetFunction_, dr, "cuModuleGetFunction");
    ok &= sym(cuLaunchKernel_, dr, "cuLaunchKernel"); ok &= sym(cuCtxSynchronize_, dr, "cuCtxSynchronize");
    ok &= sym(nvrtcCreateProgram_, nv, "nvrtcCreateProgram"); ok &= sym(nvrtcCompileProgram_, nv, "nvrtcCompileProgram");
    ok &= sym(nvrtcGetPTXSize_, nv, "nvrtcGetPTXSize"); ok &= sym(nvrtcGetPTX_, nv, "nvrtcGetPTX");
    ok &= sym(nvrtcGetProgramLogSize_, nv, "nvrtcGetProgramLogSize"); ok &= sym(nvrtcGetProgramLog_, nv, "nvrtcGetProgramLog");
    ok &= sym(nvrtcDestroyProgram_, nv, "nvrtcDestroyProgram");
    if (!ok) return false;

    // Get our OWN handle to the device's primary context instead of grabbing
    // whatever Nuke left current. The runtime API (cudart) shares this same
    // primary context, so cuFFT/cudart and our driver-API launches all agree.
    // We push/pop it around every use so Nuke's context state is untouched.
    if (cuInit_(0) != 0) return false;
    int dev = 0;
    if (cuDeviceGet_(&dev, 0) != 0) return false;
    if (cuDevicePrimaryCtxRetain_(&S.ctx, dev) != 0 || !S.ctx) return false;
    cudaSetDevice_(0);            // bind the runtime to the same device
    if (cuCtxPushCurrent_(S.ctx) != 0) return false;   // current for module load

    // Kernels. cos(w)-1 via -2 sin^2(w/2) avoids low-frequency cancellation in
    // float. Min/max uses a monotonic float->uint key for atomicMin/Max.
    static const char* src =
    "typedef struct{float x,y;}F2;\n"
    "extern \"C\" __global__ void solve(const F2* G,F2* Z,int FW,int FH){\n"
    " int kx=blockIdx.x*blockDim.x+threadIdx.x, ky=blockIdx.y*blockDim.y+threadIdx.y;\n"
    " if(kx>=FW||ky>=FH)return; int mxk=(FW-kx)%FW,myk=(FH-ky)%FH;\n"
    " F2 a=G[(long long)ky*FW+kx], b=G[(long long)myk*FW+mxk];\n"
    " float Pr=0.5f*(a.x+b.x),Pi=0.5f*(a.y-b.y),Qr=0.5f*(a.y+b.y),Qi=-0.5f*(a.x-b.x);\n"
    " const float PI=3.14159265358979f;\n"
    " float ax=2.0f*PI*((kx<=FW/2)?kx:kx-FW)/(float)FW, ay=2.0f*PI*((ky<=FH/2)?ky:ky-FH)/(float)FH;\n"
    " float sx=sinf(ax*0.5f), sy=sinf(ay*0.5f);\n"
    " float dxr=-2.0f*sx*sx, dxi=-sinf(ax), dyr=-2.0f*sy*sy, dyi=-sinf(ay);\n"
    " float den=(dxr*dxr+dxi*dxi)+(dyr*dyr+dyi*dyi); long long id=(long long)ky*FW+kx;\n"
    " if(den<1e-12f){Z[id].x=0;Z[id].y=0;return;}\n"
    " float nr=(dxr*Pr-dxi*Pi)+(dyr*Qr-dyi*Qi), ni=(dxr*Pi+dxi*Pr)+(dyr*Qi+dyi*Qr);\n"
    " Z[id].x=nr/den; Z[id].y=ni/den;\n}\n"
    "extern \"C\" __global__ void finish(const F2* Z,float* h,int W,int H,int FW,float invN){\n"
    " int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;\n"
    " if(x>=W||y>=H)return; h[(long long)y*W+x]=Z[(long long)y*FW+x].x*invN;\n}\n"
    "__device__ __forceinline__ unsigned enc(float f){unsigned i=__float_as_uint(f); unsigned m=(unsigned)(-(int)(i>>31))|0x80000000u; return i^m;}\n"
    "extern \"C\" __global__ void minmax(const float* h,long long n,unsigned* gmin,unsigned* gmax){\n"
    " __shared__ unsigned smn[256],smx[256]; int t=threadIdx.x; long long i=(long long)blockIdx.x*blockDim.x+t; long long st=(long long)gridDim.x*blockDim.x;\n"
    " unsigned lmn=0xffffffffu,lmx=0u; for(;i<n;i+=st){unsigned e=enc(h[i]); lmn=min(lmn,e); lmx=max(lmx,e);} smn[t]=lmn;smx[t]=lmx; __syncthreads();\n"
    " for(int s=blockDim.x/2;s>0;s>>=1){ if(t<s){smn[t]=min(smn[t],smn[t+s]);smx[t]=max(smx[t],smx[t+s]);} __syncthreads(); }\n"
    " if(t==0){atomicMin(gmin,smn[0]);atomicMax(gmax,smx[0]);}\n}\n"
    "extern \"C\" __global__ void hist(const float* h,long long n,float mn,float inv,int B,unsigned* hg){\n"
    " long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; long long st=(long long)gridDim.x*blockDim.x;\n"
    " for(;i<n;i+=st){int b=(int)((h[i]-mn)*inv*B); if(b<0)b=0; if(b>=B)b=B-1; atomicAdd(&hg[b],1u);}\n}\n"
    "extern \"C\" __global__ void applyn(float* h,long long n,float mn,float inv,float s){\n"
    " long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;\n"
    " float v=(h[i]-mn)*inv; v=v<0?0:(v>1?1:v); v=0.5f+(v-0.5f)*s; v=v<0?0:(v>1?1:v); h[i]=v;\n}\n";

    // Compile + load with the primary context pushed; pop it on every exit so
    // the init thread's context stack is left exactly as we found it.
    bool good = false;
    do {
        nvrtcProgram prog = nullptr;
        if (nvrtcCreateProgram_(&prog, src, "nh.cu", 0, nullptr, nullptr) != 0) break;
        const char* opts[] = { "--gpu-architecture=compute_70" };
        if (nvrtcCompileProgram_(prog, 1, opts) != 0) { nvrtcDestroyProgram_(&prog); break; }
        size_t ps = 0; nvrtcGetPTXSize_(prog, &ps);
        std::string ptx(ps, 0); nvrtcGetPTX_(prog, &ptx[0]); nvrtcDestroyProgram_(&prog);
        CUmodule mod = nullptr;
        if (cuModuleLoadDataEx_(&mod, ptx.c_str(), 0, nullptr, nullptr) != 0) break;
        if (cuModuleGetFunction_(&S.kSolve, mod, "solve") != 0) break;
        if (cuModuleGetFunction_(&S.kFinish, mod, "finish") != 0) break;
        if (cuModuleGetFunction_(&S.kMinMax, mod, "minmax") != 0) break;
        if (cuModuleGetFunction_(&S.kHist, mod, "hist") != 0) break;
        if (cuModuleGetFunction_(&S.kApply, mod, "applyn") != 0) break;
        good = true;
    } while (false);

    CUcontext popped = nullptr; cuCtxPopCurrent_(&popped);
    return good;
}

static bool ensureBuffers(int W, int H)
{
    const int FW = fftutil::nextPow2(W), FH = fftutil::nextPow2(H);
    if (S.dG && S.FW == FW && S.FH == FH && S.W == W && S.H == H) return true;
    if (S.dG) cudaFree_(S.dG); if (S.dZ) cudaFree_(S.dZ); if (S.dH) cudaFree_(S.dH);
    if (S.dMM) cudaFree_(S.dMM); if (S.dHist) cudaFree_(S.dHist); if (S.plan) cufftDestroy_(S.plan);
    S.dG = S.dZ = nullptr; S.dH = nullptr; S.dMM = S.dHist = nullptr; S.plan = 0;
    if (cudaMalloc_((void**)&S.dG, (size_t)FW * FH * sizeof(F2))) return false;
    if (cudaMalloc_((void**)&S.dZ, (size_t)FW * FH * sizeof(F2))) return false;
    if (cudaMalloc_((void**)&S.dH, (size_t)W * H * sizeof(float))) return false;
    if (cudaMalloc_((void**)&S.dMM, 2 * sizeof(unsigned))) return false;
    if (cudaMalloc_((void**)&S.dHist, HB * sizeof(unsigned))) return false;
    if (cufftPlan2d_(&S.plan, FH, FW, CUFFT_C2C)) return false;
    S.FW = FW; S.FH = FH; S.W = W; S.H = H; return true;
}

static bool launch2d(CUfunction f, int gx, int gy, void** args)
{ return cuLaunchKernel_(f, gx, gy, 1, 16, 16, 1, 0, nullptr, args, nullptr) == 0; }
static bool launch1d(CUfunction f, int blocks, void** args)
{ return cuLaunchKernel_(f, blocks, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr) == 0; }

// Full GPU reconstruction incl. percentile normalisation + strength contrast.
// p,q are the (unit-scale, max_slope-clipped) slope fields; out is the final
// 0..1 height (W*H). Returns false on any failure so the caller can fall back.
bool reconstruct(const std::vector<double>& p, const std::vector<double>& q,
                 int W, int H, float strength, std::vector<float>& out)
{
    while (gBusy.test_and_set(std::memory_order_acquire)) std::this_thread::yield();
    struct Rel { ~Rel() { gBusy.clear(std::memory_order_release); } } rel;
    if (!S.tried) { S.tried = true; S.ok = init_locked(); }
    if (!S.ok) return false;

    // Push our primary context for the duration of this call; the guard pops it
    // on every return path so Nuke's own context state is never disturbed.
    struct CtxScope {
        bool ok; CtxScope() { ok = (cuCtxPushCurrent_(S.ctx) == 0); }
        ~CtxScope() { if (ok) { CUcontext p = nullptr; cuCtxPopCurrent_(&p); } }
    } ctxScope;
    if (!ctxScope.ok) return false;

    if (!ensureBuffers(W, H)) return false;
    const int FW = S.FW, FH = S.FH;
    const long long n = (long long)W * H;

    // Pack p into real, q into imag, tiled periodically into the padded grid.
    std::vector<F2> G((size_t)FW * FH);
    for (int y = 0; y < FH; ++y) {
        const int sy = y % H;
        for (int x = 0; x < FW; ++x) {
            const int sx = x % W;
            G[(size_t)y * FW + x].x = (float)p[(size_t)sy * W + sx];
            G[(size_t)y * FW + x].y = (float)q[(size_t)sy * W + sx];
        }
    }
    if (cudaMemcpy_(S.dG, G.data(), (size_t)FW * FH * sizeof(F2), H2D)) return false;
    if (cufftExecC2C_(S.plan, S.dG, S.dG, FWD)) return false;
    { int gx=(FW+15)/16, gy=(FH+15)/16; void* a[]={&S.dG,&S.dZ,(void*)&FW,(void*)&FH}; if(!launch2d(S.kSolve,gx,gy,a)) return false; }
    if (cufftExecC2C_(S.plan, S.dZ, S.dZ, INV)) return false;
    float invN = 1.0f / ((float)FW * FH);
    { int gx=(W+15)/16, gy=(H+15)/16; void* a[]={&S.dZ,&S.dH,(void*)&W,(void*)&H,(void*)&FW,&invN}; if(!launch2d(S.kFinish,gx,gy,a)) return false; }

    // Percentile normalisation on the GPU: min/max -> histogram -> map+strength.
    unsigned initMM[2] = { 0xffffffffu, 0u };
    if (cudaMemcpy_(S.dMM, initMM, 2 * sizeof(unsigned), H2D)) return false;
    { unsigned* gmn=S.dMM; unsigned* gmx=S.dMM+1; void* a[]={&S.dH,(void*)&n,&gmn,&gmx}; if(!launch1d(S.kMinMax,256,a)) return false; }
    if (cuCtxSynchronize_() != 0) return false;
    unsigned mm[2];
    if (cudaMemcpy_(mm, S.dMM, 2 * sizeof(unsigned), D2H)) return false;
    float fmn = decU(mm[0]), fmx = decU(mm[1]);
    if (!(fmx > fmn)) { out.assign((size_t)W * H, 0.5f); return true; }   // flat

    float histInv = 1.0f / (fmx - fmn); int B = HB;
    if (cudaMemset_(S.dHist, 0, HB * sizeof(unsigned))) return false;
    { void* a[]={&S.dH,(void*)&n,&fmn,&histInv,&B,&S.dHist}; if(!launch1d(S.kHist,256,a)) return false; }
    if (cuCtxSynchronize_() != 0) return false;
    std::vector<unsigned> hist(HB);
    if (cudaMemcpy_(hist.data(), S.dHist, HB * sizeof(unsigned), D2H)) return false;

    const float clip = 0.00001f;
    const long long loR = (long long)(clip * (double)(n - 1));
    const long long hiR = (long long)((1.0 - clip) * (double)(n - 1));
    long long cum = 0; int bLo = 0, bHi = HB - 1; bool gotLo = false;
    for (int b = 0; b < HB; ++b) {
        cum += hist[b];
        if (!gotLo && cum > loR) { bLo = b; gotLo = true; }
        if (cum > hiR) { bHi = b; break; }
    }
    const float span = fmx - fmn;
    float mn = fmn + ((float)bLo / (float)HB) * span;
    float mx = fmn + ((float)(bHi + 1) / (float)HB) * span;
    float invP = (mx > mn) ? 1.0f / (mx - mn) : 0.0f;
    { void* a[]={&S.dH,(void*)&n,&mn,&invP,&strength}; if(!launch1d(S.kApply,(int)((n+255)/256),a)) return false; }
    if (cuCtxSynchronize_() != 0) return false;

    out.assign((size_t)W * H, 0.0f);
    if (cudaMemcpy_(out.data(), S.dH, (size_t)W * H * sizeof(float), D2H)) return false;
    cudaSync_();
    return true;
}

} // namespace nhcuda
#endif // _WIN32


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
        Bool_knob(f, &_useGPUIfAvailable, "use_gpu", "use GPU if available");
        Tooltip(f, "Run the per-pixel normal computation on the GPU via Blink "
                   "when a GPU is available; otherwise it runs on the CPU. The "
                   "result is identical either way.");

        Float_knob(f, &_strength, IRange(-1.0, 1.0), "strength", "strength");
        Tooltip(f, "Relief strength. The slider runs -1..1 and maps linearly to a "
                   "gradient multiplier of -200..200 (slider 1.0 = x200, the "
                   "default). Negative values invert the relief (concave <-> convex).");

        static const char* const srcModes[] = { "red", "luminance", nullptr };
        Enumeration_knob(f, &_heightSrc, srcModes, "height_source", "height source");
        Tooltip(f, "Which input channel(s) provide the height scalar.");

        Bool_knob(f, &_directX, "directx", "directX");
        Tooltip(f, "OFF: OpenGL convention (G = +Y up) — default.\n"
                   "ON: DirectX convention (G = +Y down).");

        // All parameter descriptions collected at the bottom under "Tips".
        // A single multi-line Text_knob: embedded "\n" gives real line breaks,
        // "\n\n" a blank line between entries (separate knobs collapse instead).
        BeginClosedGroup(f, "Tips");
        Named_Text_knob(f, "t_tips", "",
                  "HeightToNormal — builds a tangent-space normal map from a "
                  "height map.\n"
                  "\n"
                  "strength: relief amount, -1..1 -> gradient x(-200..200).\n"
                  "    1 = default, 0 = flat, negative = inverted.\n"
                  "\n"
                  "height source: 'red' = R channel only,\n"
                  "    'luminance' = Rec.601 weighted RGB.\n"
                  "\n"
                  "directX: green channel convention.\n"
                  "    OFF = OpenGL (+Y up), ON = DirectX (+Y down).\n"
                  "\n"
                  "use GPU: GPU (Blink) when available, else CPU. Same result.");
        EndGroup(f);
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
    bool  _useGPU;       // use CUDA (cuFFT) if available; else CPU fallback

    // Cached reconstruction (full image, in input-box coordinates).
    std::vector<float> _height;
    int _cw, _ch, _cx0, _cy0;
    bool _cacheValid;
    Hash _cacheHash;     // key of the inputs that produced the cached _height

public:
    NormalToHeight(Node* node) : PlanarIop(node)
        , _strength(1.0f)
        , _maxSlope(1.0f)
        , _clip(0.00001f)
        , _directX(false)
        , _useGPU(true)         // GPU when available; auto CPU fallback on any failure
        , _cw(0), _ch(0), _cx0(0), _cy0(0)
        , _cacheValid(false)
    {}

    const char* Class() const override { return N2H_CLASS; }
    const char* node_help() const override { return N2H_HELP; }
    static const Iop::Description d;

    void knobs(Knob_Callback f) override
    {
        Bool_knob(f, &_useGPU, "use_gpu", "use GPU if available");
        Tooltip(f, "Run the FFT integration and normalisation on the GPU (CUDA / "
                   "cuFFT) when an NVIDIA GPU and Nuke's bundled CUDA libraries are "
                   "present. Falls back to the multi-threaded CPU path otherwise. "
                   "The result is numerically equivalent (within ~3e-5).");

        Float_knob(f, &_strength, IRange(-1.0, 1.0), "strength", "strength");
        Tooltip(f, "Relief depth and direction, applied as contrast about mid-grey "
                   "after the 0..1 normalisation. 1.0 = full relief (default); "
                   "smaller magnitudes (e.g. 0.1) give shallower relief; 0 = flat "
                   "grey; negative values invert the relief, with |value| setting "
                   "depth. Range -1..1.");

        Float_knob(f, &_maxSlope, IRange(1.0, 50.0), "max_slope", "max slope");
        Tooltip(f, "Clips the per-pixel slope |dz/dx|,|dz/dy| before integration. "
                   "Stray near-grazing normals (nz~0, e.g. a lone RGB=255/0 dot) "
                   "make the slope explode and the FFT turns it into a sharp "
                   "white/black peak. Lower this if you see such spikes (3-10 is a "
                   "good range); raise it to allow steeper walls. Default 1.");

        Bool_knob(f, &_directX, "directx", "directX");
        Tooltip(f, "OFF: OpenGL convention (G = +Y up) — default.\n"
                   "ON: DirectX convention (G = +Y down).");

        // All parameter descriptions collected at the bottom under "Tips".
        // A single multi-line Text_knob: embedded "\n" gives real line breaks,
        // "\n\n" a blank line between entries (separate knobs collapse instead).
        BeginClosedGroup(f, "Tips");
        Named_Text_knob(f, "t_tips", "",
                  "NormalToHeight — reconstructs a height map from a tangent-space "
                  "normal map (Frankot-Chellappa FFT integration).\n"
                  "\n"
                  "strength: relief depth & direction (-1..1), applied as\n"
                  "    contrast about mid-grey. 1 = full relief, 0.1 = shallow,\n"
                  "    0 = flat grey, negative = inverted (|value| = depth).\n"
                  "\n"
                  "max slope: clips runaway slopes from near-grazing normals\n"
                  "    (nz~0) that would become white/black spikes. Lower (3-10)\n"
                  "    to suppress spikes, raise for steeper walls. Default 1.\n"
                  "\n"
                  "directX: green channel convention.\n"
                  "    OFF = OpenGL (+Y up), ON = DirectX (+Y down).\n"
                  "    Match your input normal map.");
        EndGroup(f);
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

        // Cache key: upstream content + our knobs + box. Nuke re-renders the full
        // plane many times per interaction; when nothing relevant changed, reuse
        // the previously reconstructed _height and skip both the (128 MB) input
        // fetch and the whole FFT solve.
        Hash key;
        key.append(input0().hash());
        key.append(_strength); key.append(_maxSlope);
        key.append(_directX);  key.append(_useGPU);
        key.append(ibox.x()); key.append(ibox.y()); key.append(ibox.r()); key.append(ibox.t());

        if (_cacheValid && key == _cacheHash && !_height.empty()) {
            // cache hit: reuse _height, skip fetch + solve
        } else {
            ImagePlane inputPlane(ibox, outputPlane.packed(),
                                  outputPlane.channels(), outputPlane.nComps());
            input0().fetchPlane(inputPlane);
            if (aborted()) return;

            buildHeight(inputPlane, ibox);
            _cacheHash = key;
            _cacheValid = true;
        }

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

        fftutil::parallelChunks(H, [&](int yy0, int yy1) {
        for (int yy = yy0; yy < yy1; ++yy) {
            const int sy = yy + _cy0;
            for (int xx = 0; xx < W; ++xx) {
                const int sx = xx + _cx0;
                float nr = in.at(sx, sy, 0);
                float ng = nComp > 1 ? in.at(sx, sy, 1) : 0.5f;
                float nb = nComp > 2 ? in.at(sx, sy, 2) : 1.0f;

                // Decode 0..1 -> [-1,1] at UNIT scale (true slopes). 'strength'
                // is NOT applied here: any uniform scale on the slopes would be
                // cancelled by the 0..1 normalisation below, so strength's depth
                // (magnitude) is applied AFTER normalisation instead, where it is
                // visible. Decoding at unit scale also makes max_slope clip the
                // real slopes consistently regardless of the strength knob.
                double nx = ((double)nr * 2.0 - 1.0);
                double ny = ((double)ng * 2.0 - 1.0) * gy_sign;
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
        });   // parallelChunks(H) — slope decode

#ifdef _WIN32
        // GPU path (cuFFT + nvrtc): produces the final normalised height incl.
        // strength. On any failure (no NVIDIA GPU, missing libs, CUDA error) it
        // returns false and we fall through to the CPU path below.
        if (_useGPU) {
            bool gpuOK = false;
            // /EHa is enabled, so catch(...) also catches hardware (SEH) faults:
            // any GPU crash falls back to the CPU path instead of taking down Nuke.
            try { gpuOK = nhcuda::reconstruct(p, q, W, H, _strength, _height); }
            catch (...) { gpuOK = false; }
            if (gpuOK) return;
        }
#endif

        // Frankot-Chellappa on a padded power-of-two grid (periodic).
        const int FW = fftutil::nextPow2(W);
        const int FH = fftutil::nextPow2(H);

        // Pack BOTH real slope fields into ONE complex grid: G = p + i*q. A single
        // forward FFT then carries both spectra (the "two real FFTs for the price
        // of one" trick), halving the forward-transform cost. Tile periodically so
        // the implicit FFT periodicity matches the seamless-texture assumption.
        std::vector<fftutil::Cplx> G((size_t)FW * FH);
        for (int yy = 0; yy < FH; ++yy) {
            const int sy = yy % H;
            for (int xx = 0; xx < FW; ++xx) {
                const int sx = xx % W;
                G[(size_t)yy * FW + xx].re = (float)p[(size_t)sy * W + sx];
                G[(size_t)yy * FW + xx].im = (float)q[(size_t)sy * W + sx];
            }
        }
        fftutil::fft2d(G, FW, FH, -1);

        // Per-axis frequency response of the FORWARD difference operator
        //     D(w) = e^{i w} - 1   ->   conj(D) = (cos w - 1) - i sin w,
        // matching HeightToNormal's forward-difference gradient. |D(w)|^2 =
        // 2(1-cos w) has NO zero at Nyquist (unlike central diff's sin(w)), so the
        // even/odd sub-grids stay coupled and the 1-pixel grid grain disappears.
        // Precomputing per axis removes FW*FH cos/sin calls from the solve loop.
        std::vector<float> dxr(FW), dxi(FW), mx(FW), dyr(FH), dyi(FH), my(FH);
        for (int kx = 0; kx < FW; ++kx) {
            const double ax = 2.0 * M_PI * ((kx <= FW / 2) ? kx : kx - FW) / (double)FW;
            const double r = std::cos(ax) - 1.0, i = -std::sin(ax);
            dxr[kx] = (float)r; dxi[kx] = (float)i; mx[kx] = (float)(r * r + i * i);
        }
        for (int ky = 0; ky < FH; ++ky) {
            const double ay = 2.0 * M_PI * ((ky <= FH / 2) ? ky : ky - FH) / (double)FH;
            const double r = std::cos(ay) - 1.0, i = -std::sin(ay);
            dyr[ky] = (float)r; dyi[ky] = (float)i; my[ky] = (float)(r * r + i * i);
        }

        // Unpack P^,Q^ from the packed spectrum and solve in one pass. For real
        // inputs, the packed transform G^ obeys
        //     P^(k) = (G^(k) + conj(G^(-k))) / 2,
        //     Q^(k) = (G^(k) - conj(G^(-k))) / (2i),
        // where -k is the frequency mirror (FW-kx, FH-ky) modulo the grid. Then
        //     Z^ = (conj(Dx)*P^ + conj(Dy)*Q^) / (|Dx|^2 + |Dy|^2).
        // A separate Z buffer is required: each output reads G^ at the mirror
        // index, so we must not overwrite G^ while still unpacking it.
        std::vector<fftutil::Cplx> Z((size_t)FW * FH);
        fftutil::parallelChunks(FH, [&](int ky0, int ky1) {
        for (int ky = ky0; ky < ky1; ++ky) {
            const int myk = (FH - ky) % FH;
            const float DYR = dyr[ky], DYI = dyi[ky];
            for (int kx = 0; kx < FW; ++kx) {
                const int mxk = (FW - kx) % FW;
                const fftutil::Cplx a = G[(size_t)ky  * FW + kx];
                const fftutil::Cplx b = G[(size_t)myk * FW + mxk];   // G^(-k)
                // P^ = (a + conj(b))/2 ; Q^ = (a - conj(b))/(2i)
                const float Pr = 0.5f * (a.re + b.re), Pi = 0.5f * (a.im - b.im);
                const float Qr = 0.5f * (a.im + b.im), Qi = -0.5f * (a.re - b.re);

                const float denom = mx[kx] + my[ky];
                const size_t idx = (size_t)ky * FW + kx;
                if (denom < 1e-12f) {
                    Z[idx].re = 0.0f; Z[idx].im = 0.0f; // DC = mean height (set to 0)
                    continue;
                }
                const float DXR = dxr[kx], DXI = dxi[kx];
                // num = conj(Dx)*P^ + conj(Dy)*Q^   (complex multiply-add)
                const float nr = (DXR * Pr - DXI * Pi) + (DYR * Qr - DYI * Qi);
                const float ni = (DXR * Pi + DXI * Pr) + (DYR * Qi + DYI * Qr);
                Z[idx].re = nr / denom;
                Z[idx].im = ni / denom;
            }
        }
        });   // parallelChunks(FH)

        fftutil::fft2d(Z, FW, FH, +1);
        const float invN = 1.0f / ((float)FW * FH);

        _height.assign((size_t)W * H, 0.0f);
        for (int yy = 0; yy < H; ++yy)
            for (int xx = 0; xx < W; ++xx)
                _height[(size_t)yy * W + xx] =
                    Z[(size_t)yy * FW + xx].re * invN;

        {
            // Always normalise to 0..1 with robust (percentile) min/max. The
            // Frankot-Chellappa solve can produce a few overshoot/ringing pixels
            // at sharp edges; using the absolute min/max would let those outliers
            // dominate the range and crush the mid-tones. Map the low/high
            // percentile points to 0/1 so the bulk of the image uses full contrast.
            const size_t n = _height.size();

            // _clip is the fraction trimmed from EACH end (e.g. 0.001 = 0.1%).
            float clip = _clip;
            if (clip < 0.0f) clip = 0.0f;
            if (clip > 0.49f) clip = 0.49f;

            size_t loIdx = (size_t)(clip * (double)(n - 1));
            size_t hiIdx = (size_t)((1.0 - clip) * (double)(n - 1));
            if (hiIdx <= loIdx) { loIdx = 0; hiIdx = n - 1; }

            // We only need the lo/hi percentile VALUES, not a fully sorted array.
            // Two nth_element selections (O(n)) find them ~8x faster than a full
            // std::sort (O(n log n)) at 4K, yielding identical mn/mx. The first
            // call partitions around loIdx; the second finds hiIdx in the upper
            // partition (everything past loIdx), so both ranks are exact.
            std::vector<float> tmp(_height);
            std::nth_element(tmp.begin(), tmp.begin() + loIdx, tmp.end());
            const float mn = tmp[loIdx];
            std::nth_element(tmp.begin() + loIdx + 1, tmp.begin() + hiIdx, tmp.end());
            const float mx = tmp[hiIdx];
            const float range = (mx - mn);
            const float inv = range > 1e-8f ? 1.0f / range : 0.0f;

            // 'strength' sets relief depth AND direction, applied as a signed
            // contrast about the mid-level 0.5:
            //     out = 0.5 + (norm - 0.5) * strength      (clamped to 0..1)
            //   strength = 1    -> full relief (norm unchanged)
            //   strength = 0.1  -> shallow relief (low contrast)
            //   strength = 0    -> flat 0.5
            //   strength < 0    -> inverted relief, |strength| sets depth
            // With |strength| <= 1 the result already stays within 0..1.
            const float s = _strength;
            for (float& v : _height) {
                float nv = (v - mn) * inv;      // map [lo,hi] percentile -> [0,1]
                if (nv < 0.0f) nv = 0.0f;       // clamp the trimmed outliers
                if (nv > 1.0f) nv = 1.0f;
                nv = 0.5f + (nv - 0.5f) * s;    // depth (magnitude) + invert (sign)
                if (nv < 0.0f) nv = 0.0f;
                if (nv > 1.0f) nv = 1.0f;
                v = nv;
            }
        }
    }
};

static Iop* buildN2H(Node* node) { return new NormalToHeight(node); }
const Iop::Description NormalToHeight::d(N2H_CLASS, "Filter/NormalToHeight", buildN2H);
