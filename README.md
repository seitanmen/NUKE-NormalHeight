# NUKE-NormalHeight

Two Nuke native C++ plugins that convert between **tangent-space normal maps** and **height maps**.

## Overview

- **HeightToNormal** (`vmt_HeightToNormal`): derive a tangent-space normal map from a height map via finite-difference gradients.
- **NormalToHeight** (`vmt_NormalToHeight`): reconstruct a height map from a normal map by least-squares gradient integration using the **Frankot-Chellappa** method (Poisson solve in the frequency domain via FFT).

Both nodes use the **OpenGL** normal convention (+Y up) by default, with a `directx` toggle (+Y down). The FFT integrator assumes a periodic boundary, so **tileable** maps round-trip seamlessly.

## Features

- **HeightToNormal** (node `HeightToNormal`): central-difference gradients, `strength` control, height source = red channel or luminance, OpenGL/DirectX toggle. Wrap-around sampling keeps tileable maps seamless.
- **NormalToHeight** (node `NormalToHeight`): Frankot-Chellappa FFT integration — a least-squares-optimal, drift-free height reconstruction. `scale` and `normalize 0-1` controls, OpenGL/DirectX toggle. The full-image solve runs once per validate and is cached.
- **Self-contained FFT**: a small radix-2 Cooley-Tukey implementation is built in — no external dependencies.
- **Round-trip accurate**: the NormalToHeight FFT kernel uses the central-difference frequency response (`sin ω`) so it matches HeightToNormal's discrete gradient. A standalone test recovers the height field to ~1e-16 RMS error.
- **Multi-version**: Nuke 15.1 and 16.0 support (separate builds).

## Build

```bash
# Nuke 16.0
cmake -G "Visual Studio 17 2022" -A x64 -B build -DNUKE_ROOT="C:/Program Files/Nuke16.0v6"
cmake --build build --config Release

# Nuke 15.1
cmake -G "Visual Studio 17 2022" -A x64 -B build151 -DNUKE_ROOT="C:/Program Files/Nuke15.1v5"
cmake --build build151 --config Release
```

Both nodes are compiled into a single `NormalHeight.dll`.

### Deployment

Set the `VMT_NUKE_DEPLOY_DIR` environment variable (or pass `-DDEPLOY_DIR=...`). The build's POST_BUILD step copies the DLL into a per-version subfolder (`16.0/` or `15.1/`), chosen automatically from the NDK include path.

## Usage

```python
nuke.createNode("HeightToNormal")  # menu: Filter > HeightToNormal
nuke.createNode("NormalToHeight")  # menu: Filter > NormalToHeight
```

### HeightToNormal parameters

| Knob (script name) | Description | Default |
|---|---|---|
| `strength` | Gradient strength multiplier (0–10) | 1.0 |
| `height_source` | `red` or `luminance` | red |
| `directx` | OFF = OpenGL (+Y up), ON = DirectX (+Y down) | off |

### NormalToHeight parameters

| Knob (script name) | Description | Default |
|---|---|---|
| `scale` | Height scale before normalisation (0–10) | 1.0 |
| `normalize` | Remap reconstructed height to 0–1 | on |
| `directx` | OFF = OpenGL (+Y up), ON = DirectX (+Y down) | off |

## Algorithm Details

### HeightToNormal
For each pixel, central differences give the height gradient `(∂h/∂x, ∂h/∂y)`, scaled by `strength`. The tangent-space normal is `normalize(-∂h/∂x, -∂h/∂y, 1)`, encoded `[-1,1] → [0,1]` into RGB. Sampling wraps around the image so tileable input stays seamless.

### NormalToHeight (Frankot-Chellappa)
1. Decode RGB normal `[0,1] → [-1,1]`, form the slope field `p = -nx/nz`, `q = -ny/nz`.
2. FFT both slope fields.
3. Solve the Poisson equation in the frequency domain:
   `Z(ωx, ωy) = -i(ωx·P + ωy·Q) / (ωx² + ωy²)`, with `Z(0,0) = 0`.
   The kernel frequencies use the central-difference response `sin ω` to match HeightToNormal.
4. Inverse FFT → height field, optionally normalised to 0–1.

This is the least-squares-optimal integration: it finds the height whose gradients best match the input slopes, with no directional drift (the artifact of naive line-by-line integration). The grid is padded to the next power of two per axis (periodic tiling of the slope field).

## Requirements

- Visual Studio 2022 (MSVC v143)
- CMake 3.10+
- Nuke 16.0v6 or 15.1v5
- Windows only

## License

MPL-2.0 (Mozilla Public License 2.0) — See [LICENSE](LICENSE)

## References

1. Robert T. Frankot and Rama Chellappa. "A Method for Enforcing Integrability in
   Shape from Shading Algorithms." *IEEE Transactions on Pattern Analysis and Machine
   Intelligence*, vol. 10, no. 4, pp. 439–451, 1988. doi:10.1109/34.3909

## Author

Seitanmen — https://github.com/seitanmen
