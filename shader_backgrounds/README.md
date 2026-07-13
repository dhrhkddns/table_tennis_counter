# Seamless gray shader background GIFs

1920×1080 looping background animations in a cool gray palette.

- **20 frames** · **80ms/frame** · infinite loop
- Motion uses **integer 2π harmonics** so the last frame connects to the first without a jump
- Soft procedural looks (flow, smoke, scan, aurora, sheen, …)

## Files

| File | Style |
|------|--------|
| `shader_bg_01_soft_flow.gif` | Soft flowing fog |
| `shader_bg_02_drift_bands.gif` | Diagonal drifting bands |
| `shader_bg_03_radial_pulse.gif` | Radial pulse rings |
| `shader_bg_04_scan_wrap.gif` | Soft wrapping scan band |
| `shader_bg_05_smoke_warp.gif` | Domain-warped smoke |
| `shader_bg_06_grid_shimmer.gif` | Faint grid shimmer |
| `shader_bg_07_aurora_gray.gif` | Vertical gray ribbons |
| `shader_bg_08_soft_wash.gif` | Large soft wash |
| `shader_bg_09_orbit_blobs.gif` | Orbiting soft blobs |
| `shader_bg_10_metal_sheen.gif` | Brushed metal sheen |

## Regenerate

```bash
python3 generate_shader_backgrounds.py
```

Requires: `numpy`, `Pillow`, and optionally `gifsicle` (for size).
