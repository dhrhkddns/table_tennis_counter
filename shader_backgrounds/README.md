# Seamless black shader background GIFs

1920×1080 looping background animations in a near-black / charcoal palette.

- **20 frames** · **80ms/frame** · infinite loop
- Motion uses **integer 2π harmonics** so the last frame connects to the first without a jump
- New pattern set (distinct from the earlier gray flow pack)

## Files

| File | Style |
|------|--------|
| `shader_bg_01_ink_ripple.gif` | Concentric ink ripples |
| `shader_bg_02_hex_crawl.gif` | Soft hexagonal lattice crawl |
| `shader_bg_03_vortex_spiral.gif` | Rotating spiral arms |
| `shader_bg_04_rain_streaks.gif` | Diagonal rain / scratch streaks |
| `shader_bg_05_ember_sparks.gif` | Sparse rising charcoal sparks |
| `shader_bg_06_cross_hatch.gif` | Animated cross-hatch etch |
| `shader_bg_07_horizon_wave.gif` | Stacked horizon wave layers |
| `shader_bg_08_caustic_cells.gif` | Dark caustic / cellular membrane |
| `shader_bg_09_radar_sweep.gif` | Radar sweep with faint rings |
| `shader_bg_10_pixel_static.gif` | Coarse blocky void static |

## Regenerate

```bash
python3 generate_shader_backgrounds.py
```

Requires: `numpy`, `Pillow`, and optionally `gifsicle` (for size).
