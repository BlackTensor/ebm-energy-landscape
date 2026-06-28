import { useEffect, useRef } from 'react'
import { colormapLUT } from '../lib/colormap'
import { computeContours } from '../lib/contours'
import './EnergyHeatmap.css'

/*
 * EnergyHeatmap — the signature element, rendered as designed topography.
 *
 * The exported energy field is painted as a terrain plate: low-energy valleys in
 * deep water-blue, rising through teal/moss/sage and warm sand to terracotta
 * ridges (see ../lib/colormap). Three layers build the relief, all from the real
 * energy values:
 *
 *   1. Colour — the terrain ramp, monotonic in energy.
 *   2. Hillshade — a directional relief shade derived from the field's own
 *      gradient (light from the upper-left), so slopes catch light and hollows
 *      fall into shadow and the plate reads as raised terrain, not a flat fill.
 *   3. Contours — thin iso-energy lines at evenly spaced real thresholds
 *      (marching squares over the grid), the "survey map" furniture that
 *      reinforces the valley/ridge structure.
 *
 * The shaded field is painted once at native resolution onto an offscreen tile,
 * then drawn up to display size with image smoothing for a smooth interpolation
 * between cells; contours are stroked on afterward at display resolution so they
 * stay crisp. The backing store follows the device pixel ratio and a
 * ResizeObserver keeps everything sharp as the layout changes.
 *
 * Orientation: the field is `row -> y, col -> x`. Rows are flipped when painting
 * so world +y points up on screen, and contours use the same mapping, so every
 * layer and any trajectory overlay align exactly.
 *
 * `children` are rendered inside the framed, clipped square (e.g. the descent
 * animation's path overlay).
 */
export default function EnergyHeatmap({ field, children }) {
  const canvasRef = useRef(null)
  const wrapRef = useRef(null)

  useEffect(() => {
    if (!field) return
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    const lut = colormapLUT()
    const res = field.resolution
    const norm = field.energy_normalized

    // Screen-oriented height field (row flipped so world +y points up), reused
    // for both the colour/shade tile and the contour mapping below.
    const S = new Array(res)
    for (let y = 0; y < res; y++) S[y] = norm[res - 1 - y]
    const clampIdx = (i) => (i < 0 ? 0 : i > res - 1 ? res - 1 : i)

    // Light direction (upper-left, lifted out of the screen) and a vertical
    // exaggeration that turns the field's gentle gradients into legible relief.
    const Z = 4.0
    let Lx = -0.55
    let Ly = -0.55
    let Lz = 0.7
    const Linv = 1 / Math.hypot(Lx, Ly, Lz)
    Lx *= Linv
    Ly *= Linv
    Lz *= Linv

    // Paint the shaded terrain once at native resolution onto an offscreen tile.
    const tile = document.createElement('canvas')
    tile.width = res
    tile.height = res
    const tctx = tile.getContext('2d')
    const image = tctx.createImageData(res, res)
    const px = image.data
    for (let y = 0; y < res; y++) {
      for (let x = 0; x < res; x++) {
        let t = S[y][x]
        t = t < 0 ? 0 : t > 1 ? 1 : t
        const li = Math.round(t * 255) * 3

        // Surface normal from the local gradient, then Lambert shading.
        const dx = S[y][clampIdx(x + 1)] - S[y][clampIdx(x - 1)]
        const dy = S[clampIdx(y + 1)][x] - S[clampIdx(y - 1)][x]
        const nx = -dx * Z
        const ny = -dy * Z
        const ninv = 1 / Math.hypot(nx, ny, 1)
        let illum = (nx * Lx + ny * Ly + Lz) * ninv
        if (illum < 0) illum = 0
        const shade = 0.8 + 0.42 * illum // ~0.8 (shadow) .. ~1.22 (lit slope)

        const di = (y * res + x) * 4
        px[di] = Math.min(255, lut[li] * shade)
        px[di + 1] = Math.min(255, lut[li + 1] * shade)
        px[di + 2] = Math.min(255, lut[li + 2] * shade)
        px[di + 3] = 255
      }
    }
    tctx.putImageData(image, 0, 0)

    // Iso-energy contour segments, in grid coords, computed once for the field.
    const segs = computeContours(norm)

    const ctx = canvas.getContext('2d')
    const draw = () => {
      const cssW = wrap.clientWidth
      const cssH = wrap.clientHeight
      if (!cssW || !cssH) return
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const W = Math.round(cssW * dpr)
      const H = Math.round(cssH * dpr)
      canvas.width = W
      canvas.height = H
      ctx.imageSmoothingEnabled = true
      ctx.imageSmoothingQuality = 'high'
      ctx.drawImage(tile, 0, 0, res, res, 0, 0, W, H)

      // Map grid coords -> device pixels, flipping rows (world +y up) and using
      // sample-centre offsets so lines sit on the colour bands they describe.
      const gx = (c) => ((c + 0.5) / res) * W
      const gy = (r) => ((res - 0.5 - r) / res) * H
      ctx.lineWidth = Math.max(1, dpr * 0.75)
      ctx.strokeStyle = 'rgba(28, 24, 18, 0.26)' // fine earthy-ink survey lines
      ctx.beginPath()
      for (let i = 0; i < segs.length; i += 4) {
        ctx.moveTo(gx(segs[i]), gy(segs[i + 1]))
        ctx.lineTo(gx(segs[i + 2]), gy(segs[i + 3]))
      }
      ctx.stroke()
    }

    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [field])

  return (
    <div className="plate-mount">
      <figure className="heatmap" ref={wrapRef}>
        <canvas
          ref={canvasRef}
          className="heatmap-canvas"
          role="img"
          aria-label="Energy landscape rendered as terrain: deep blue low-energy valleys rising through green uplands to clay-red high-energy ridges, with iso-energy contour lines."
        />
        {children}
      </figure>
    </div>
  )
}
