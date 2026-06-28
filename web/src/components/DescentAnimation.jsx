import { useEffect, useMemo, useRef, useState } from 'react'
import EnergyHeatmap from './EnergyHeatmap'
import { colorForEnergy01 } from '../lib/colormap'
import './DescentAnimation.css'

/*
 * DescentAnimation — the Langevin descent, animated.
 *
 * The exported descent history (descent.json) holds several distinct paths, each
 * recorded at the same Langevin steps (frame_steps, step 0 = chaos -> final =
 * settled). This component overlays those paths on the energy heatmap and plays
 * them forward: tangled scribbles untangle and slide into the green valleys.
 *
 * Smoothness comes from interpolating each path between recorded frames along a
 * continuous timeline rather than snapping frame to frame. Each path is tinted
 * by its *current* energy through the field's own colormap — ridge-orange while
 * chaotic, valley-green once settled — over a dark casing so it stays legible
 * against the field.
 *
 * Motion is restrained and reduced-motion is respected: when the user prefers
 * reduced motion we do not autoplay; we show the settled result and leave the
 * scrubber for manual, user-driven exploration.
 */

const DESCEND_SECONDS = 7 // chaos -> settled
const HOLD_SECONDS = 1.8 // dwell on the settled result before looping
const TOTAL_SECONDS = DESCEND_SECONDS + HOLD_SECONDS

const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches

// Cinematic pacing: a gentle ease-in out of chaos, then a long ease-out as the
// paths settle into their valleys — so the descent reads as motion coming to
// rest, not a linear sweep. (easeInOutQuart.)
const ease = (u) =>
  u < 0.5 ? 8 * u * u * u * u : 1 - ((-2 * u + 2) ** 4) / 2

// Inverse of `ease` (bisection) — so scrubbing to a progress value resumes from
// the matching point on the eased timeline without a jump.
const easeInv = (y) => {
  let lo = 0
  let hi = 1
  for (let i = 0; i < 24; i++) {
    const mid = (lo + hi) / 2
    if (ease(mid) < y) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

// Linear interpolation of a path's points between two recorded frames.
function pathAt(frames, p) {
  const i0 = Math.floor(p)
  const i1 = Math.min(i0 + 1, frames.length - 1)
  const f = p - i0
  const a = frames[i0]
  const b = frames[i1]
  const out = new Array(a.length)
  for (let k = 0; k < a.length; k++) {
    out[k] = [a[k][0] + (b[k][0] - a[k][0]) * f, a[k][1] + (b[k][1] - a[k][1]) * f]
  }
  return out
}

const lerpArr = (arr, p) => {
  const i0 = Math.floor(p)
  const i1 = Math.min(i0 + 1, arr.length - 1)
  return arr[i0] + (arr[i1] - arr[i0]) * (p - i0)
}

export default function DescentAnimation({ field, descent }) {
  const overlayRef = useRef(null)
  const timelineRef = useRef(0) // seconds along the loop
  const reduced = useMemo(prefersReducedMotion, [])

  const last = descent.n_frames - 1
  const [progress, setProgress] = useState(reduced ? last : 0)
  const [playing, setPlaying] = useState(!reduced)
  const [size, setSize] = useState({ w: 0, h: 0, dpr: 1 })

  const size01 = field.size
  const descents = descent.descents
  const steps = descent.frame_steps

  // Constant, pinned endpoints (same across all frames and descents).
  const start = descents[0].frames[0][0]
  const goal = descents[0].frames[0][descent.n_points - 1]

  // Keep the overlay's backing store matched to its box and the device pixels.
  useEffect(() => {
    const canvas = overlayRef.current
    const measure = () => {
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      setSize((s) => (s.w === w && s.h === h && s.dpr === dpr ? s : { w, h, dpr }))
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [])

  // The animation loop: advance a continuous timeline, derive progress from it.
  useEffect(() => {
    if (!playing) return
    let raf
    let prev = null
    const tick = (ts) => {
      if (prev === null) prev = ts
      let t = timelineRef.current + (ts - prev) / 1000
      prev = ts
      if (t >= TOTAL_SECONDS) t -= TOTAL_SECONDS
      timelineRef.current = t
      const p = t <= DESCEND_SECONDS ? ease(t / DESCEND_SECONDS) * last : last
      setProgress(p)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, last])

  // Draw the overlay whenever progress or size changes.
  useEffect(() => {
    const canvas = overlayRef.current
    const { w, h, dpr } = size
    if (!w || !h) return
    canvas.width = Math.round(w * dpr)
    canvas.height = Math.round(h * dpr)
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'

    const tx = (x) => (x / size01) * w
    const ty = (y) => (1 - y / size01) * h // world +y points up
    const casing = Math.max(3, w * 0.013)
    const stroke = Math.max(1.5, w * 0.0058)

    const trace = (pts) => {
      ctx.beginPath()
      for (let k = 0; k < pts.length; k++) {
        const X = tx(pts[k][0])
        const Y = ty(pts[k][1])
        if (k === 0) ctx.moveTo(X, Y)
        else ctx.lineTo(X, Y)
      }
      ctx.stroke()
    }

    // Tint by descent progress: ridge-orange while chaotic, cooling to
    // valley-green as the paths settle. (Path energies never reach the field's
    // ridge-driven vmax, so progress — not absolute energy — is what reliably
    // carries the full orange-to-green sweep.) A dark casing under every line
    // keeps it legible even where a green path lies in a green valley.
    const tint = colorForEnergy01(1 - progress / last)
    for (const d of descents) {
      const pts = pathAt(d.frames, progress)
      ctx.strokeStyle = 'rgba(8, 11, 16, 0.85)'
      ctx.lineWidth = casing
      trace(pts)
      ctx.strokeStyle = tint
      ctx.lineWidth = stroke
      ctx.globalAlpha = 0.94
      trace(pts)
      ctx.globalAlpha = 1
    }

    // Start (hollow ring) and goal (filled) markers, neutral ink.
    const r = Math.max(3, w * 0.014)
    const ink = '#e6eaf2'
    ctx.lineWidth = Math.max(1.5, w * 0.004)
    ctx.strokeStyle = ink
    ctx.beginPath()
    ctx.arc(tx(start[0]), ty(start[1]), r, 0, Math.PI * 2)
    ctx.stroke()
    ctx.fillStyle = ink
    ctx.beginPath()
    ctx.arc(tx(goal[0]), ty(goal[1]), r, 0, Math.PI * 2)
    ctx.fill()
  }, [progress, size, descents, size01, last, start, goal])

  const togglePlay = () => {
    if (!playing && progress >= last) {
      timelineRef.current = 0
      setProgress(0)
    }
    setPlaying((p) => !p)
  }

  const onScrub = (e) => {
    const p = Number(e.target.value)
    setPlaying(false)
    timelineRef.current = easeInv(last === 0 ? 0 : p / last) * DESCEND_SECONDS
    setProgress(p)
  }

  const stepNow = Math.round(lerpArr(steps, progress))
  const meanEnergy =
    descents.reduce((acc, d) => acc + lerpArr(d.energy, progress), 0) / descents.length
  const settled = progress >= last - 1e-6
  const phase = settled ? 'SETTLED' : progress <= 0.001 ? 'CHAOS' : 'DESCENDING'

  return (
    <div className="descent">
      <EnergyHeatmap field={field}>
        <canvas
          ref={overlayRef}
          className="descent-overlay"
          role="img"
          aria-label="Six trajectories descending from chaos into the energy landscape's valleys."
        />
      </EnergyHeatmap>

      <div className="descent-controls">
        <button
          type="button"
          className="descent-play"
          onClick={togglePlay}
          aria-label={playing ? 'Pause descent' : 'Play descent'}
        >
          {playing ? 'Pause' : settled ? 'Replay' : 'Play'}
        </button>
        <input
          className="descent-scrub"
          type="range"
          min={0}
          max={last}
          step={0.01}
          value={progress}
          onChange={onScrub}
          aria-label="Scrub descent step"
        />
      </div>

      <dl className="descent-readout mono">
        <div className="readout-cell">
          <dt>step</dt>
          <dd>
            {stepNow}
            <span className="readout-dim"> / {descent.n_steps}</span>
          </dd>
        </div>
        <div className="readout-cell">
          <dt>mean energy</dt>
          <dd>{meanEnergy.toFixed(2)}</dd>
        </div>
        <div className="readout-cell readout-phase">
          <dt>phase</dt>
          <dd>{phase}</dd>
        </div>
      </dl>
    </div>
  )
}
