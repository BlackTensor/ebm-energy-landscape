import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import EnergyHeatmap from './EnergyHeatmap'
import './Comparison.css'

/*
 * Comparison — one answer, or every answer.
 *
 * Makes the value of the energy model obvious to a non-expert by putting two
 * predictions side by side on the same scene:
 *
 *   - Regression: a model trained to minimize average error predicts a single
 *     trajectory — the pointwise mean of the valid routes. Drawn in neutral ink
 *     (no energy claim is asserted about it). It is a smooth compromise that, by
 *     construction, matches none of the real routes.
 *   - Energy model: every distinct valid route survives as its own valley, drawn
 *     in valley-green (the accent's energy meaning — all valid / low energy).
 *
 * Both sit on the same calmed field with the same start/goal markers, so the
 * only difference the eye registers is one path versus many.
 */

function stroke(ctx, pts, tx, ty) {
  ctx.beginPath()
  for (let k = 0; k < pts.length; k++) {
    const X = tx(pts[k][0])
    const Y = ty(pts[k][1])
    if (k === 0) ctx.moveTo(X, Y)
    else ctx.lineTo(X, Y)
  }
  ctx.stroke()
}

// One framed field with a path overlay. `render(ctx, helpers)` draws the route(s);
// the scrim, transform, and start/goal markers are shared across both panels.
function Stage({ field, start, goal, render, ariaLabel }) {
  const overlayRef = useRef(null)
  const [size, setSize] = useState({ w: 0, h: 0, dpr: 1 })

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

  useEffect(() => {
    const canvas = overlayRef.current
    const { w, h, dpr } = size
    if (!w || !h) return
    canvas.width = Math.round(w * dpr)
    canvas.height = Math.round(h * dpr)
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)

    // Calm the terrain so the prediction is the subject — a light, even dimming
    // that keeps the colour bands and contours legible underneath.
    ctx.fillStyle = 'rgba(14, 15, 19, 0.36)'
    ctx.fillRect(0, 0, w, h)

    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'
    const tx = (x) => (x / field.size) * w
    const ty = (y) => (1 - y / field.size) * h // world +y points up
    render(ctx, { w, h, tx, ty })

    // Shared start (hollow) and goal (filled) markers, neutral ink.
    const rad = Math.max(3, w * 0.015)
    const ink = '#e6eaf2'
    ctx.lineWidth = Math.max(1.5, w * 0.004)
    ctx.strokeStyle = ink
    ctx.beginPath()
    ctx.arc(tx(start[0]), ty(start[1]), rad, 0, Math.PI * 2)
    ctx.stroke()
    ctx.fillStyle = ink
    ctx.beginPath()
    ctx.arc(tx(goal[0]), ty(goal[1]), rad, 0, Math.PI * 2)
    ctx.fill()
  }, [size, field, start, goal, render])

  return (
    <EnergyHeatmap field={field}>
      <canvas
        ref={overlayRef}
        className="cmp-overlay"
        role="img"
        aria-label={ariaLabel}
      />
    </EnergyHeatmap>
  )
}

export default function Comparison({ field, trajectories }) {
  const routes = trajectories.trajectories
  const start = routes[0].points[0]
  const goal = routes[0].points[routes[0].points.length - 1]

  // The regression answer: the pointwise mean of every valid route.
  const mean = useMemo(() => {
    const n = routes.length
    const m = routes[0].points.length
    const out = new Array(m)
    for (let k = 0; k < m; k++) {
      let sx = 0
      let sy = 0
      for (const r of routes) {
        sx += r.points[k][0]
        sy += r.points[k][1]
      }
      out[k] = [sx / n, sy / n]
    }
    return out
  }, [routes])

  const renderRegression = useCallback(
    (ctx, { w, tx, ty }) => {
      ctx.strokeStyle = 'rgba(8, 11, 16, 0.85)'
      ctx.lineWidth = Math.max(3, w * 0.012)
      stroke(ctx, mean, tx, ty)
      ctx.strokeStyle = '#e6eaf2' // --ink: a single neutral prediction
      ctx.lineWidth = Math.max(1.5, w * 0.0058)
      ctx.globalAlpha = 0.95
      stroke(ctx, mean, tx, ty)
      ctx.globalAlpha = 1
    },
    [mean],
  )

  const renderEbm = useCallback(
    (ctx, { w, tx, ty }) => {
      for (const r of routes) {
        ctx.strokeStyle = 'rgba(8, 11, 16, 0.85)'
        ctx.lineWidth = Math.max(3, w * 0.012)
        stroke(ctx, r.points, tx, ty)
        ctx.strokeStyle = '#3dd2a4' // --valley: every route valid / low energy
        ctx.lineWidth = Math.max(1.5, w * 0.0055)
        ctx.globalAlpha = 0.82
        stroke(ctx, r.points, tx, ty)
        ctx.globalAlpha = 1
      }
    },
    [routes],
  )

  return (
    <section className="compare" aria-label="One prediction versus many">
      <header className="compare-head">
        <h2 className="compare-title">One answer, or every answer</h2>
        <p className="compare-note">
          Train a model to minimize average error and it predicts a single
          trajectory — the mean of all the valid routes, a smooth compromise that
          matches none of them. The energy model keeps every valid route as its own
          valley, so the full set of options survives.
        </p>
      </header>

      <div className="compare-grid">
        <article className="compare-col">
          <div className="compare-tag mono">
            <span>REGRESSION</span>
            <span className="compare-count">1 prediction</span>
          </div>
          <Stage
            field={field}
            start={start}
            goal={goal}
            render={renderRegression}
            ariaLabel="Regression baseline: a single averaged trajectory."
          />
          <p className="compare-cap">
            The options collapse into one averaged path.
          </p>
        </article>

        <article className="compare-col">
          <div className="compare-tag mono">
            <span>ENERGY MODEL</span>
            <span className="compare-count">{routes.length} valleys</span>
          </div>
          <Stage
            field={field}
            start={start}
            goal={goal}
            render={renderEbm}
            ariaLabel={`Energy model: ${routes.length} distinct valid trajectories.`}
          />
          <p className="compare-cap">
            Every distinct valid route is kept as its own valley.
          </p>
        </article>
      </div>
    </section>
  )
}
