import { useEffect, useRef, useState } from 'react'
import EnergyHeatmap from './EnergyHeatmap'
import './MultiSample.css'

/*
 * MultiSample — the multimodality proof.
 *
 * The exported set (trajectories.json) holds several *distinct* valid routes for
 * one scene. This is the static counterpart to the descent animation: all routes
 * are drawn at once so the fan of equally-valid futures is legible at a glance.
 *
 * Every route is valid — low energy — so all are drawn in valley-green (the
 * accent's energy meaning), distinguished by geometry rather than by colour. A
 * faint scrim calms the field beneath so the routes read as the subject, while
 * the ridges stay just visible as context for what the paths avoid.
 */
export default function MultiSample({ field, trajectories }) {
  const overlayRef = useRef(null)
  const [size, setSize] = useState({ w: 0, h: 0, dpr: 1 })

  const size01 = field.size
  const routes = trajectories.trajectories
  const start = routes[0].points[0]
  const goal = routes[0].points[routes[0].points.length - 1]

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

  useEffect(() => {
    const canvas = overlayRef.current
    const { w, h, dpr } = size
    if (!w || !h) return
    canvas.width = Math.round(w * dpr)
    canvas.height = Math.round(h * dpr)
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)

    // Calm the terrain so the routes are the subject — a light, even dimming
    // that keeps the colour bands and contours legible underneath.
    ctx.fillStyle = 'rgba(14, 15, 19, 0.34)'
    ctx.fillRect(0, 0, w, h)

    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'
    const tx = (x) => (x / size01) * w
    const ty = (y) => (1 - y / size01) * h // world +y points up
    const casing = Math.max(3, w * 0.012)
    const stroke = Math.max(1.5, w * 0.0055)

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

    for (const r of routes) {
      ctx.strokeStyle = 'rgba(8, 11, 16, 0.85)'
      ctx.lineWidth = casing
      trace(r.points)
      ctx.strokeStyle = '#3dd2a4' // --valley: every route is valid / low energy
      ctx.lineWidth = stroke
      ctx.globalAlpha = 0.82
      trace(r.points)
      ctx.globalAlpha = 1
    }

    // Shared start (hollow ring) and goal (filled) markers, neutral ink.
    const rad = Math.max(3, w * 0.014)
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
  }, [size, routes, size01, start, goal])

  return (
    <EnergyHeatmap field={field}>
      <canvas
        ref={overlayRef}
        className="multi-overlay"
        role="img"
        aria-label={`${routes.length} distinct valid trajectories overlaid on one scene.`}
      />
    </EnergyHeatmap>
  )
}
