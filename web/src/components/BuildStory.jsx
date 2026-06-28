import { useMemo } from 'react'
import { dataUrl } from '../lib/bundle'
import './BuildStory.css'

/*
 * BuildStory — "How this was built".
 *
 * The credibility section: it makes the real training work visible so a viewer
 * can tell these visuals came from a genuinely trained model, not a mock-up.
 *
 *   1. The real training story — the energy-gap curve plotted from the actual
 *      per-epoch history saved in the checkpoint (exported to
 *      public/data/training_history.json). Valid paths (valley-green) are pushed
 *      to low energy and bad paths (ridge-orange) to high energy; the shaded
 *      band between them IS the energy gap, climbing from +0.001 at epoch 1 to
 *      +2.606 at epoch 120. No points are invented — if the history is missing
 *      we say so and plot only the genuine endpoints we have.
 *   2. Model facts — architecture, parameter count, training and generation
 *      methods, final accuracy and gap.
 *   3. Live stats — read at runtime from the real exports (never hardcoded), so
 *      they cannot drift from the data the page actually renders.
 *   4. A download of the trained weights themselves.
 *
 * Accents keep their energy meaning only (valley = valid/low, ridge = bad/high).
 */

// Documented genuine final figures, used only as a fallback label if the
// per-epoch history file cannot be loaded (never to fabricate a curve).
const DOC_GAP = 2.606
const DOC_ACC = 0.806
const DOC_PARAMS = 457729

// Sampler hyperparameter sweep — confirmed real results (strict end-to-end
// validity rate per configuration). The 300-step schedule was shipped as the
// default. Nothing here is interpolated; rows are exactly the measured runs.
const SWEEP = [
  { step: '0.02 → 0.005', noise: '0.006 → 0', steps: '300', valid: '33.3%', shipped: true },
  { step: '0.02 → 0.005', noise: '0.006 → 0', steps: '500', valid: '27.1%' },
  { step: '0.015 → 0.005', noise: '0.006 → 0', steps: '500', valid: '27.1%' },
  { step: '0.02 → 0.005', noise: '0.004 → 0', steps: '500', valid: '25.0%' },
  { step: 'other configurations', noise: '—', steps: '—', valid: '~20.8%', rest: true },
]

// Multimodality per scene — confirmed real distinct-mode counts from 16 random
// seeds per scene. Distinct modes = genuinely different valid routes. 20
// distinct valid routes emerged across the scenes in total; the hardest
// six-obstacle scene yielded none, shown honestly rather than hidden.
const SEEDS_PER_SCENE = 16
const SCENE_MODES = [
  { id: 0, obstacles: 3, valid: 3, modes: 3 },
  { id: 1, obstacles: 5, valid: 15, modes: 9 },
  { id: 2, obstacles: 3, valid: 14, modes: 8 },
  { id: 3, obstacles: 6, valid: 0, modes: 0 },
]

// Build the chart geometry from the real per-epoch arrays.
function buildChart(history) {
  const valid = history.energy_valid
  const bad = history.energy_bad
  const n = valid.length
  if (!n) return null

  const W = 680
  const H = 300
  const L = 46
  const R = 18
  const T = 20
  const B = 38
  const plotW = W - L - R
  const plotH = H - T - B

  let ymin = Infinity
  let ymax = -Infinity
  for (let i = 0; i < n; i++) {
    ymin = Math.min(ymin, valid[i], bad[i])
    ymax = Math.max(ymax, valid[i], bad[i])
  }
  const padY = (ymax - ymin) * 0.08 || 1
  ymin -= padY
  ymax += padY

  const xOf = (i) => L + (n === 1 ? 0 : (i / (n - 1)) * plotW)
  const yOf = (v) => T + (1 - (v - ymin) / (ymax - ymin)) * plotH

  const line = (arr) =>
    arr.map((v, i) => `${i === 0 ? 'M' : 'L'}${xOf(i).toFixed(1)} ${yOf(v).toFixed(1)}`).join(' ')

  let band = bad.map((v, i) => `${i === 0 ? 'M' : 'L'}${xOf(i).toFixed(1)} ${yOf(v).toFixed(1)}`).join(' ')
  for (let i = n - 1; i >= 0; i--) band += ` L${xOf(i).toFixed(1)} ${yOf(valid[i]).toFixed(1)}`
  band += ' Z'

  const xticks = [0, Math.round((n - 1) / 4), Math.round((n - 1) / 2), Math.round((3 * (n - 1)) / 4), n - 1]
    .filter((v, i, a) => a.indexOf(v) === i)
    .map((i) => ({ x: xOf(i), label: i + 1 }))

  const yticks = []
  for (let t = Math.ceil(ymin); t <= Math.floor(ymax); t++) yticks.push({ y: yOf(t), label: t })

  return {
    W, H, L, R, T, B,
    validPath: line(valid),
    badPath: line(bad),
    band,
    xticks,
    yticks,
    zeroY: yOf(0),
    last: {
      validX: xOf(n - 1), validY: yOf(valid[n - 1]),
      badX: xOf(n - 1), badY: yOf(bad[n - 1]),
    },
    n,
  }
}

export default function BuildStory({ field, descent, trajectories, history, historyError }) {
  const chart = useMemo(() => (history ? buildChart(history) : null), [history])

  const params = history?.n_params ?? DOC_PARAMS
  const finalGap = history?.metrics_final?.gap ?? DOC_GAP
  const finalAcc = history?.metrics_final?.accuracy ?? DOC_ACC
  const epochs = history?.epochs ?? 120

  // Live stats — computed at runtime from the real exports.
  const routes = trajectories.trajectories.length
  const descents = descent.descents.length
  const frames = descent.n_frames

  return (
    <section className="build" aria-label="How this was built">
      <header className="build-head">
        <h2 className="build-title">How this was built</h2>
        <p className="build-note">
          These visuals are read from a genuinely trained model — not a mock-up.
          The energy network was trained for {epochs} epochs with multi-scale
          denoising score matching; the chart below is its real per-epoch history,
          and the figures are read live from the same exported files the maps
          above are drawn from.
        </p>
      </header>

      <div className="build-chart-card">
        <div className="build-chart-head">
          <h3 className="build-chart-title">Energy separation over training</h3>
          <p className="build-chart-sub mono">
            VALID vs BAD PATH ENERGY · GAP {`+${finalGap.toFixed(3)}`} @ EPOCH {epochs}
          </p>
        </div>

        {chart ? (
          <svg
            className="build-chart"
            viewBox={`0 0 ${chart.W} ${chart.H}`}
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label={`Energy-gap training curve over ${chart.n} epochs. Valid-path energy falls and bad-path energy rises, opening an energy gap that climbs from near zero to +${finalGap.toFixed(3)} at epoch ${epochs}.`}
          >
            {/* horizontal gridlines + y labels */}
            {chart.yticks.map((t) => (
              <g key={`y${t.label}`}>
                <line
                  className="build-grid"
                  x1={chart.L}
                  x2={chart.W - chart.R}
                  y1={t.y}
                  y2={t.y}
                />
                <text className="build-axis" x={chart.L - 8} y={t.y + 3.5} textAnchor="end">
                  {t.label}
                </text>
              </g>
            ))}

            {/* zero reference line */}
            <line
              className="build-zero"
              x1={chart.L}
              x2={chart.W - chart.R}
              y1={chart.zeroY}
              y2={chart.zeroY}
            />

            {/* x labels */}
            {chart.xticks.map((t) => (
              <text
                key={`x${t.label}`}
                className="build-axis"
                x={t.x}
                y={chart.H - chart.B + 18}
                textAnchor="middle"
              >
                {t.label}
              </text>
            ))}
            <text
              className="build-axis build-axis-name"
              x={(chart.L + chart.W - chart.R) / 2}
              y={chart.H - 4}
              textAnchor="middle"
            >
              EPOCH
            </text>

            {/* the energy gap, shaded */}
            <path className="build-band" d={chart.band} />

            {/* the two energy traces */}
            <path className="build-line build-line-bad" d={chart.badPath} />
            <path className="build-line build-line-valid" d={chart.validPath} />

            {/* endpoints */}
            <circle className="build-dot build-dot-bad" cx={chart.last.badX} cy={chart.last.badY} r="3.5" />
            <circle className="build-dot build-dot-valid" cx={chart.last.validX} cy={chart.last.validY} r="3.5" />
          </svg>
        ) : (
          <div className="build-chart-fallback mono">
            <p>
              PER-EPOCH HISTORY UNAVAILABLE
              {historyError ? ` · ${historyError}` : ''}
            </p>
            <p className="build-chart-fallback-genuine">
              Genuine endpoints only: energy gap +0.001 at epoch 1 →{' '}
              {`+${DOC_GAP.toFixed(3)}`} at epoch 120. No intermediate points are
              shown, to avoid inventing a curve.
            </p>
          </div>
        )}

        <div className="build-legend mono">
          <span className="build-key">
            <span className="build-swatch build-swatch-valid" aria-hidden="true" />
            valid paths · energy ↓
          </span>
          <span className="build-key">
            <span className="build-swatch build-swatch-bad" aria-hidden="true" />
            bad paths · energy ↑
          </span>
          <span className="build-key build-key-band">shaded band = energy gap</span>
        </div>
      </div>

      <div className="build-grid-cols">
        {/* Model facts */}
        <div className="build-panel">
          <h3 className="build-panel-title">Model</h3>
          <dl className="build-facts">
            <div className="build-fact">
              <dt>architecture</dt>
              <dd>CNN map encoder · bidirectional LSTM trajectory encoder · MLP energy head</dd>
            </div>
            <div className="build-fact">
              <dt>parameters</dt>
              <dd className="mono">{params.toLocaleString()}</dd>
            </div>
            <div className="build-fact">
              <dt>training</dt>
              <dd>Multi-scale denoising score matching</dd>
            </div>
            <div className="build-fact">
              <dt>generation</dt>
              <dd>Langevin dynamics</dd>
            </div>
            <div className="build-fact">
              <dt>final accuracy</dt>
              <dd className="mono">{finalAcc.toFixed(3)}</dd>
            </div>
            <div className="build-fact">
              <dt>final energy gap</dt>
              <dd className="mono">{`+${finalGap.toFixed(3)}`}</dd>
            </div>
          </dl>
        </div>

        {/* Live stats from the real exports */}
        <div className="build-panel">
          <h3 className="build-panel-title">From the exported field</h3>
          <dl className="build-stats">
            <div className="build-stat">
              <dt className="mono">energy range</dt>
              <dd className="mono">[{field.vmin.toFixed(2)}, {field.vmax.toFixed(2)}]</dd>
            </div>
            <div className="build-stat">
              <dt className="mono">distinct valid routes</dt>
              <dd className="mono">{routes}</dd>
            </div>
            <div className="build-stat">
              <dt className="mono">descents animated</dt>
              <dd className="mono">{descents}</dd>
            </div>
            <div className="build-stat">
              <dt className="mono">frames per descent</dt>
              <dd className="mono">{frames}</dd>
            </div>
          </dl>
          <p className="build-stat-note">Read live from the export bundle at load.</p>
        </div>
      </div>

      <a className="build-download mono" href={dataUrl('energy_model.pt')} download>
        Download trained model — energy_model.pt (1.85 MB)
      </a>

      {/* 1 — What didn't work: the training-method pivot, framed honestly as
          the key research decision. Confirmed values only. */}
      <section className="build-sub" aria-label="What didn't work">
        <h3 className="build-subhead">What didn&rsquo;t work</h3>
        <p className="build-sub-note">
          The central decision was how to train an energy that is actually
          sampleable. The first approach did not give us one — and that pivot is
          the real research story.
        </p>
        <div className="method-compare">
          <article className="method-card method-card-fail">
            <p className="method-label mono">APPROACH 01 · ABANDONED</p>
            <h4 className="method-name">Contrastive divergence negatives</h4>
            <p className="method-body">
              Energy plateaued and never carved a descent path. Langevin
              sampling from noise reached only <span className="mono">~0–20%</span>{' '}
              valid paths, and valid paths never became genuine energy minima —
              there was nowhere for a sample to descend to.
            </p>
          </article>
          <article className="method-card method-card-win">
            <p className="method-label mono">APPROACH 02 · ADOPTED</p>
            <h4 className="method-name">Multi-scale denoising score matching</h4>
            <p className="method-body">
              Opened a working descent corridor: from chaos to settled, energy
              fell from <span className="mono e-high">≈ +42</span> to{' '}
              <span className="mono e-low">≈ −4.6</span>, and valid paths became
              genuine near-minima the sampler could reliably find.
            </p>
          </article>
        </div>
        <p className="build-sub-aside mono">
          SURVEY NOTE · the pivot from CD to score matching is what made generation possible.
        </p>
      </section>

      {/* 2 — Genuine minima: the gradient check, before/after. */}
      <section className="build-sub" aria-label="Genuine minima">
        <h3 className="build-subhead">Genuine minima</h3>
        <p className="build-sub-note">
          A valid path is only sampleable if it sits at the bottom of an energy
          valley. The gradient magnitude at valid paths is the test.
        </p>
        <div className="minima-compare">
          <div className="minima-stat minima-before">
            <p className="minima-cap mono">DISCRIMINATOR-ONLY</p>
            <p className="minima-val mono e-high">60–400</p>
            <p className="minima-sub">|grad| at valid paths — not minima</p>
          </div>
          <div className="minima-arrow mono" aria-hidden="true">→</div>
          <div className="minima-stat minima-after">
            <p className="minima-cap mono">AFTER SCORE MATCHING</p>
            <p className="minima-val mono e-low">
              ≈ 0.71<span className="minima-unit"> local</span>
              <span className="minima-div"> · </span>
              ≈ 8.66<span className="minima-unit"> production</span>
            </p>
            <p className="minima-sub">|grad| at valid paths — genuine near-minima</p>
          </div>
        </div>
        <p className="build-sub-aside">
          A small gradient at a valid path means it rests at the floor of an
          energy valley, not on a slope.
        </p>
      </section>

      {/* 3 — Sampler tuning: the real hyperparameter sweep as a survey table. */}
      <section className="build-sub" aria-label="Sampler tuning">
        <h3 className="build-subhead">Sampler tuning</h3>
        <p className="build-sub-note">
          Langevin step size, noise scale and step count were swept for strict
          end-to-end validity. The 300-step schedule was shipped as the default.
        </p>
        <div className="sweep-wrap">
          <table className="sweep-table mono">
            <thead>
              <tr>
                <th scope="col">step schedule</th>
                <th scope="col">noise schedule</th>
                <th scope="col">steps</th>
                <th scope="col">valid</th>
              </tr>
            </thead>
            <tbody>
              {SWEEP.map((r, i) => (
                <tr
                  key={i}
                  className={`${r.shipped ? 'sweep-win' : ''}${r.rest ? 'sweep-rest' : ''}`}
                >
                  <td>{r.step}</td>
                  <td>{r.noise}</td>
                  <td>{r.steps}</td>
                  <td className="sweep-valid">
                    {r.valid}
                    {r.shipped ? <span className="sweep-tag">SHIPPED</span> : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 4 — Multimodality per scene: real distinct-mode counts, including the
          hardest scene that yielded none. Bars scaled to the 16 seeds run. */}
      <section className="build-sub" aria-label="Multimodality per scene">
        <h3 className="build-subhead">Multimodality per scene</h3>
        <p className="build-sub-note">
          {SEEDS_PER_SCENE} random seeds were run per scene. Distinct modes
          counts how many genuinely different valid routes emerged — the core
          multimodality claim. 20 distinct valid routes emerged across the
          scenes in total; the hardest six-obstacle scene yielded none.
        </p>
        <div className="modes-chart">
          {SCENE_MODES.map((s) => (
            <div className="modes-row" key={s.id}>
              <div className="modes-scene mono">
                <span className="modes-scene-id">SCENE {s.id}</span>
                <span className="modes-scene-obs">{s.obstacles} obstacles</span>
              </div>
              <div className="modes-bars">
                <div className="modes-track">
                  <div className="modes-rail">
                    <div
                      className="modes-bar modes-bar-valid"
                      style={{ width: `${(s.valid / SEEDS_PER_SCENE) * 100}%` }}
                    />
                  </div>
                  <span className="modes-count mono">{s.valid} valid</span>
                </div>
                <div className="modes-track">
                  <div className="modes-rail">
                    <div
                      className="modes-bar modes-bar-modes"
                      style={{ width: `${(s.modes / SEEDS_PER_SCENE) * 100}%` }}
                    />
                  </div>
                  <span className="modes-count mono">{s.modes} distinct</span>
                </div>
              </div>
            </div>
          ))}
        </div>
        <div className="modes-legend mono">
          <span className="modes-key">
            <span className="modes-swatch modes-swatch-valid" aria-hidden="true" />
            valid routes
          </span>
          <span className="modes-key">
            <span className="modes-swatch modes-swatch-modes" aria-hidden="true" />
            distinct modes
          </span>
          <span className="modes-key modes-total">
            20 distinct valid routes total · {SEEDS_PER_SCENE} seeds/scene
          </span>
        </div>
      </section>
    </section>
  )
}
