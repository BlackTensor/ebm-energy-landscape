import { Link } from 'react-router-dom'
import DescentAnimation from '../components/DescentAnimation'
import MultiSample from '../components/MultiSample'
import Comparison from '../components/Comparison'
import { colormapCss } from '../lib/colormap'

/*
 * HomePage (/) — the energy landscape visualizations.
 *
 * The signature heatmap is the stage for the descent animation; the multi-sample
 * view overlays the distinct valid routes to make multimodality plain; the
 * comparison panel lands the one-vs-many thesis for a non-expert. The method /
 * credibility material now lives on its own page (/results); a prominent
 * instrument-style control in the hero navigates there.
 */
export default function HomePage({ field, descent, trajectories, ready, error }) {
  return (
    <main className="instrument">
      <header className="masthead">
        <p className="eyebrow mono">ENERGY-BASED MODEL · FIELD INSTRUMENT</p>
        <h1 className="title">Energy Landscape</h1>
        <p className="lede">
          A model that learns a landscape of every valid future, then lets
          chaotic paths descend into its valleys — instead of predicting one
          answer.
        </p>
        <p className="byline mono">Architected by Shayan Ansari · 2026</p>
      </header>

      <Link className="jump-build mono" to="/results">
        View Results &amp; Architecture
      </Link>

      <section className="hero" aria-label="Energy landscape">
        {ready ? (
          <DescentAnimation field={field} descent={descent} />
        ) : (
          <div className="hero-placeholder mono" role="status">
            {error ? `FIELD UNAVAILABLE · ${error}` : 'READING FIELD…'}
          </div>
        )}

        <div className="scale" aria-hidden="true">
          <span className="scale-end">valley · low</span>
          <span className="scale-bar" style={{ backgroundImage: colormapCss() }} />
          <span className="scale-end scale-end-high">high · ridge</span>
        </div>
      </section>

      {ready && (
        <section className="sample" aria-label="Multimodality">
          <div className="sample-head">
            <h2 className="sample-title">Many valid futures</h2>
            <p className="sample-note">
              The model does not pick one route. Here are{' '}
              <span className="mono">{trajectories.count}</span> distinct valid
              trajectories for the same scene — each settling into its own valley,
              all avoiding the same obstacles.
            </p>
          </div>
          <MultiSample field={field} trajectories={trajectories} />
          <p className="sample-caption mono">
            {trajectories.count} DISTINCT ROUTES · ONE SCENE
          </p>
        </section>
      )}

      {ready && <Comparison field={field} trajectories={trajectories} />}

      <footer className="status mono">
        <span>SCENE 00</span>
        <span>{field ? `FIELD ${field.resolution}×${field.resolution}` : 'FIELD —'}</span>
        <span>
          {field
            ? `E [${field.vmin.toFixed(2)}, ${field.vmax.toFixed(2)}]`
            : 'E —'}
        </span>
      </footer>
    </main>
  )
}
