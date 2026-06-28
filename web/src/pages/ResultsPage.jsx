import { Link } from 'react-router-dom'
import BuildStory from '../components/BuildStory'

/*
 * ResultsPage (/results) — the standalone method / credibility page.
 *
 * A second page of the same field-survey document: same shell (.instrument),
 * CSS variables, fonts and terrain aesthetic as the landscape. It opens with a
 * Back control, then flows top to bottom through the heading, intro, the real
 * training curve, model facts, live stats, and the model download (all in
 * BuildStory). Data is shared from App, so the live stats match the landscape.
 */
export default function ResultsPage({
  field,
  descent,
  trajectories,
  history,
  historyError,
  ready,
  error,
}) {
  return (
    <main className="instrument">
      <header className="results-head">
        <Link className="back-link mono" to="/">
          Back to the landscape
        </Link>
        <p className="eyebrow mono">ENERGY LANDSCAPE · METHOD</p>
      </header>

      {ready ? (
        <BuildStory
          field={field}
          descent={descent}
          trajectories={trajectories}
          history={history}
          historyError={historyError}
        />
      ) : (
        <div className="hero-placeholder mono" role="status">
          {error ? `FIELD UNAVAILABLE · ${error}` : 'READING FIELD…'}
        </div>
      )}
    </main>
  )
}
