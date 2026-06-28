import { useEffect, useState } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import HomePage from './pages/HomePage'
import ResultsPage from './pages/ResultsPage'
import { fetchJSON } from './lib/bundle'
import './App.css'

/*
 * App — the router root and single data owner.
 *
 * The export bundle is fetched once here and shared with both pages, so
 * navigating between the landscape (/) and the method page (/results) never
 * refetches and the live stats stay identical across pages. Routing is
 * client-side (HashRouter, see main.jsx).
 */

// Reset scroll to the top on every route change (instant — route navigation is
// not a scroll, and the global reduced-motion guard makes smooth scrolling moot
// anyway).
function ScrollToTop() {
  const { pathname } = useLocation()
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])
  return null
}

function App() {
  const [field, setField] = useState(null)
  const [descent, setDescent] = useState(null)
  const [trajectories, setTrajectories] = useState(null)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState(null)
  const [historyError, setHistoryError] = useState(null)

  useEffect(() => {
    let alive = true
    Promise.all([
      fetchJSON('energy_field.json'),
      fetchJSON('descent.json'),
      fetchJSON('trajectories.json'),
    ])
      .then(([f, d, t]) => {
        if (!alive) return
        setField(f)
        setDescent(d)
        setTrajectories(t)
      })
      .catch((e) => alive && setError(e.message))
    return () => {
      alive = false
    }
  }, [])

  // The per-epoch training history is loaded separately so a missing/failed
  // history file degrades only the method page's chart, not the whole app.
  useEffect(() => {
    let alive = true
    fetchJSON('training_history.json')
      .then((h) => alive && setHistory(h))
      .catch((e) => alive && setHistoryError(e.message))
    return () => {
      alive = false
    }
  }, [])

  const ready = field && descent && trajectories

  return (
    <>
      <ScrollToTop />
      <Routes>
        <Route
          path="/"
          element={
            <HomePage
              field={field}
              descent={descent}
              trajectories={trajectories}
              ready={ready}
              error={error}
            />
          }
        />
        <Route
          path="/results"
          element={
            <ResultsPage
              field={field}
              descent={descent}
              trajectories={trajectories}
              history={history}
              historyError={historyError}
              ready={ready}
              error={error}
            />
          }
        />
      </Routes>
    </>
  )
}

export default App
