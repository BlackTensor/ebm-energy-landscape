import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'

// Self-hosted type faces (bundled by Vite — no external CDN, works offline and
// on GitHub Pages). The field-survey identity pairs a characterful old-style
// serif for display, a refined document serif for body, and monospace for data.
// Weights are scoped to what the design actually uses.
import '@fontsource/fraunces/500.css'
import '@fontsource/fraunces/600.css'
import '@fontsource/fraunces/700.css'
import '@fontsource/fraunces/600-italic.css'
import '@fontsource/spectral/400.css'
import '@fontsource/spectral/500.css'
import '@fontsource/spectral/600.css'
import '@fontsource/spectral/400-italic.css'
import '@fontsource/jetbrains-mono/400.css'
import '@fontsource/jetbrains-mono/500.css'

import './styles/theme.css'
import './index.css'
import App from './App.jsx'

// HashRouter keeps client-side routing working on GitHub Pages static hosting
// (no server rewrites needed) and under any base path.
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </StrictMode>,
)
