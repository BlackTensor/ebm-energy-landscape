/*
 * Loads the exported data bundle the web app consumes.
 *
 * The four JSON files (scene, energy_field, trajectories, descent) are written
 * by the training side into /exports, staged under public/data, and validated
 * against a fixed schema (training/validate_export.py) before they ship — so the
 * front end can read them without defensive parsing.
 */

// Resolve against Vite's base URL so paths work under any GitHub Pages path.
export const dataUrl = (name) => `${import.meta.env.BASE_URL}data/${name}`

export async function fetchJSON(name) {
  const res = await fetch(dataUrl(name))
  if (!res.ok) throw new Error(`Could not load ${name} (HTTP ${res.status})`)
  return res.json()
}
