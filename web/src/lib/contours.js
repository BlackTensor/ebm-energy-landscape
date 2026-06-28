/*
 * Iso-energy contour extraction (marching squares).
 *
 * Given the field's normalized energy grid (rows = y, cols = x, values in [0,1])
 * this returns iso-energy line segments at evenly spaced thresholds — the same
 * idea as elevation contours on a survey map. Because normalized energy is a
 * linear remap of the model's real energy, evenly spaced normalized levels are
 * evenly spaced *real* energy thresholds; the lines are computed from the actual
 * grid values, never faked.
 *
 * Segments are returned in grid coordinates (x = column, y = row, both as
 * fractional sample indices). The renderer maps them onto the canvas with the
 * same orientation it paints the field, so contours land exactly on the colour
 * bands they describe.
 */

// Evenly spaced interior thresholds (0 and 1 are the field extremes, skipped).
export function contourLevels(count = 11) {
  const levels = []
  for (let i = 1; i <= count; i++) levels.push(i / (count + 1))
  return levels
}

// Linear crossing position where the iso-value sits between two corner samples.
const cross = (v0, v1, t) => (v0 === v1 ? 0.5 : (t - v0) / (v1 - v0))

/*
 * Marching squares over `grid` at one threshold `t`. For each cell, the four
 * corners are classified above/below `t`; the case index selects which cell
 * edges the iso-line crosses, and the crossing points are linearly interpolated.
 * Saddle cases (5, 10) emit both segments — acceptable for fine, subtle lines.
 */
function marchLevel(grid, t, out) {
  const rows = grid.length
  const cols = grid[0].length
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const tl = grid[r][c]
      const tr = grid[r][c + 1]
      const br = grid[r + 1][c + 1]
      const bl = grid[r + 1][c]

      const idx =
        (tl >= t ? 8 : 0) | (tr >= t ? 4 : 0) | (br >= t ? 2 : 0) | (bl >= t ? 1 : 0)
      if (idx === 0 || idx === 15) continue

      // Edge crossing points, in grid coords (x = col, y = row).
      const top = [c + cross(tl, tr, t), r]
      const right = [c + 1, r + cross(tr, br, t)]
      const bottom = [c + cross(bl, br, t), r + 1]
      const left = [c, r + cross(tl, bl, t)]

      const seg = (a, b) => out.push(a[0], a[1], b[0], b[1])

      switch (idx) {
        case 1: seg(left, bottom); break
        case 2: seg(bottom, right); break
        case 3: seg(left, right); break
        case 4: seg(top, right); break
        case 5: seg(top, left); seg(bottom, right); break
        case 6: seg(top, bottom); break
        case 7: seg(top, left); break
        case 8: seg(top, left); break
        case 9: seg(top, bottom); break
        case 10: seg(top, right); seg(left, bottom); break
        case 11: seg(top, right); break
        case 12: seg(left, right); break
        case 13: seg(bottom, right); break
        case 14: seg(left, bottom); break
        default: break
      }
    }
  }
}

// All contour segments across every threshold, as a flat Float32Array of
// [x0, y0, x1, y1, ...] in grid coordinates.
export function computeContours(grid, levels = contourLevels()) {
  const out = []
  for (const t of levels) marchLevel(grid, t, out)
  return Float32Array.from(out)
}
