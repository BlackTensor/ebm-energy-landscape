/*
 * The energy colormap — a topographic elevation ramp.
 *
 * Maps normalized energy (0 = valley / low / valid, 1 = ridge / high / invalid)
 * to colour through a continuous hypsometric terrain ramp: deep water-blue lows,
 * rising through teal, moss and sage, into warm sand/parchment mid-tones, up to
 * terracotta/clay-red highs. The ramp is strictly monotonic in energy, so the
 * low-to-high meaning is preserved exactly — but the field now reads as designed
 * terrain that belongs on the paper rather than a dark rectangle.
 *
 * Interpolation happens in linear-light space (sRGB is decoded, mixed, then
 * re-encoded) so the elevation bands stay perceptually smooth and never pass
 * through muddy transitions.
 */

// Terrain gradient stops: [position 0..1, sRGB [r, g, b]], low energy -> high.
const STOPS = [
  [0.0, [0x14, 0x32, 0x4e]], // deep water-blue — lowest valleys
  [0.16, [0x1f, 0x5e, 0x63]], // teal shallows
  [0.34, [0x3f, 0x7a, 0x55]], // moss
  [0.5, [0x8a, 0x9b, 0x58]], // sage uplands
  [0.66, [0xd8, 0xc0, 0x84]], // warm sand / parchment
  [0.83, [0xcf, 0x8b, 0x5a]], // ochre slopes
  [1.0, [0xb3, 0x4a, 0x2f]], // terracotta / clay — highest ridges
]

const srgbToLinear = (c) => {
  const x = c / 255
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4
}

const linearToSrgb = (x) => {
  const v = x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055
  return Math.round(Math.min(1, Math.max(0, v)) * 255)
}

const LINEAR_STOPS = STOPS.map(([p, rgb]) => [p, rgb.map(srgbToLinear)])

function sampleLinear(stops, t) {
  const clamped = t < 0 ? 0 : t > 1 ? 1 : t
  let i = 1
  while (i < stops.length - 1 && clamped > stops[i][0]) i++
  const [p0, c0] = stops[i - 1]
  const [p1, c1] = stops[i]
  const f = p1 === p0 ? 0 : (clamped - p0) / (p1 - p0)
  return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f]
}

// 256-entry lookup table ([r, g, b] per entry), built once and reused.
let lut = null

export function colormapLUT() {
  if (lut) return lut
  lut = new Uint8ClampedArray(256 * 3)
  for (let i = 0; i < 256; i++) {
    const lin = sampleLinear(LINEAR_STOPS, i / 255)
    lut[i * 3] = linearToSrgb(lin[0])
    lut[i * 3 + 1] = linearToSrgb(lin[1])
    lut[i * 3 + 2] = linearToSrgb(lin[2])
  }
  return lut
}

// A CSS linear-gradient sampled from the same terrain LUT, so the legend bar
// matches the canvas exactly. Sweeps low -> high energy across `steps` colours.
export function colormapCss(steps = 24, angle = '90deg') {
  const table = colormapLUT()
  const parts = []
  for (let s = 0; s <= steps; s++) {
    const t = s / steps
    const i = Math.round(t * 255) * 3
    parts.push(`rgb(${table[i]}, ${table[i + 1]}, ${table[i + 2]}) ${(t * 100).toFixed(1)}%`)
  }
  return `linear-gradient(${angle}, ${parts.join(', ')})`
}

/*
 * The descending-path tint — a dedicated accent ramp, NOT the terrain LUT.
 *
 * A path is tinted by its current energy as it descends: ridge-orange while
 * chaotic (high energy), cooling to valley-green once settled (low energy).
 * It uses the two committed energy accents directly rather than sampling the
 * terrain ramp, so a settled green path never dissolves into a green valley and
 * a chaotic path never washes out against the clay highs — the stroke stays
 * legible over every terrain band (paired with a dark casing underneath).
 */
const PATH_STOPS = [
  [0.0, [0x3d, 0xd2, 0xa4]], // --valley: settled / low energy
  [1.0, [0xe8, 0x63, 0x3a]], // --ridge: chaotic / high energy
].map(([p, rgb]) => [p, rgb.map(srgbToLinear)])

// A single colour for a normalized energy (0..1) along the descent ramp.
export function colorForEnergy01(t) {
  const lin = sampleLinear(PATH_STOPS, t)
  return `rgb(${linearToSrgb(lin[0])}, ${linearToSrgb(lin[1])}, ${linearToSrgb(lin[2])})`
}
