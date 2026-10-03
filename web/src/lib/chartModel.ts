// Pure mapping from the exported chart JSON to drawable layers.
//
// Mirrors signalcheck/ui/charts.py (the Streamlit/Plotly chart) so both frontends
// draw exactly what the checks saw: recent-window shading, the baseline band,
// threshold lines (an upper threshold far above the data is omitted), highlighted
// points, fitted segments, the seasonal baseline, change-point lines, hollow
// imputed points and the greyed-out dropped partial period.

import type { Annotation, ChartData, Freq } from '../types'

export const COLORS = {
  series: '#1f4e79',
  seriesDark: '#7cb7f0',
  band: 'rgba(31, 119, 180, 0.12)',
  bandDark: 'rgba(96, 165, 250, 0.16)',
  recent: 'rgba(255, 193, 7, 0.16)',
  recentDark: 'rgba(250, 204, 21, 0.10)',
  threshold: '#9e9e9e',
  changePoint: '#6a1b9a',
  changePointDark: '#c084fc',
  partial: '#9e9e9e',
  seasonal: '#8e24aa',
} as const

export type MarkerSymbol = 'circle' | 'diamond' | 'x' | 'square'
export type Dash = 'solid' | 'dash' | 'dot'

export const POINT_STYLES: Record<string, { color: string; symbol: MarkerSymbol; size: number }> = {
  'beyond baseline band': { color: '#2e7d32', symbol: 'circle', size: 8 },
  'largest excess': { color: '#ef6c00', symbol: 'diamond', size: 10 },
  outlier: { color: '#c62828', symbol: 'x', size: 10 },
  'single-origin bucket': { color: '#8e24aa', symbol: 'square', size: 8 },
}
export const DEFAULT_POINT_STYLE = { color: '#c62828', symbol: 'circle' as MarkerSymbol, size: 8 }

export const SEGMENT_STYLES: Record<string, { color: string; dash: Dash; visible: boolean }> = {
  'Theil-Sen trend': { color: '#2e7d32', dash: 'solid', visible: true },
  'level before': { color: '#6a1b9a', dash: 'dash', visible: true },
  'level after': { color: '#6a1b9a', dash: 'dash', visible: true },
  'baseline mean rate': { color: '#616161', dash: 'dot', visible: false },
  'recent mean rate': { color: '#ef6c00', dash: 'dot', visible: false },
}
export const DEFAULT_SEGMENT_STYLE = { color: '#616161', dash: 'dot' as Dash, visible: false }

export interface XY {
  t: number
  v: number
}

export interface SpanShape {
  group: string
  x0: number
  x1: number
}

export type BandShape =
  | { kind: 'rect'; group: string; x0: number; x1: number; y0: number; y1: number }
  | { kind: 'hline'; group: string; x0: number; x1: number; y: number; dash: Dash }

export interface PointGroup {
  group: string
  color: string
  symbol: MarkerSymbol
  size: number
  points: (XY & { label: string })[]
}

export interface SegmentShape {
  group: string
  color: string
  dash: Dash
  from: XY
  to: XY
}

export interface LineShape {
  group: string
  points: XY[]
}

export interface LegendEntry {
  id: string
  label: string
  kind: 'span' | 'band' | 'threshold' | 'series' | 'imputed' | 'line' | 'segment' | 'points' | 'vline' | 'partial'
  color?: string
  dash?: Dash
  symbol?: MarkerSymbol
  visible: boolean
}

export interface ChartModel {
  freq: Freq
  seriesLabel: string
  series: { t: number; v: number | null }[]
  imputed: XY[]
  partial: { from: XY | null; to: XY } | null
  spans: SpanShape[]
  bands: BandShape[]
  points: PointGroup[]
  segments: SegmentShape[]
  lines: LineShape[]
  vlines: { group: string; t: number }[]
  peak: number | null
  ceiling: number | null
  xDomain: [number, number]
  yDomain: [number, number]
  legend: LegendEntry[]
}

const DAY = 86_400_000

/** Milliseconds since the epoch (UTC) for an ISO day ``YYYY-MM-DD`` (time part ignored). */
export function toTime(day: string): number {
  const [y, m, d] = day.slice(0, 10).split('-').map(Number) as [number, number, number]
  return Date.UTC(y, m - 1, d)
}

/** Exclusive end of the period starting at ``day`` (charts.py ``_period_end``). */
export function periodEnd(day: string, freq: Freq): number {
  const start = toTime(day)
  if (freq === 'D') return start + DAY
  if (freq === 'W') return start + 7 * DAY
  const d = new Date(start)
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + 1, 1)
}

function itemKey(item: object): string {
  return Object.entries(item)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join('|')
}

/** All annotation entries of one kind across the evidence, de-duplicated, in order. */
export function annotationItems<K extends keyof Annotation>(
  annotations: Annotation[],
  kind: K,
): NonNullable<Annotation[K]> {
  const seen = new Set<string>()
  const items: object[] = []
  for (const ann of annotations) {
    for (const item of (ann[kind] ?? []) as object[]) {
      const key = itemKey(item)
      if (!seen.has(key)) {
        seen.add(key)
        items.push(item)
      }
    }
  }
  return items as NonNullable<Annotation[K]>
}

/** Legend group for a highlighted point: ``top 1 excess``, ``top 2 excess`` -> one entry. */
export function pointGroup(label: string): string {
  return /^top \d+ excess$/.test(label) ? 'largest excess' : label
}

const finite = (v: number | null | undefined): v is number => typeof v === 'number' && Number.isFinite(v)

/** Highest value the chart plots (observed, imputed, highlighted, fitted or baseline band). */
export function dataPeak(chart: ChartData): number | null {
  const a = chart.annotations
  const values: (number | null)[] = [
    ...chart.observed.map(([, v]) => v),
    ...chart.imputed.map(([, v]) => v),
    ...annotationItems(a, 'points').map((p) => p.value),
    ...annotationItems(a, 'segments').flatMap((s) => [s.start_value, s.end_value]),
    ...annotationItems(a, 'lines').flatMap((l) => (l.points ?? []).map((p) => p.value)),
    ...annotationItems(a, 'bands')
      .filter((b) => b.label === 'baseline band')
      .map((b) => b.upper),
  ]
  const ok = values.filter(finite)
  return ok.length ? Math.max(...ok) : null
}

/** Build every layer of the annotated chart (see charts.py ``build_chart``). */
export function buildChartModel(chart: ChartData, thresholdHeadroom: number | null = null): ChartModel {
  const { freq, annotations: a } = chart
  const legend: LegendEntry[] = []
  const addLegend = (entry: LegendEntry) => {
    if (!legend.some((e) => e.id === entry.id)) legend.push(entry)
  }

  // Recent window (falls back to the verdict window).
  let spanItems = annotationItems(a, 'spans')
  if (!spanItems.length && chart.window) {
    spanItems = [{ label: 'recent window', start: chart.window.recent_start, end: chart.window.recent_end }]
  }
  const spans: SpanShape[] = spanItems.map((s) => ({
    group: s.label ?? 'recent window',
    x0: toTime(s.start),
    x1: periodEnd(s.end, freq),
  }))
  for (const s of spans) addLegend({ id: `span:${s.group}`, label: s.group, kind: 'span', visible: true })

  const peak = dataPeak(chart)
  const ceiling = peak !== null && thresholdHeadroom ? peak * thresholdHeadroom : null

  // Bands: baseline band (filled), thresholds (dotted), equal bounds (dashed).
  const bands: BandShape[] = []
  for (const band of annotationItems(a, 'bands')) {
    const group = band.label ?? 'band'
    const x0 = toTime(band.start)
    const x1 = periodEnd(band.end, freq)
    const { lower, upper } = band
    if (group === 'baseline band' && upper > lower) {
      bands.push({ kind: 'rect', group, x0, x1, y0: lower, y1: upper })
      addLegend({ id: `band:${group}`, label: group, kind: 'band', visible: true })
    } else if (lower === upper) {
      bands.push({ kind: 'hline', group, x0, x1, y: lower, dash: 'dash' })
      addLegend({ id: `band:${group}`, label: group, kind: 'threshold', dash: 'dash', visible: true })
    } else {
      const drawUpper = ceiling === null || upper <= ceiling
      if (drawUpper) bands.push({ kind: 'hline', group, x0, x1, y: upper, dash: 'dot' })
      if (lower > 0) bands.push({ kind: 'hline', group, x0, x1, y: lower, dash: 'dot' })
      if (drawUpper || lower > 0) {
        addLegend({ id: `band:${group}`, label: group, kind: 'threshold', dash: 'dot', visible: true })
      }
    }
  }

  // Observed series (gaps break the line) and hollow imputed points.
  const series = chart.observed.map(([d, v]) => ({ t: toTime(d), v: finite(v) ? v : null }))
  const seriesLabel = chart.scale.replaceAll('_', ' ')
  addLegend({ id: 'series', label: seriesLabel, kind: 'series', visible: true })
  const imputed = chart.imputed.filter(([, v]) => finite(v)).map(([d, v]) => ({ t: toTime(d), v: v as number }))
  if (imputed.length) addLegend({ id: 'imputed', label: 'imputed (not observed)', kind: 'imputed', visible: true })

  // Seasonal baseline and other multi-point reference lines.
  const lines: LineShape[] = []
  for (const line of annotationItems(a, 'lines')) {
    const pts = (line.points ?? []).filter((p) => finite(p.value)).map((p) => ({ t: toTime(p.ts), v: p.value }))
    if (!pts.length) continue
    const group = line.label ?? 'reference'
    lines.push({ group, points: pts })
    addLegend({ id: `line:${group}`, label: group, kind: 'line', color: COLORS.seasonal, dash: 'dot', visible: true })
  }

  // Fitted segments (mean rates hidden by default, as in Plotly's legendonly).
  const segments: SegmentShape[] = []
  for (const seg of annotationItems(a, 'segments')) {
    const group = seg.label ?? 'segment'
    const style = SEGMENT_STYLES[group] ?? DEFAULT_SEGMENT_STYLE
    segments.push({
      group,
      color: style.color,
      dash: style.dash,
      from: { t: toTime(seg.start), v: seg.start_value },
      to: { t: toTime(seg.end), v: seg.end_value },
    })
    addLegend({ id: `segment:${group}`, label: group, kind: 'segment', color: style.color, dash: style.dash, visible: style.visible })
  }

  // Highlighted points, one group per legend entry.
  const byGroup = new Map<string, PointGroup>()
  for (const p of annotationItems(a, 'points')) {
    if (!finite(p.value)) continue
    const label = p.label ?? 'highlighted'
    const group = pointGroup(label)
    let entry = byGroup.get(group)
    if (!entry) {
      const style = POINT_STYLES[group] ?? DEFAULT_POINT_STYLE
      entry = { group, ...style, points: [] }
      byGroup.set(group, entry)
    }
    entry.points.push({ t: toTime(p.ts), v: p.value, label })
  }
  const points = [...byGroup.values()]
  for (const g of points) {
    addLegend({ id: `points:${g.group}`, label: g.group, kind: 'points', color: g.color, symbol: g.symbol, visible: true })
  }

  // Change points.
  const vlines = annotationItems(a, 'vlines').map((v) => ({ group: v.label ?? 'change point', t: toTime(v.ts) }))
  for (const v of vlines) addLegend({ id: `vline:${v.group}`, label: v.group, kind: 'vline', visible: true })

  // Dropped partial period: grey marker linked from the last observed value.
  let partial: ChartModel['partial'] = null
  if (chart.dropped_partial && finite(chart.dropped_partial.value)) {
    const last = [...series].reverse().find((p) => p.v !== null)
    partial = {
      from: last ? { t: last.t, v: last.v as number } : null,
      to: { t: toTime(chart.dropped_partial.ts), v: chart.dropped_partial.value },
    }
    addLegend({ id: 'partial', label: 'partial period (dropped)', kind: 'partial', visible: true })
  }

  // Domains: x covers every drawn element; y starts at zero (Plotly rangemode tozero).
  const xs: number[] = [
    ...series.map((p) => p.t),
    ...imputed.map((p) => p.t),
    ...spans.flatMap((s) => [s.x0, s.x1]),
    ...vlines.map((v) => v.t),
    ...(partial ? [partial.to.t] : []),
  ]
  const ys: number[] = [
    ...series.map((p) => p.v).filter(finite),
    ...imputed.map((p) => p.v),
    ...bands.flatMap((b) => (b.kind === 'rect' ? [b.y0, b.y1] : [b.y])),
    ...lines.flatMap((l) => l.points.map((p) => p.v)),
    ...segments.flatMap((s) => [s.from.v, s.to.v]),
    ...points.flatMap((g) => g.points.map((p) => p.v)),
    ...(partial ? [partial.to.v] : []),
  ].filter(finite)
  const xDomain: [number, number] = xs.length ? [Math.min(...xs), Math.max(...xs)] : [0, 1]
  const yMax = ys.length ? Math.max(0, ...ys) : 1
  const yMin = ys.length ? Math.min(0, ...ys) : 0
  const pad = (yMax - yMin || 1) * 0.06
  const yDomain: [number, number] = [yMin < 0 ? yMin - pad : 0, yMax + pad]

  return {
    freq,
    seriesLabel,
    series,
    imputed,
    partial,
    spans,
    bands,
    points,
    segments,
    lines,
    vlines,
    peak,
    ceiling,
    xDomain,
    yDomain,
    legend,
  }
}

/** Legend id for a drawable element's group (used to toggle visibility). */
export const legendId = {
  span: (g: string) => `span:${g}`,
  band: (g: string) => `band:${g}`,
  line: (g: string) => `line:${g}`,
  segment: (g: string) => `segment:${g}`,
  points: (g: string) => `points:${g}`,
  vline: (g: string) => `vline:${g}`,
}
