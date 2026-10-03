// JSON contract written by signalcheck/web/export.py (SCHEMA_VERSION = 1).

export type BadgeColor = 'green' | 'red' | 'orange' | 'violet' | 'blue' | 'gray'
export type LabelKey = 'TREND_up' | 'TREND_down' | 'FLUKE' | 'SEASONAL' | 'NO_CHANGE' | 'INCONCLUSIVE'

export interface Badge {
  text: string
  color: BadgeColor
  key: LabelKey
}

export interface EvidenceItem {
  check: string
  name: string
  stance: string
  stance_text: string
  summary: string
}

export interface Window {
  baseline_start: string
  baseline_end: string
  recent_start: string
  recent_end: string
  [key: string]: unknown
}

export interface SpanAnn {
  label?: string
  start: string
  end: string
}
export interface BandAnn {
  label?: string
  start: string
  end: string
  lower: number
  upper: number
}
export interface PointAnn {
  label?: string
  ts: string
  value: number
}
export interface SegmentAnn {
  label?: string
  start: string
  end: string
  start_value: number
  end_value: number
}
export interface LineAnn {
  label?: string
  points?: { ts: string; value: number }[]
}
export interface VlineAnn {
  label?: string
  ts: string
}

export interface Annotation {
  spans?: SpanAnn[]
  bands?: BandAnn[]
  points?: PointAnn[]
  segments?: SegmentAnn[]
  lines?: LineAnn[]
  vlines?: VlineAnn[]
}

export type Freq = 'D' | 'W' | 'M'

export interface ChartData {
  freq: Freq
  scale: string
  observed: [string, number | null][]
  imputed: [string, number | null][]
  dropped_partial: { ts: string; value: number } | null
  window: Window | null
  annotations: Annotation[]
}

export interface Card {
  source: string
  title: string
  message: string | null
  disabled: boolean
  snapshot: boolean
  /** When this source's data was fetched (ISO, UTC); null for disabled sources. */
  fetched_at: string | null
  badge: Badge | null
  label: string | null
  direction: string | null
  confidence: string | null
  rule: string | null
  rule_text: string | null
  reason: string | null
  window_text: string | null
  evidence: EvidenceItem[]
  skipped: { name: string; reason: string }[]
  change_my_mind: string[]
  caveats: string[]
  wiki: { article: string; overridden: boolean; candidates: string[] } | null
  narration: { text: string; label: string } | null
  chart: ChartData | null
}

export interface Summary {
  sentence: string
  chips: { source: string; title: string; badge: Badge }[]
  unavailable: string[]
  not_comparable: string[]
}

export interface Result {
  schema: number
  kind: 'topic' | 'csv'
  query: string
  sample: boolean
  generated_at: string
  summary: Summary | null
  cards: Card[]
}

export interface ColumnChoice {
  schema: number
  kind: 'columns'
  columns: string[]
}

export interface SiteSource {
  source: string
  title: string
  live: boolean
  reason: string | null
}

export interface Site {
  schema: number
  version: string
  generated_at: string
  methodology: { title: string; body: string }[]
  rules: Record<string, string>
  labels: Record<string, { text: string; color: BadgeColor }>
  sources: SiteSource[]
  ui: {
    timeframe_days: number[]
    chart_height_px: number
    threshold_headroom: number
    history_days: Record<string, number>
    template_label: string
  }
  samples: { query: string; slug: string; sources: string[] }[]
  /** Last daily data refresh (data/manifest.json); null before the first run. */
  data: DataFreshness | null
}

export interface DataFreshness {
  refreshed_at: string
  status: 'ok' | 'partial' | 'failed' | 'skipped'
  sources: Record<string, string>
}

export interface PyManifest {
  bundle: string
  bundle_bytes: number
  wheels: string[]
  pyodide_packages: string[]
}
