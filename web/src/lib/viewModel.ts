// Presentation helpers shared by the cards (pure, unit-tested).

import type { BadgeColor, Card, LabelKey, Result } from '../types'

/** Tailwind classes per verdict colour (trend up green, down red, fluke orange, ...). */
export const BADGE_CLASSES: Record<BadgeColor, string> = {
  green: 'bg-emerald-50 text-emerald-800 ring-emerald-600/25 dark:bg-emerald-400/10 dark:text-emerald-300 dark:ring-emerald-400/30',
  red: 'bg-rose-50 text-rose-800 ring-rose-600/25 dark:bg-rose-400/10 dark:text-rose-300 dark:ring-rose-400/30',
  orange: 'bg-orange-50 text-orange-800 ring-orange-600/25 dark:bg-orange-400/10 dark:text-orange-300 dark:ring-orange-400/30',
  violet: 'bg-violet-50 text-violet-800 ring-violet-600/25 dark:bg-violet-400/10 dark:text-violet-300 dark:ring-violet-400/30',
  blue: 'bg-sky-50 text-sky-800 ring-sky-600/25 dark:bg-sky-400/10 dark:text-sky-300 dark:ring-sky-400/30',
  gray: 'bg-slate-100 text-slate-700 ring-slate-500/25 dark:bg-slate-400/10 dark:text-slate-300 dark:ring-slate-400/30',
}

/** Accent bar colour on each card, matching the badge. */
export const ACCENT_CLASSES: Record<BadgeColor, string> = {
  green: 'bg-emerald-500',
  red: 'bg-rose-500',
  orange: 'bg-orange-500',
  violet: 'bg-violet-500',
  blue: 'bg-sky-500',
  gray: 'bg-slate-400',
}

/** Verdict key -> badge colour (mirrors signalcheck.ui.view_models.LABEL_STYLES). */
export const LABEL_COLORS: Record<LabelKey, BadgeColor> = {
  TREND_up: 'green',
  TREND_down: 'red',
  FLUKE: 'orange',
  SEASONAL: 'violet',
  NO_CHANGE: 'blue',
  INCONCLUSIVE: 'gray',
}

export type Stance = 'trend' | 'fluke' | 'seasonal' | 'no_change' | 'neutral'

/** Which way a check's evidence points, from its ``stance`` (``supports_trend`` ...). */
export function stanceKind(stance: string): Stance {
  const s = stance.replace(/^supports_/, '')
  if (s === 'trend' || s === 'fluke' || s === 'seasonal' || s === 'no_change') return s
  return 'neutral'
}

export const STANCE_CLASSES: Record<Stance, string> = {
  trend: 'bg-emerald-500',
  fluke: 'bg-orange-500',
  seasonal: 'bg-violet-500',
  no_change: 'bg-sky-500',
  neutral: 'bg-slate-300 dark:bg-slate-600',
}

/** Confidence level as a 0–3 meter value. */
export function confidenceLevel(confidence: string | null): number {
  return { low: 1, medium: 2, high: 3 }[confidence ?? ''] ?? 0
}

const SNAPSHOT_RE = /^Cached snapshot fetched on (\d{4}-\d{2}-\d{2}) \(\d+ day\(s\) ago\)/

/**
 * Precomputed samples were analysed at build time, so the snapshot-age caveat
 * ("... (N day(s) ago)") is recomputed for today; everything else is untouched.
 */
export function refreshSnapshotCaveat(caveat: string, now: Date = new Date()): string {
  const m = SNAPSHOT_RE.exec(caveat)
  if (!m || !m[1]) return caveat
  const fetched = Date.parse(`${m[1]}T00:00:00Z`)
  const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate())
  const days = Math.max(0, Math.round((today - fetched) / 86_400_000))
  return caveat.replace(/\(\d+ day\(s\) ago\)/, `(${days} day(s) ago)`)
}

/** File-name slug used for precomputed results (signalcheck.adapters.base.slugify). */
export function slugify(text: string): string {
  return text
    .trim()
    .split(/\s+/)
    .join(' ')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

/** Compact number for axes and tooltips (1.2k, 3.4M; up to 4 significant digits). */
export function formatNumber(v: number): string {
  const abs = Math.abs(v)
  if (abs >= 1e9) return `${+(v / 1e9).toPrecision(3)}B`
  if (abs >= 1e6) return `${+(v / 1e6).toPrecision(3)}M`
  if (abs >= 1e4) return `${+(v / 1e3).toPrecision(3)}k`
  if (abs >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 0 })
  return `${+v.toPrecision(4)}`
}

/** "2 Oct 2026" from an ISO day or timestamp (UTC). */
export function formatDay(value: string | number): string {
  const d = typeof value === 'number' ? new Date(value) : new Date(`${value.slice(0, 10)}T00:00:00Z`)
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })
}

/** Cards in display order: analysed first, then failures, then disabled sources. */
export function orderCards(cards: Card[]): Card[] {
  const rank = (c: Card) => (c.disabled ? 2 : c.badge ? 0 : 1)
  return cards.map((c, i) => [c, i] as const).sort((a, b) => rank(a[0]) - rank(b[0]) || a[1] - b[1]).map(([c]) => c)
}

/** A short status line for a result header. */
export function resultMeta(result: Result): string {
  const analysed = result.cards.filter((c) => c.badge).length
  const parts = [`${analysed} source${analysed === 1 ? '' : 's'} analysed`]
  if (result.sample) parts.push(`sample snapshot · precomputed ${formatDay(result.generated_at)}`)
  else parts.push(`fetched ${new Date(result.generated_at).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}`)
  return parts.join(' · ')
}

/** Split the "<Source> disabled (reason)" / "Couldn't fetch <Source>: reason" message. */
export function cardMessage(card: Card): { title: string; detail: string | null } {
  const msg = card.message ?? ''
  const disabled = /^(.*?) disabled \((.*)\)$/.exec(msg)
  if (disabled) return { title: 'Not available on the static demo', detail: disabled[2] ?? null }
  return { title: msg, detail: null }
}
