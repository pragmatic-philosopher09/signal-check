import { fetchedText } from '../lib/freshness'
import { ACCENT_CLASSES, BADGE_CLASSES, cardMessage, confidenceLevel, refreshSnapshotCaveat, STANCE_CLASSES, stanceKind } from '../lib/viewModel'
import type { Badge, Card, Summary } from '../types'
import { Chart } from './Chart'
import { Icon } from './Icon'

const BADGE_ICON: Record<string, Parameters<typeof Icon>[0]['name']> = {
  TREND_up: 'trendUp',
  TREND_down: 'trendDown',
  FLUKE: 'bolt',
  SEASONAL: 'repeat',
  NO_CHANGE: 'flat',
  INCONCLUSIVE: 'help',
}

export function VerdictBadge({ badge, size = 'md' }: { badge: Badge; size?: 'sm' | 'md' | 'lg' }) {
  const sz = size === 'lg' ? 'px-3 py-1.5 text-base gap-2' : size === 'sm' ? 'px-2 py-0.5 text-xs gap-1' : 'px-2.5 py-1 text-sm gap-1.5'
  return (
    <span className={`inline-flex items-center rounded-full font-semibold ring-1 ring-inset ${sz} ${BADGE_CLASSES[badge.color]}`}>
      <Icon name={BADGE_ICON[badge.key] ?? 'help'} className={size === 'lg' ? 'size-5' : 'size-4'} />
      {badge.text}
    </span>
  )
}

export function ConfidenceMeter({ confidence }: { confidence: string | null }) {
  const level = confidenceLevel(confidence)
  return (
    <span className="inline-flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300" title="Confidence from the effect size, data volume and agreement between checks (see Methodology)">
      <span className="flex gap-0.5" aria-hidden>
        {[1, 2, 3].map((i) => (
          <span key={i} className={`h-3 w-1.5 rounded-sm ${i <= level ? 'bg-indigo-500 dark:bg-indigo-400' : 'bg-slate-200 dark:bg-slate-700'}`} />
        ))}
      </span>
      <span>
        <span className="sr-only">Confidence: </span>
        {confidence ?? 'n/a'} confidence
      </span>
    </span>
  )
}

function Section({ title, children, icon }: { title: string; children: React.ReactNode; icon?: Parameters<typeof Icon>[0]['name'] }) {
  return (
    <section className="mt-5">
      <h4 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        {icon && <Icon name={icon} className="size-3.5" />}
        {title}
      </h4>
      {children}
    </section>
  )
}

function WikiPicker({ card, onArticle, busy }: { card: Card; onArticle?: (article: string) => void; busy: boolean }) {
  if (!card.wiki) return null
  const { article, candidates, overridden } = card.wiki
  const options = [article, ...candidates.filter((c) => c !== article)]
  return (
    <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
      <span>Article:</span>
      {onArticle && options.length > 1 ? (
        <select
          aria-label="Wikipedia article"
          className="max-w-full truncate rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
          value={article}
          disabled={busy}
          onChange={(e) => onArticle(e.target.value)}
        >
          {options.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      ) : (
        <span className="font-medium text-slate-700 dark:text-slate-200">{article}</span>
      )}
      {overridden && <span className="rounded bg-slate-100 px-1.5 py-0.5 dark:bg-slate-800">chosen by you</span>}
    </div>
  )
}

export function SourceCard({
  card,
  headroom,
  chartHeight,
  onArticle,
  busy = false,
}: {
  card: Card
  headroom: number | null
  chartHeight: number
  onArticle?: (article: string) => void
  busy?: boolean
}) {
  if (!card.badge) {
    const { title, detail } = cardMessage(card)
    return (
      <article className="card flex flex-col gap-2 p-5" aria-label={`${card.title}: ${card.message ?? ''}`}>
        <header className="flex items-center justify-between gap-2">
          <h3 className="font-semibold text-slate-900 dark:text-slate-100">{card.title}</h3>
          <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${card.disabled ? 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400' : 'bg-rose-50 text-rose-700 dark:bg-rose-400/10 dark:text-rose-300'}`}>
            {card.disabled ? 'disabled' : 'failed'}
          </span>
        </header>
        <p className={`text-sm ${card.disabled ? 'text-slate-600 dark:text-slate-300' : 'text-rose-700 dark:text-rose-300'}`}>{title}</p>
        {detail && <p className="text-sm text-slate-500 dark:text-slate-400">{detail}</p>}
        {!card.disabled && card.caveats.length > 0 && (
          <ul className="list-disc space-y-1 pl-5 text-xs text-slate-500 dark:text-slate-400">
            {card.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        )}
      </article>
    )
  }

  const caveats = card.caveats.map((c) => refreshSnapshotCaveat(c))
  const fetched = fetchedText(card.fetched_at)
  return (
    <article className="card relative overflow-hidden" aria-label={`${card.title}: ${card.badge.text}`}>
      <span className={`absolute inset-y-0 left-0 w-1 ${ACCENT_CLASSES[card.badge.color]}`} aria-hidden />
      <div className="p-5 sm:p-6">
        <header className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 className="flex items-center gap-2 text-lg font-semibold text-slate-900 dark:text-slate-50">
              {card.title}
              {card.snapshot && <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-500 dark:bg-slate-800 dark:text-slate-400">snapshot</span>}
            </h3>
            {fetched && <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{fetched}</p>}
            <WikiPicker card={card} onArticle={onArticle} busy={busy} />
          </div>
          <div className="flex flex-col items-end gap-1.5">
            <VerdictBadge badge={card.badge} size="lg" />
            <ConfidenceMeter confidence={card.confidence} />
          </div>
        </header>

        <p className="mt-4 text-[15px] leading-relaxed text-slate-800 dark:text-slate-200">
          <abbr title={card.rule_text ?? undefined} className="mr-2 inline-flex cursor-help items-center rounded-md bg-indigo-50 px-1.5 py-0.5 font-mono text-xs font-semibold text-indigo-700 no-underline ring-1 ring-inset ring-indigo-600/20 dark:bg-indigo-400/10 dark:text-indigo-300 dark:ring-indigo-400/30">
            {card.rule}
          </abbr>
          {card.reason}.
        </p>
        {card.window_text && <p className="mt-1.5 text-xs text-slate-500 dark:text-slate-400">{card.window_text}</p>}

        {card.chart && (
          <div className="mt-4">
            <Chart chart={card.chart} headroom={headroom} height={chartHeight} title={card.title} />
          </div>
        )}

        <Section title="Evidence" icon="list">
          <ul className="space-y-2.5">
            {card.evidence.map((e) => (
              <li key={e.check} className="flex gap-3 text-sm">
                <span className={`mt-1.5 size-2 shrink-0 rounded-full ${STANCE_CLASSES[stanceKind(e.stance)]}`} aria-hidden />
                <span className="text-slate-700 dark:text-slate-300">
                  <span className="font-medium text-slate-900 dark:text-slate-100">{e.name}</span>{' '}
                  <span className="text-xs text-slate-500 dark:text-slate-400">({e.stance_text})</span> — {e.summary}
                </span>
              </li>
            ))}
          </ul>
          {card.skipped.length > 0 && (
            <details className="group mt-3 text-sm">
              <summary className="cursor-pointer text-xs text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200">
                {card.skipped.length} check{card.skipped.length === 1 ? '' : 's'} skipped
              </summary>
              <ul className="mt-2 space-y-1 pl-4 text-xs text-slate-500 dark:text-slate-400">
                {card.skipped.map((s) => (
                  <li key={s.name}>
                    <span className="font-medium">{s.name}</span>: {s.reason}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </Section>

        {card.change_my_mind.length > 0 && (
          <Section title="What would change my mind" icon="eye">
            <ul className="space-y-1.5 rounded-xl bg-indigo-50/70 p-3 text-sm text-indigo-950 ring-1 ring-inset ring-indigo-600/10 dark:bg-indigo-400/[0.07] dark:text-indigo-100 dark:ring-indigo-400/20">
              {card.change_my_mind.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          </Section>
        )}

        {caveats.length > 0 && (
          <Section title="Caveats" icon="alert">
            <ul className="list-disc space-y-1 pl-5 text-xs leading-relaxed text-slate-600 marker:text-amber-500 dark:text-slate-400">
              {caveats.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          </Section>
        )}

        {card.narration && (
          <Section title={card.narration.label} icon="quote">
            <p className="border-l-2 border-slate-200 pl-3 text-sm leading-relaxed text-slate-600 dark:border-slate-700 dark:text-slate-300">{card.narration.text}</p>
          </Section>
        )}
      </div>
    </article>
  )
}

export function SummaryCard({ summary, query }: { summary: Summary; query: string }) {
  return (
    <section className="card relative overflow-hidden p-5 sm:p-6" aria-label="Cross-source summary">
      <div className="pointer-events-none absolute -right-16 -top-20 size-56 rounded-full bg-indigo-400/10 blur-3xl dark:bg-indigo-500/15" aria-hidden />
      <p className="text-xs font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-300">Across sources · {query}</p>
      <p className="mt-2 text-xl font-semibold leading-snug text-slate-900 sm:text-2xl dark:text-slate-50">{summary.sentence}</p>
      {summary.chips.length > 0 && (
        <ul className="mt-4 flex flex-wrap gap-2">
          {summary.chips.map((c) => (
            <li key={c.source} className="flex items-center gap-2 rounded-full bg-slate-50 py-1 pl-3 pr-1 text-sm ring-1 ring-inset ring-slate-200 dark:bg-slate-800/60 dark:ring-slate-700">
              <span className="text-slate-600 dark:text-slate-300">{c.title}</span>
              <VerdictBadge badge={c.badge} size="sm" />
            </li>
          ))}
        </ul>
      )}
      {(summary.unavailable.length > 0 || summary.not_comparable.length > 0) && (
        <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
          {summary.unavailable.length > 0 && <>Unavailable: {summary.unavailable.join(', ')}. </>}
          {summary.not_comparable.length > 0 && <>Not comparable (different window): {summary.not_comparable.join(', ')}.</>}
        </p>
      )}
    </section>
  )
}

export function CardSkeleton() {
  return (
    <div className="card animate-pulse p-6" aria-hidden>
      <div className="flex justify-between">
        <div className="h-5 w-32 rounded bg-slate-200 dark:bg-slate-700" />
        <div className="h-8 w-28 rounded-full bg-slate-200 dark:bg-slate-700" />
      </div>
      <div className="mt-5 h-4 w-3/4 rounded bg-slate-200 dark:bg-slate-700" />
      <div className="mt-2 h-3 w-1/2 rounded bg-slate-100 dark:bg-slate-800" />
      <div className="mt-5 h-56 rounded-xl bg-slate-100 dark:bg-slate-800" />
      <div className="mt-5 space-y-2">
        <div className="h-3 w-full rounded bg-slate-100 dark:bg-slate-800" />
        <div className="h-3 w-5/6 rounded bg-slate-100 dark:bg-slate-800" />
        <div className="h-3 w-2/3 rounded bg-slate-100 dark:bg-slate-800" />
      </div>
    </div>
  )
}
