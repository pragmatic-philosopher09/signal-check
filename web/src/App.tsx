import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { CardSkeleton, SourceCard, SummaryCard, VerdictBadge } from './components/Cards'
import type { DocsTab } from './components/DocsDrawer'
import { DataPill, EnginePill, Header } from './components/Header'
import { Icon } from './components/Icon'
import { CsvPanel, ModeTabs, SearchPanel, type CsvRequest, type Mode } from './components/Inputs'
import { engine } from './engine/client'
import { loadSample, loadSite, parseReply } from './lib/data'
import { dataRefreshedText } from './lib/freshness'
import { REPO_URL, useEngineStatus } from './lib/hooks'
import { LABEL_COLORS, orderCards, resultMeta } from './lib/viewModel'
import type { LabelKey, Result, Site } from './types'

// Methodology/About (and the Markdown renderer) load on first open.
const DocsDrawer = lazy(() => import('./components/DocsDrawer').then((m) => ({ default: m.DocsDrawer })))

type Origin = { kind: 'sample'; slug: string; query: string } | { kind: 'topic'; query: string; days: number | null } | { kind: 'csv' }

type View =
  | { status: 'empty' }
  | { status: 'loading'; label: string; live: boolean }
  | { status: 'done'; result: Result; origin: Origin }
  | { status: 'error'; message: string }

const VERDICTS: { key: LabelKey; text: string; hint: string }[] = [
  { key: 'TREND_up', text: 'Trend', hint: 'a sustained rise or fall' },
  { key: 'FLUKE', text: 'Fluke', hint: 'one or two spikes, then back' },
  { key: 'SEASONAL', text: 'Seasonal', hint: 'the same time every year' },
  { key: 'NO_CHANGE', text: 'No change', hint: 'flat within normal noise' },
  { key: 'INCONCLUSIVE', text: 'Inconclusive', hint: 'too little data or conflicting checks' },
]

function readUrl(): { sample: string | null; q: string | null; days: number | null } {
  const p = new URLSearchParams(window.location.search)
  const days = Number(p.get('days'))
  return { sample: p.get('sample'), q: p.get('q'), days: Number.isFinite(days) && days > 0 ? days : null }
}

function writeUrl(params: Record<string, string | null>): void {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v) p.set(k, v)
  const qs = p.toString()
  window.history.replaceState(null, '', `${window.location.pathname}${qs ? `?${qs}` : ''}`)
}

function initialView({ sample, q }: ReturnType<typeof readUrl>): View {
  if (sample) return { status: 'loading', label: 'Loading sample', live: false }
  if (q) return { status: 'loading', label: `Analysing “${q}” live`, live: true }
  return { status: 'empty' }
}

function scheduleWarmUp(): () => void {
  const conn = (navigator as Navigator & { connection?: { saveData?: boolean } }).connection
  if (conn?.saveData) return () => undefined
  const timer = window.setTimeout(() => {
    if ('requestIdleCallback' in window) window.requestIdleCallback(() => engine.warmUp(), { timeout: 4000 })
    else engine.warmUp()
  }, 2500)
  return () => window.clearTimeout(timer)
}

export default function App() {
  const [initial] = useState(readUrl)
  const [site, setSite] = useState<Site | null>(null)
  const [siteError, setSiteError] = useState<string | null>(null)
  const [mode, setMode] = useState<Mode>('topic')
  const [view, setView] = useState<View>(() => initialView(initial))
  const [csvColumns, setCsvColumns] = useState<string[] | null>(null)
  const [docs, setDocs] = useState<{ open: boolean; tab: DocsTab }>({ open: false, tab: 'methodology' })
  const status = useEngineStatus()
  const ticket = useRef(0)
  const resultsRef = useRef<HTMLDivElement>(null)

  const busy = view.status === 'loading'

  const showResult = useCallback((result: Result, origin: Origin) => {
    setView({ status: 'done', result, origin })
  }, [])

  const fetchSample = useCallback(
    async (slug: string, my: number) => {
      try {
        const result = await loadSample(slug)
        if (my === ticket.current) showResult(result, { kind: 'sample', slug, query: result.query })
      } catch (err) {
        if (my === ticket.current) setView({ status: 'error', message: (err as Error).message })
      }
    },
    [showResult],
  )

  const fetchTopic = useCallback(
    async (query: string, days: number | null, article: string | undefined, my: number) => {
      try {
        const reply = parseReply(await engine.call({ type: 'topic', query, ...(days ? { days } : {}), ...(article ? { article } : {}) }))
        if (my !== ticket.current) return
        if (reply.kind === 'columns') throw new Error('unexpected reply')
        showResult(reply, { kind: 'topic', query, days })
      } catch (err) {
        if (my === ticket.current) setView({ status: 'error', message: (err as Error).message })
      }
    },
    [showResult],
  )

  const openSample = (slug: string) => {
    setMode('topic')
    setView({ status: 'loading', label: 'Loading sample', live: false })
    writeUrl({ sample: slug })
    void fetchSample(slug, ++ticket.current)
  }

  const runTopic = (query: string, days: number | null, article?: string) => {
    setView({ status: 'loading', label: `Analysing “${query}” live`, live: true })
    writeUrl({ q: query, days: days ? String(days) : null })
    void fetchTopic(query, days, article, ++ticket.current)
  }

  const rerunSampleArticle = useCallback(
    async (origin: Extract<Origin, { kind: 'sample' }>, article: string) => {
      const my = ++ticket.current
      setView({ status: 'loading', label: `Re-analysing with “${article}”`, live: true })
      try {
        const reply = parseReply(await engine.call({ type: 'sample', query: origin.query, article }))
        if (my !== ticket.current) return
        if (reply.kind === 'columns') throw new Error('unexpected reply')
        showResult(reply, origin)
      } catch (err) {
        if (my === ticket.current) setView({ status: 'error', message: (err as Error).message })
      }
    },
    [showResult],
  )

  const runCsv = useCallback(
    async (req: CsvRequest) => {
      const my = ++ticket.current
      setView({ status: 'loading', label: `Analysing ${req.name}`, live: true })
      writeUrl({})
      try {
        const reply = parseReply(
          await engine.call({ type: 'csv', name: req.name, data: req.data, incomplete: req.incomplete, ...(req.column ? { column: req.column } : {}) }, [req.data]),
        )
        if (my !== ticket.current) return
        if (reply.kind === 'columns') {
          setCsvColumns(reply.columns)
          setView({ status: 'empty' })
          return
        }
        if (!req.column) setCsvColumns(null)
        showResult(reply, { kind: 'csv' })
      } catch (err) {
        if (my === ticket.current) setView({ status: 'error', message: (err as Error).message })
      }
    },
    [showResult],
  )

  useEffect(() => {
    loadSite().then(setSite, (err: Error) => setSiteError(err.message))
    const { sample, q, days } = initial
    if (sample) void fetchSample(sample, ++ticket.current)
    else if (q) void fetchTopic(q, days, undefined, ++ticket.current)
    return scheduleWarmUp()
  }, [initial, fetchSample, fetchTopic])

  useEffect(() => {
    if (view.status === 'loading' || view.status === 'done') {
      resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [view.status])

  const handleArticle = (origin: Origin, article: string) => {
    if (origin.kind === 'sample') void rerunSampleArticle(origin, article)
    else if (origin.kind === 'topic') runTopic(origin.query, origin.days, article)
  }

  const closeDocs = useCallback(() => setDocs((d) => ({ ...d, open: false })), [])
  const activeSample = view.status === 'done' && view.origin.kind === 'sample' ? view.origin.slug : null
  const chartHeight = site?.ui.chart_height_px ?? 320
  const headroom = site?.ui.threshold_headroom ?? null

  return (
    <div className="min-h-dvh">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-lg focus:bg-white focus:px-3 focus:py-2 focus:shadow">
        Skip to content
      </a>
      <Header onDocs={(tab) => setDocs({ open: true, tab })} />

      <main id="main" className="mx-auto max-w-5xl px-4 pb-20 sm:px-6">
        <section className="relative pb-6 pt-10 sm:pt-16">
          <div className="pointer-events-none absolute inset-x-0 -top-10 -z-10 mx-auto h-72 max-w-3xl rounded-full bg-gradient-to-r from-indigo-300/30 via-sky-200/30 to-emerald-200/30 blur-3xl dark:from-indigo-600/20 dark:via-sky-700/10 dark:to-emerald-700/10" aria-hidden />
          <p className="text-sm font-semibold text-indigo-600 dark:text-indigo-300">Signal Check</p>
          <h1 className="mt-2 max-w-3xl text-balance text-3xl font-bold tracking-tight text-slate-900 sm:text-5xl dark:text-white">
            Is this change real and lasting, or is it noise?
          </h1>
          <p className="mt-4 max-w-2xl text-pretty text-base leading-relaxed text-slate-600 sm:text-lg dark:text-slate-300">
            Pick a topic and Signal Check runs transparent statistical checks on its public attention — Wikipedia pageviews and
            Hacker News activity — then says whether the change is a trend, a fluke, seasonal or nothing at all. Statistics decide;
            the text only explains.
          </p>
          <ul className="mt-5 flex flex-wrap gap-2" aria-label="Possible verdicts">
            {VERDICTS.map((v) => (
              <li key={v.key} title={v.hint}>
                <VerdictBadge badge={{ key: v.key, text: v.text, color: LABEL_COLORS[v.key] }} size="sm" />
              </li>
            ))}
          </ul>
        </section>

        <section className="card p-4 sm:p-6" aria-label="Analyse">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <ModeTabs mode={mode} onMode={setMode} />
            <div className="flex flex-wrap items-center gap-2">
              <DataPill data={site?.data} />
              <EnginePill status={status} />
            </div>
          </div>
          {mode === 'topic' ? (
            <SearchPanel
              site={site}
              initialQuery={initial.q ?? ''}
              initialDays={initial.days}
              busy={busy}
              activeSample={activeSample}
              onSearch={(q, d) => runTopic(q, d)}
              onSample={(slug) => openSample(slug)}
              onIntent={() => engine.warmUp()}
            />
          ) : (
            <CsvPanel busy={busy} columns={csvColumns} onSubmit={(r) => void runCsv(r)} onIntent={() => engine.warmUp()} />
          )}
          {siteError && <p className="mt-3 text-sm text-rose-600">Couldn't load the site data: {siteError}</p>}
        </section>

        <div ref={resultsRef} className="scroll-mt-20 pt-8" aria-live="polite" aria-busy={busy}>
          {view.status === 'loading' && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-3 text-sm text-slate-600 dark:text-slate-300">
                <span className="inline-block size-4 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" aria-hidden />
                <span className="font-medium">{view.label}</span>
                {view.live && status.phase !== 'ready' && status.phase !== 'idle' && (
                  <span className="text-slate-500 dark:text-slate-400">
                    · {status.detail} ({status.step}/{status.steps})
                  </span>
                )}
                {view.live && status.phase === 'ready' && status.requests > 0 && (
                  <span className="text-slate-500 dark:text-slate-400">· {status.requests} API request{status.requests === 1 ? '' : 's'}</span>
                )}
              </div>
              <CardSkeleton />
              <CardSkeleton />
            </div>
          )}

          {view.status === 'error' && (
            <div className="card flex items-start gap-3 border-rose-200 p-5 text-sm text-rose-800 dark:border-rose-500/30 dark:text-rose-200" role="alert">
              <Icon name="alert" className="mt-0.5 size-5 shrink-0" />
              <div>
                <p className="font-semibold">Something went wrong</p>
                <p className="mt-1 break-words">{view.message}</p>
              </div>
            </div>
          )}

          {view.status === 'done' && (
            <ResultView
              key={`${view.result.query}:${view.result.generated_at}`}
              result={view.result}
              origin={view.origin}
              busy={busy}
              chartHeight={chartHeight}
              headroom={headroom}
              onArticle={(article) => handleArticle(view.origin, article)}
              onLive={view.origin.kind === 'sample' ? () => runTopic(view.result.query, null) : undefined}
            />
          )}

          {view.status === 'empty' && (
            <div className="grid gap-4 sm:grid-cols-3">
              {[
                { icon: 'search' as const, title: 'Fetch', text: 'Daily attention from public APIs, called straight from your browser and cached for 6 hours.' },
                { icon: 'list' as const, title: 'Check', text: 'Trend, level-shift, spike, seasonality and outlier tests with thresholds from config.yaml.' },
                { icon: 'check' as const, title: 'Decide', text: 'A fixed decision table (R1–R6) picks the verdict and lists what would change its mind.' },
              ].map((s) => (
                <div key={s.title} className="card p-5">
                  <span className="inline-flex size-9 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600 dark:bg-indigo-400/10 dark:text-indigo-300">
                    <Icon name={s.icon} className="size-5" />
                  </span>
                  <h2 className="mt-3 font-semibold text-slate-900 dark:text-white">{s.title}</h2>
                  <p className="mt-1 text-sm leading-relaxed text-slate-600 dark:text-slate-400">{s.text}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>

      <footer className="border-t border-slate-200 py-8 text-center text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
        <p>
          Signal Check{site ? ` v${site.version}` : ''} · statistics decide, the text only explains ·{' '}
          <a className="underline underline-offset-2 hover:text-slate-800 dark:hover:text-slate-200" href={REPO_URL} target="_blank" rel="noreferrer">
            GitHub
          </a>
        </p>
        <p className="mt-1">Data: Wikimedia pageviews (CC0), Hacker News via Algolia. Not affiliated with either.</p>
        {site?.data && (
          <p className="mt-1">
            {dataRefreshedText(site.data.refreshed_at)} by a daily GitHub Actions job; sample verdicts are recomputed on every refresh.
          </p>
        )}
      </footer>

      {docs.open && (
        <Suspense fallback={null}>
          <DocsDrawer open={docs.open} tab={docs.tab} onTab={(tab) => setDocs({ open: true, tab })} onClose={closeDocs} site={site} />
        </Suspense>
      )}
    </div>
  )
}

function ResultView({
  result,
  origin,
  busy,
  chartHeight,
  headroom,
  onArticle,
  onLive,
}: {
  result: Result
  origin: Origin
  busy: boolean
  chartHeight: number
  headroom: number | null
  onArticle: (article: string) => void
  onLive?: () => void
}) {
  const cards = orderCards(result.cards)
  const analysed = cards.filter((c) => c.badge)
  const others = cards.filter((c) => !c.badge)
  return (
    <div className="rise space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-2xl font-bold tracking-tight text-slate-900 dark:text-white">{result.query}</h2>
          <p className="mt-0.5 text-sm text-slate-500 dark:text-slate-400">{resultMeta(result)}</p>
        </div>
        {onLive && (
          <button type="button" className="btn-ghost ring-1 ring-inset ring-slate-200 dark:ring-slate-700" onClick={onLive} disabled={busy}>
            <Icon name="repeat" />
            Fetch live data
          </button>
        )}
      </div>
      {result.summary && <SummaryCard summary={result.summary} query={result.query} />}
      {analysed.map((card) => (
        <SourceCard
          key={card.source}
          card={card}
          headroom={headroom}
          chartHeight={chartHeight}
          busy={busy}
          onArticle={card.source === 'wikipedia' && origin.kind !== 'csv' ? onArticle : undefined}
        />
      ))}
      {others.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {others.map((card) => (
            <SourceCard key={card.source} card={card} headroom={headroom} chartHeight={chartHeight} />
          ))}
        </div>
      )}
    </div>
  )
}
