import type { EngineStatus } from '../engine/protocol'
import { dataRefreshedText, isStale, utcDateTime } from '../lib/freshness'
import type { DataFreshness } from '../types'
import { REPO_URL } from '../lib/hooks'
import { useTheme } from '../lib/theme'
import { Icon } from './Icon'

export function Header({ onDocs }: { onDocs: (tab: 'methodology' | 'about') => void }) {
  const { dark, toggle } = useTheme()
  return (
    <header className="sticky top-0 z-30 border-b border-slate-200/70 bg-white/75 backdrop-blur-md dark:border-slate-800/80 dark:bg-[#0b1020]/75">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between gap-2 px-4 sm:px-6">
        <a href={import.meta.env.BASE_URL} className="flex min-w-0 items-center gap-2 font-semibold tracking-tight text-slate-900 dark:text-white">
          <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" className="size-7" />
          <span className="truncate">Signal Check</span>
        </a>
        <nav className="flex items-center gap-0.5 sm:gap-1" aria-label="Main">
          <button type="button" className="btn-ghost" onClick={() => onDocs('methodology')}>
            <Icon name="book" />
            <span className="hidden sm:inline">Methodology</span>
            <span className="sr-only sm:hidden">Methodology</span>
          </button>
          <button type="button" className="btn-ghost" onClick={() => onDocs('about')}>
            <Icon name="info" />
            <span className="hidden sm:inline">About</span>
            <span className="sr-only sm:hidden">About</span>
          </button>
          <a className="btn-ghost" href={REPO_URL} target="_blank" rel="noreferrer">
            <Icon name="github" />
            <span className="hidden sm:inline">GitHub</span>
            <span className="sr-only sm:hidden">GitHub repository</span>
          </a>
          <button type="button" className="btn-ghost" onClick={toggle} aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}>
            <Icon name={dark ? 'sun' : 'moon'} />
          </button>
        </nav>
      </div>
    </header>
  )
}

/** Small pill describing the in-browser Python engine. */
export function EnginePill({ status }: { status: EngineStatus }) {
  const loading = status.phase === 'runtime' || status.phase === 'packages' || status.phase === 'engine'
  const tone =
    status.phase === 'ready'
      ? 'bg-emerald-50 text-emerald-700 ring-emerald-600/20 dark:bg-emerald-400/10 dark:text-emerald-300 dark:ring-emerald-400/25'
      : status.phase === 'error'
        ? 'bg-rose-50 text-rose-700 ring-rose-600/20 dark:bg-rose-400/10 dark:text-rose-300 dark:ring-rose-400/25'
        : 'bg-slate-100 text-slate-600 ring-slate-500/15 dark:bg-slate-800 dark:text-slate-300 dark:ring-slate-600/40'
  const text =
    status.phase === 'idle'
      ? 'Python engine: loads on first live query'
      : status.phase === 'ready'
        ? `Python engine ready${status.bootMs ? ` · ${(status.bootMs / 1000).toFixed(1)}s` : ''}`
        : status.phase === 'error'
          ? `Engine failed: ${status.detail}`
          : `${status.detail} (${status.step}/${status.steps})`
  return (
    <span role="status" aria-live="polite" className={`inline-flex max-w-full items-center gap-2 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${tone}`}>
      {loading ? (
        <span className="relative flex size-2" aria-hidden>
          <span className="absolute inline-flex size-full animate-ping rounded-full bg-indigo-400 opacity-75" />
          <span className="relative inline-flex size-2 rounded-full bg-indigo-500" />
        </span>
      ) : (
        <Icon name="cpu" className="size-3.5 shrink-0" />
      )}
      <span className="truncate">{text}</span>
    </span>
  )
}

const SOURCE_NAMES: Record<string, string> = { wikipedia: 'Wikipedia', hackernews: 'Hacker News', reddit: 'Reddit' }

/** "Data refreshed 5 hours ago (3 Oct 2026 UTC)" from the daily refresh manifest. */
export function DataPill({ data, now }: { data: DataFreshness | null | undefined; now?: Date }) {
  const text = dataRefreshedText(data?.refreshed_at, now)
  if (!data || !text) return null
  const warn = data.status !== 'ok' || isStale(data.refreshed_at, now)
  const detail = Object.entries(data.sources)
    .map(([s, status]) => `${SOURCE_NAMES[s] ?? s}: ${status}`)
    .join(' · ')
  return (
    <span
      title={`Daily refresh at ${utcDateTime(data.refreshed_at)} · ${detail}`}
      className="inline-flex max-w-full items-center gap-2 rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600 ring-1 ring-inset ring-slate-500/15 dark:bg-slate-800 dark:text-slate-300 dark:ring-slate-600/40"
    >
      <span className={`size-2 shrink-0 rounded-full ${warn ? 'bg-amber-500' : 'bg-emerald-500'}`} aria-hidden />
      <span className="truncate">{text}</span>
      {warn && <span className="sr-only"> (some sources were not refreshed)</span>}
    </span>
  )
}
