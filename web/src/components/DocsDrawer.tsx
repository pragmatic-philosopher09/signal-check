import { useEffect, useRef } from 'react'
import { REPO_URL } from '../lib/hooks'
import type { Site } from '../types'
import { Icon } from './Icon'
import { Markdown } from './Markdown'

export type DocsTab = 'methodology' | 'about'

const ABOUT = `
**Signal Check** answers one question about a topic's public attention: *is this change real and lasting, or is it noise?*
Each source gets one of five verdicts — **trend** (up or down), **fluke**, **seasonal**, **no change** — or **inconclusive**.

### Who decides
Deterministic statistics decide every verdict: a fixed set of checks feeds a transparent decision table (R1–R6, see Methodology),
with every threshold read from the shipped \`config.yaml\`. There is no machine-learning model and no forecasting.
The written summary only narrates the validated numbers; on this static demo it is always the deterministic template.

### How this demo runs
This page is static (GitHub Pages). The real \`signalcheck\` Python package runs **inside your browser** via
[Pyodide](https://pyodide.org) (CPython compiled to WebAssembly) in a Web Worker — the same code as the server app.
Sample topics are precomputed at build time by that engine, so they appear instantly. Live queries call the
public APIs directly from your browser and are cached locally for 6 hours (IndexedDB). Only aggregate counts are
stored; never author names or post text.

| Source | Static demo | Server deployment |
|---|---|---|
| Wikipedia pageviews | ✅ live (Wikimedia REST API) | ✅ live |
| Hacker News | ✅ live (Algolia API) | ✅ live |
| Google Trends | CSV upload only | CSV upload (+ snapshots) |
| Reddit | ❌ needs server-side credentials | ✅ with API credentials |
| X | ❌ needs server-side credentials | ✅ with API credentials |
| AI narration | template only (no keys shipped) | LLM, validated against the numbers |

### Limitations
- Public attention is not demand, sentiment or quality; a verdict says nothing about *why* attention moved.
- Short histories (Hacker News: 90 days) can't show seasonality; many checks are skipped and the verdict may be inconclusive.
- Wikipedia counts one article (pick another from the card if the guess is wrong); \`agent=user\` excludes known bots, not all automation.
- Hacker News counts stories and comments containing the query words; a common word can match unrelated posts.
- Browsers can't set \`User-Agent\`; the demo sends \`Api-User-Agent\` to Wikipedia's Action API. The Wikimedia REST pageviews API
  rejects that header from browsers (CORS preflight), so pageview calls are anonymous simple requests.
- The first live query downloads the Python runtime and scientific libraries (≈ 30–40 MB, cached by your browser afterwards).
`

export function DocsDrawer({ open, tab, onTab, onClose, site }: { open: boolean; tab: DocsTab; onTab: (t: DocsTab) => void; onClose: () => void; site: Site | null }) {
  const panelRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const prev = document.activeElement as HTMLElement | null
    panelRef.current?.focus()
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
      prev?.focus()
    }
  }, [open, onClose])
  if (!open) return null
  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-labelledby="docs-title">
      <div className="fade-in absolute inset-0 bg-slate-950/40 backdrop-blur-[2px]" onClick={onClose} aria-hidden />
      <div
        ref={panelRef}
        tabIndex={-1}
        className="drawer-in absolute inset-y-0 right-0 flex w-full max-w-2xl flex-col bg-white shadow-2xl outline-none dark:bg-slate-900"
      >
        <div className="flex items-center justify-between gap-2 border-b border-slate-200 px-5 py-3 dark:border-slate-800">
          <div role="tablist" aria-label="Documentation" className="inline-flex rounded-xl bg-slate-100 p-1 dark:bg-slate-800">
            {(['methodology', 'about'] as const).map((t) => (
              <button
                key={t}
                type="button"
                role="tab"
                aria-selected={tab === t}
                onClick={() => onTab(t)}
                className={`rounded-lg px-3 py-1.5 text-sm font-medium capitalize transition ${
                  tab === t ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-white' : 'text-slate-500 hover:text-slate-800 dark:text-slate-400'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
          <button type="button" className="btn-ghost" onClick={onClose} aria-label="Close">
            <Icon name="close" className="size-5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-5 py-6 sm:px-8">
          <h2 id="docs-title" className="text-2xl font-semibold tracking-tight text-slate-900 dark:text-white">
            {tab === 'methodology' ? 'Methodology' : 'About Signal Check'}
          </h2>
          {tab === 'methodology' ? (
            site ? (
              <>
                <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
                  Every threshold below is read from the shipped <code className="font-mono">config.yaml</code> (engine v{site.version}).
                </p>
                {site.methodology.map((s) => (
                  <section key={s.title} className="mt-7">
                    <h3 className="mb-2 text-lg font-semibold text-slate-900 dark:text-slate-100">{s.title}</h3>
                    <Markdown text={s.body} />
                  </section>
                ))}
              </>
            ) : (
              <p className="mt-4 text-sm text-slate-500">Loading…</p>
            )
          ) : (
            <>
              <Markdown text={ABOUT} className="mt-4" />
              <p className="mt-6 text-sm">
                <a className="font-medium text-indigo-600 hover:underline dark:text-indigo-300" href={REPO_URL} target="_blank" rel="noreferrer">
                  Source code, server deployment and evaluation on GitHub →
                </a>
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
