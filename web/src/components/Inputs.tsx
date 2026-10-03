import { useId, useRef, useState, type DragEvent, type FormEvent } from 'react'
import type { Site } from '../types'
import { Icon } from './Icon'

export type Mode = 'topic' | 'csv'

export function ModeTabs({ mode, onMode }: { mode: Mode; onMode: (m: Mode) => void }) {
  const tabs: { id: Mode; label: string; icon: 'search' | 'upload' }[] = [
    { id: 'topic', label: 'Search a topic', icon: 'search' },
    { id: 'csv', label: 'Upload a CSV', icon: 'upload' },
  ]
  return (
    <div role="tablist" aria-label="Input" className="inline-flex rounded-xl bg-slate-100 p-1 dark:bg-slate-800/80">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          type="button"
          aria-selected={mode === t.id}
          onClick={() => onMode(t.id)}
          className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition ${
            mode === t.id
              ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-white'
              : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
          }`}
        >
          <Icon name={t.icon} className="size-3.5" />
          {t.label}
        </button>
      ))}
    </div>
  )
}

export function SearchPanel({
  site,
  initialQuery,
  initialDays,
  busy,
  activeSample,
  onSearch,
  onSample,
  onIntent,
}: {
  site: Site | null
  initialQuery: string
  initialDays: number | null
  busy: boolean
  activeSample: string | null
  onSearch: (query: string, days: number | null) => void
  onSample: (slug: string) => void
  onIntent: () => void
}) {
  const [query, setQuery] = useState(initialQuery)
  const [days, setDays] = useState<number | null>(initialDays)
  const inputId = useId()
  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (query.trim()) onSearch(query.trim(), days)
  }
  const history = site?.ui.history_days
  return (
    <div>
      <form onSubmit={submit} className="flex flex-col gap-2 sm:flex-row" role="search">
        <label htmlFor={inputId} className="sr-only">
          Topic
        </label>
        <div className="relative flex-1">
          <Icon name="search" className="pointer-events-none absolute left-3.5 top-1/2 size-4 -translate-y-1/2 text-slate-400" />
          <input
            id={inputId}
            type="search"
            className="field pl-10"
            placeholder="Try “rust programming” or “perplexity ai”"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={onIntent}
            autoComplete="off"
            enterKeyHint="search"
            maxLength={200}
          />
        </div>
        <div className="flex gap-2">
          <label className="sr-only" htmlFor={`${inputId}-days`}>
            Timeframe
          </label>
          <select
            id={`${inputId}-days`}
            className="field w-auto flex-1 pr-8 sm:flex-none"
            value={days ?? ''}
            onChange={(e) => setDays(e.target.value ? Number(e.target.value) : null)}
            title={history ? `Source defaults: Wikipedia ${history.wikipedia} days, Hacker News ${history.hackernews} days` : undefined}
          >
            <option value="">Source defaults</option>
            {(site?.ui.timeframe_days ?? []).map((d) => (
              <option key={d} value={d}>
                Last {d} days
              </option>
            ))}
          </select>
          <button type="submit" className="btn-primary shrink-0" disabled={busy || !query.trim()}>
            Analyse
            <Icon name="arrowRight" />
          </button>
        </div>
      </form>
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-slate-500 dark:text-slate-400">Instant samples:</span>
        {site
          ? site.samples.map((s) => (
              <button
                key={s.slug}
                type="button"
                onClick={() => onSample(s.slug)}
                aria-pressed={activeSample === s.slug}
                className={`rounded-full px-3 py-1 text-sm font-medium ring-1 ring-inset transition ${
                  activeSample === s.slug
                    ? 'bg-indigo-600 text-white ring-indigo-600 dark:bg-indigo-500 dark:ring-indigo-500'
                    : 'bg-white text-slate-700 ring-slate-200 hover:bg-indigo-50 hover:text-indigo-700 hover:ring-indigo-200 dark:bg-slate-900 dark:text-slate-200 dark:ring-slate-700 dark:hover:bg-indigo-400/10 dark:hover:text-indigo-200'
                }`}
              >
                {s.query}
              </button>
            ))
          : [1, 2, 3, 4].map((i) => <span key={i} className="h-7 w-24 animate-pulse rounded-full bg-slate-200 dark:bg-slate-800" />)}
      </div>
    </div>
  )
}

export interface CsvRequest {
  name: string
  data: ArrayBuffer
  column: string | undefined
  incomplete: boolean
}

export function CsvPanel({
  busy,
  columns,
  onSubmit,
  onIntent,
}: {
  busy: boolean
  columns: string[] | null
  onSubmit: (req: CsvRequest) => void
  onIntent: () => void
}) {
  const [file, setFile] = useState<{ name: string; data: ArrayBuffer } | null>(null)
  const [label, setLabel] = useState('')
  const [incomplete, setIncomplete] = useState(false)
  const [column, setColumn] = useState('')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const id = useId()

  const run = (next: { file?: typeof file; incomplete?: boolean; column?: string; label?: string }) => {
    const f = next.file !== undefined ? next.file : file
    if (!f) return
    const col = next.column !== undefined ? next.column : column
    const name = (next.label !== undefined ? next.label : label).trim() || f.name
    onSubmit({ name, data: f.data.slice(0), column: col || undefined, incomplete: next.incomplete ?? incomplete })
  }

  const accept = async (picked: File | undefined) => {
    if (!picked) return
    const loaded = { name: picked.name, data: await picked.arrayBuffer() }
    setFile(loaded)
    setColumn('')
    run({ file: loaded, column: '' })
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    void accept(e.dataTransfer.files[0])
  }

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed px-4 py-7 text-center transition ${
          dragging ? 'border-indigo-400 bg-indigo-50/60 dark:bg-indigo-400/10' : 'border-slate-200 dark:border-slate-700'
        }`}
      >
        <Icon name={file ? 'file' : 'upload'} className="size-6 text-indigo-500" />
        {file ? (
          <p className="text-sm font-medium text-slate-800 dark:text-slate-100">{file.name}</p>
        ) : (
          <p className="text-sm text-slate-600 dark:text-slate-300">Drop a CSV here — date + value, or a Google Trends export</p>
        )}
        <button
          type="button"
          className="btn-ghost text-indigo-600 dark:text-indigo-300"
          onClick={() => {
            onIntent()
            inputRef.current?.click()
          }}
        >
          {file ? 'Choose another file' : 'Choose a file'}
        </button>
        <input
          ref={inputRef}
          id={`${id}-file`}
          type="file"
          accept=".csv,text/csv"
          className="sr-only"
          aria-label="CSV file"
          onChange={(e) => {
            void accept(e.target.files?.[0])
            e.target.value = ''
          }}
        />
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <label htmlFor={`${id}-name`} className="mb-1 block text-xs font-medium text-slate-600 dark:text-slate-300">
            What does this series measure? (optional)
          </label>
          <input
            id={`${id}-name`}
            className="field"
            value={label}
            placeholder={file?.name ?? 'e.g. weekly signups'}
            onChange={(e) => setLabel(e.target.value)}
            onBlur={() => run({})}
          />
        </div>
        <div className="flex items-end">
          <label className="flex cursor-pointer items-center gap-3 py-2.5" title="Drop the last row if its period had not ended when the data was exported.">
            <input
              type="checkbox"
              role="switch"
              className="peer sr-only"
              checked={incomplete}
              onChange={(e) => {
                setIncomplete(e.target.checked)
                run({ incomplete: e.target.checked })
              }}
            />
            <span className="relative h-6 w-11 shrink-0 rounded-full bg-slate-300 transition peer-checked:bg-indigo-600 peer-focus-visible:ring-4 peer-focus-visible:ring-indigo-500/30 after:absolute after:left-0.5 after:top-0.5 after:size-5 after:rounded-full after:bg-white after:shadow after:transition peer-checked:after:translate-x-5 dark:bg-slate-600 dark:peer-checked:bg-indigo-500" />
            <span className="text-sm text-slate-700 dark:text-slate-200">Last period is incomplete</span>
          </label>
        </div>
      </div>
      {columns && (
        <div className="rise rounded-xl bg-amber-50 p-3 ring-1 ring-inset ring-amber-600/20 dark:bg-amber-400/10 dark:ring-amber-400/25">
          <label htmlFor={`${id}-col`} className="mb-1.5 block text-sm font-medium text-amber-900 dark:text-amber-100">
            This file has several value columns. Which one should be analysed?
          </label>
          <select
            id={`${id}-col`}
            className="field"
            value={column}
            disabled={busy}
            onChange={(e) => {
              setColumn(e.target.value)
              if (e.target.value) run({ column: e.target.value })
            }}
          >
            <option value="">Choose a column</option>
            {columns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
      )}
      <p className="text-xs text-slate-500 dark:text-slate-400">
        The file is analysed in your browser by the same Python engine; nothing is uploaded to a server.
      </p>
    </div>
  )
}
