/// <reference lib="webworker" />
// Runs the real signalcheck Python package in Pyodide, off the main thread.
//
// Pyodide and its scientific packages come from the jsDelivr CDN; the
// signalcheck bundle (package + config.yaml + sample snapshots) and the
// pure-Python wheels come from this site (built by scripts/build_web.py).
// HTTP goes through synchronous XHR (allowed in workers) via
// signalcheck.web.worker.XhrSession; cached payloads persist to IndexedDB.

import type { PyodideInterface } from 'pyodide'
import type { PyProxy } from 'pyodide/ffi'
import type { PyManifest } from '../types'
import { loadCache, persistRow } from './cache'
import { PYODIDE_CDN } from './config'
import type { EngineMessage, EnginePhase, EngineRequest } from './protocol'

declare const self: DedicatedWorkerGlobalScope

const STEPS = 4
const post = (msg: EngineMessage, transfer: Transferable[] = []) => self.postMessage(msg, transfer)
const status = (phase: EnginePhase, step: number, detail: string, extra: object = {}) =>
  post({ type: 'status', phase, step, steps: STEPS, detail, ...extra })

const asset = (path: string) => new URL(`${import.meta.env.BASE_URL}${path}`, self.location.origin).href

let ready: Promise<{ py: PyodideInterface; mod: PyProxy }> | null = null

async function fetchOk(url: string): Promise<Response> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`couldn't load ${url} (HTTP ${res.status})`)
  return res
}

async function boot(): Promise<{ py: PyodideInterface; mod: PyProxy }> {
  const t0 = performance.now()
  status('runtime', 1, 'Downloading the Python runtime')
  const manifestP = fetchOk(asset('py/manifest.json')).then((r) => r.json() as Promise<PyManifest>)
  const cacheP = loadCache()
  const { loadPyodide } = (await import(/* @vite-ignore */ `${PYODIDE_CDN}pyodide.mjs`)) as typeof import('pyodide')
  const py = await loadPyodide({ indexURL: PYODIDE_CDN })
  const manifest = await manifestP

  status('packages', 2, 'Loading numpy, pandas, scipy, statsmodels')
  const wheels = manifest.wheels.map((w) => asset(`py/${w}`))
  await py.loadPackage([...manifest.pyodide_packages, ...wheels], { messageCallback: () => undefined })

  status('engine', 3, 'Starting the Signal Check engine')
  const bundle = await (await fetchOk(asset(`py/${manifest.bundle}`))).arrayBuffer()
  py.unpackArchive(bundle, 'zip', { extractDir: '/home/pyodide/app' })
  py.runPython("import sys; sys.path.insert(0, '/home/pyodide/app')")
  const mod = py.pyimport('signalcheck.web.worker') as PyProxy
  const rows = await cacheP
  const info = mod.boot.callKwargs(JSON.stringify(rows), persistRow, {
    on_request: (host: string) => post({ type: 'request', host }),
  }) as PyProxy
  const cached = Number(info.get('cached_entries'))
  info.destroy()
  const bootMs = Math.round(performance.now() - t0)
  status('ready', 4, 'Engine ready', { cachedEntries: cached, bootMs })
  return { py, mod }
}

function pythonError(err: unknown): string {
  const text = err instanceof Error ? err.message : String(err)
  const lines = text.trim().split('\n').filter(Boolean)
  return lines[lines.length - 1] ?? 'unknown error'
}

async function handle(req: EngineRequest): Promise<string | null> {
  // A failed boot (e.g. offline) is reported and retried on the next request.
  ready ??= boot().catch((err: unknown) => {
    ready = null
    status('error', 0, pythonError(err))
    throw err
  })
  const { py, mod } = await ready
  switch (req.type) {
    case 'init':
      return null
    case 'topic':
      return mod.run_topic_json(req.query, req.days, req.article) as string
    case 'sample':
      return mod.run_sample_json(req.query, req.article) as string
    case 'csv': {
      py.FS.writeFile('/tmp/upload.csv', new Uint8Array(req.data))
      const data = py.runPython("open('/tmp/upload.csv', 'rb').read()") as PyProxy
      try {
        return mod.run_csv_json(data, req.name, req.column, req.incomplete) as string
      } finally {
        data.destroy()
      }
    }
  }
}

// Requests run one at a time (Python is single-threaded and the XHRs are synchronous).
let queue: Promise<unknown> = Promise.resolve()

self.onmessage = (event: MessageEvent<EngineRequest>) => {
  const req = event.data
  queue = queue.then(async () => {
    try {
      const json = await handle(req)
      post({ type: 'result', id: req.id, ok: true, json: json ?? 'null' })
    } catch (err) {
      post({ type: 'result', id: req.id, ok: false, error: pythonError(err) })
    }
  })
}
