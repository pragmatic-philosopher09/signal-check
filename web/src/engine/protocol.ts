// Messages between the page and the Pyodide worker.

export type EnginePhase = 'idle' | 'runtime' | 'packages' | 'engine' | 'ready' | 'error'

export interface EngineStatus {
  phase: EnginePhase
  detail: string
  step: number
  steps: number
  cachedEntries: number
  requests: number
  busy: boolean
  bootMs: number | null
}

export type EngineRequest =
  | { id: number; type: 'init' }
  | { id: number; type: 'topic'; query: string; days?: number; article?: string }
  | { id: number; type: 'sample'; query: string; article?: string }
  | { id: number; type: 'csv'; name: string; data: ArrayBuffer; column?: string; incomplete: boolean }

export type EngineMessage =
  | { type: 'status'; phase: EnginePhase; detail: string; step: number; steps: number; cachedEntries?: number; bootMs?: number }
  | { type: 'request'; host: string }
  | { type: 'result'; id: number; ok: true; json: string }
  | { type: 'result'; id: number; ok: false; error: string }
