// Main-thread handle on the Pyodide worker: lazy start, request/response, status.

import type { EngineMessage, EngineRequest, EngineStatus } from './protocol'

type Pending = { resolve: (json: string) => void; reject: (err: Error) => void }
type Payload = EngineRequest extends infer R ? (R extends EngineRequest ? Omit<R, 'id'> : never) : never

const INITIAL: EngineStatus = {
  phase: 'idle',
  detail: 'Not loaded — starts on your first live query',
  step: 0,
  steps: 4,
  cachedEntries: 0,
  requests: 0,
  busy: false,
  bootMs: null,
}

class Engine {
  private worker: Worker | null = null
  private nextId = 1
  private pending = new Map<number, Pending>()
  private listeners = new Set<() => void>()
  private state: EngineStatus = INITIAL

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  getSnapshot = (): EngineStatus => this.state

  private update(patch: Partial<EngineStatus>): void {
    this.state = { ...this.state, ...patch }
    for (const l of this.listeners) l()
  }

  private ensureWorker(): Worker {
    if (this.worker) return this.worker
    const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module', name: 'signalcheck-engine' })
    worker.onmessage = (event: MessageEvent<EngineMessage>) => this.onMessage(event.data)
    worker.onerror = (event) => {
      this.update({ phase: 'error', detail: event.message || 'the engine worker crashed' })
      for (const p of this.pending.values()) p.reject(new Error(event.message || 'engine worker crashed'))
      this.pending.clear()
    }
    this.worker = worker
    return worker
  }

  private onMessage(msg: EngineMessage): void {
    if (msg.type === 'status') {
      this.update({
        phase: msg.phase,
        detail: msg.detail,
        step: msg.step,
        steps: msg.steps,
        ...(msg.cachedEntries !== undefined ? { cachedEntries: msg.cachedEntries } : {}),
        ...(msg.bootMs !== undefined ? { bootMs: msg.bootMs } : {}),
      })
    } else if (msg.type === 'request') {
      this.update({ requests: this.state.requests + 1 })
    } else {
      const p = this.pending.get(msg.id)
      if (!p) return
      this.pending.delete(msg.id)
      if (this.pending.size === 0) this.update({ busy: false })
      if (msg.ok) p.resolve(msg.json)
      else p.reject(new Error(msg.error))
    }
  }

  /** Start loading Pyodide in the background (idempotent). */
  warmUp(): void {
    if (this.state.phase === 'idle') void this.call({ type: 'init' }).catch(() => undefined)
  }

  /** Send one request; resolves with the JSON text returned by Python. */
  call(payload: Payload, transfer: Transferable[] = []): Promise<string> {
    const worker = this.ensureWorker()
    const id = this.nextId++
    if (this.state.phase === 'idle') this.update({ phase: 'runtime', detail: 'Starting the engine', step: 0 })
    if (payload.type !== 'init') this.update({ busy: true, requests: 0 })
    return new Promise<string>((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      worker.postMessage({ ...payload, id } as EngineRequest, transfer)
    })
  }
}

export const engine = new Engine()
