import { useSyncExternalStore } from 'react'
import { engine } from '../engine/client'
import type { EngineStatus } from '../engine/protocol'

export const REPO_URL = 'https://github.com/pragmatic-philosopher09/signal-check'

/** Live status of the in-browser Python engine. */
export function useEngineStatus(): EngineStatus {
  return useSyncExternalStore(engine.subscribe, engine.getSnapshot)
}
