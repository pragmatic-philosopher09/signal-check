// Browser replacement for the on-disk HTTP cache: IndexedDB rows of
// [expiresAt (epoch seconds) | null, valueJson], keyed like signalcheck.cache.
// Python keeps the working copy in memory (MemoryStore) and calls ``persist``
// on every write; this module loads unexpired rows at boot and prunes the rest.

import { createStore, delMany, entries, set, type UseStore } from 'idb-keyval'

export type CacheRow = [key: string, expiresAt: number | null, valueJson: string]

let store: UseStore | null = null

function db(): UseStore {
  store ??= createStore('signalcheck', 'http-cache')
  return store
}

/** Rows that have not expired at ``nowSeconds``; expired keys are returned separately. */
export function partitionRows(
  rows: [IDBValidKey, unknown][],
  nowSeconds: number,
): { live: CacheRow[]; expired: IDBValidKey[] } {
  const live: CacheRow[] = []
  const expired: IDBValidKey[] = []
  for (const [key, value] of rows) {
    if (typeof key !== 'string' || !Array.isArray(value) || value.length !== 2) {
      expired.push(key)
      continue
    }
    const [expiresAt, json] = value as [unknown, unknown]
    const exp = typeof expiresAt === 'number' ? expiresAt : null
    if (typeof json !== 'string' || (exp !== null && exp <= nowSeconds)) expired.push(key)
    else live.push([key, exp, json])
  }
  return { live, expired }
}

/** Load unexpired rows (and delete expired ones). Never throws: no cache is fine. */
export async function loadCache(nowSeconds = Date.now() / 1000): Promise<CacheRow[]> {
  try {
    const { live, expired } = partitionRows(await entries(db()), nowSeconds)
    if (expired.length) void delMany(expired, db()).catch(() => undefined)
    return live
  } catch {
    return []
  }
}

/** Persist one row (fire and forget; private browsing may refuse IndexedDB). */
export function persistRow(key: string, expiresAt: number | null | undefined, valueJson: string): void {
  void set(key, [expiresAt ?? null, valueJson], db()).catch(() => undefined)
}
