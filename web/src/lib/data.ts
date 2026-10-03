// Static assets produced by scripts/build_web.py (served next to index.html).

import type { ColumnChoice, Result, Site } from '../types'

const base = import.meta.env.BASE_URL

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${base}${path}`)
  if (!res.ok) throw new Error(`couldn't load ${path} (HTTP ${res.status})`)
  return (await res.json()) as T
}

export const loadSite = () => getJson<Site>('site.json')

/** A precomputed sample result (instant; no Python needed). */
export const loadSample = (slug: string) => getJson<Result>(`results/${slug}.json`)

/** Parse the engine's JSON reply (a topic/CSV result or a CSV column choice). */
export function parseReply(json: string): Result | ColumnChoice {
  const value = JSON.parse(json) as Result | ColumnChoice
  if (value.schema !== 1) throw new Error(`unsupported result schema ${String(value.schema)}`)
  return value
}
