// "Data refreshed 5 hours ago (3 Oct 2026 UTC)" from data/manifest.json and per-card fetch times.

const MINUTE = 60_000
const HOUR = 60 * MINUTE
const DAY = 24 * HOUR

function parse(iso: string | null | undefined): number | null {
  if (!iso) return null
  const t = Date.parse(iso)
  return Number.isFinite(t) ? t : null
}

/** "just now", "12 minutes ago", "5 hours ago", "yesterday", "3 days ago" (future → "just now"). */
export function relativeTime(iso: string, now: Date = new Date()): string | null {
  const t = parse(iso)
  if (t === null) return null
  const diff = Math.max(0, now.getTime() - t)
  if (diff < MINUTE) return 'just now'
  if (diff < HOUR) {
    const n = Math.floor(diff / MINUTE)
    return `${n} minute${n === 1 ? '' : 's'} ago`
  }
  if (diff < DAY) {
    const n = Math.floor(diff / HOUR)
    return `${n} hour${n === 1 ? '' : 's'} ago`
  }
  const n = Math.floor(diff / DAY)
  return n === 1 ? 'yesterday' : `${n} days ago`
}

/** "3 Oct 2026" in UTC. */
export function utcDate(iso: string): string | null {
  const t = parse(iso)
  if (t === null) return null
  return new Date(t).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })
}

/** "3 Oct 2026, 02:41 UTC". */
export function utcDateTime(iso: string): string | null {
  const t = parse(iso)
  if (t === null) return null
  const time = new Date(t).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'UTC' })
  return `${utcDate(iso)}, ${time} UTC`
}

/** Site-level freshness: "Data refreshed 5 hours ago (3 Oct 2026 UTC)". */
export function dataRefreshedText(iso: string | null | undefined, now: Date = new Date()): string | null {
  if (!iso) return null
  const rel = relativeTime(iso, now)
  const day = utcDate(iso)
  return rel && day ? `Data refreshed ${rel} (${day} UTC)` : null
}

/** Card-level freshness: "Fetched 5 hours ago · 3 Oct 2026, 02:41 UTC". */
export function fetchedText(iso: string | null | undefined, now: Date = new Date()): string | null {
  if (!iso) return null
  const rel = relativeTime(iso, now)
  const when = utcDateTime(iso)
  return rel && when ? `Fetched ${rel} · ${when}` : null
}

/** Whether the last refresh is older than `maxDays` days (the daily job may have stopped). */
export function isStale(iso: string | null | undefined, now: Date = new Date(), maxDays = 2): boolean {
  const t = parse(iso)
  return t !== null && now.getTime() - t > maxDays * DAY
}
