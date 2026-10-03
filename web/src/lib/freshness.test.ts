import { describe, expect, it } from 'vitest'

import { dataRefreshedText, fetchedText, isStale, relativeTime, utcDate, utcDateTime } from './freshness'

const NOW = new Date('2026-10-03T12:00:00Z')

describe('relativeTime', () => {
  it.each([
    ['2026-10-03T11:59:30Z', 'just now'],
    ['2026-10-03T12:05:00Z', 'just now'], // clock skew: never "in the future"
    ['2026-10-03T11:59:00Z', '1 minute ago'],
    ['2026-10-03T11:15:00Z', '45 minutes ago'],
    ['2026-10-03T11:00:00Z', '1 hour ago'],
    ['2026-10-03T02:41:07+00:00', '9 hours ago'],
    ['2026-10-02T11:00:00Z', 'yesterday'],
    ['2026-09-30T12:00:00Z', '3 days ago'],
  ])('%s → %s', (iso, expected) => {
    expect(relativeTime(iso, NOW)).toBe(expected)
  })

  it('rejects invalid input', () => {
    expect(relativeTime('not a date', NOW)).toBeNull()
  })
})

describe('UTC formatting', () => {
  it('uses the UTC calendar day, not the local one', () => {
    expect(utcDate('2026-10-02T23:30:00-05:00')).toBe('3 Oct 2026')
    expect(utcDateTime('2026-10-03T02:41:07+00:00')).toBe('3 Oct 2026, 02:41 UTC')
  })
})

describe('dataRefreshedText', () => {
  it('reads like the header line', () => {
    expect(dataRefreshedText('2026-10-03T02:41:07+00:00', NOW)).toBe('Data refreshed 9 hours ago (3 Oct 2026 UTC)')
  })

  it('is null without a manifest', () => {
    expect(dataRefreshedText(null, NOW)).toBeNull()
    expect(dataRefreshedText(undefined, NOW)).toBeNull()
    expect(dataRefreshedText('garbage', NOW)).toBeNull()
  })
})

describe('fetchedText', () => {
  it('shows relative and absolute UTC time', () => {
    expect(fetchedText('2026-10-01T09:14:50+00:00', NOW)).toBe('Fetched 2 days ago · 1 Oct 2026, 09:14 UTC')
    expect(fetchedText(null, NOW)).toBeNull()
  })
})

describe('isStale', () => {
  it('flags refreshes older than two days', () => {
    expect(isStale('2026-10-02T02:00:00Z', NOW)).toBe(false)
    expect(isStale('2026-09-30T02:00:00Z', NOW)).toBe(true)
    expect(isStale(null, NOW)).toBe(false)
  })
})
