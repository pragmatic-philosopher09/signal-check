import { describe, expect, it } from 'vitest'

import type { Card } from '../types'
import {
  cardMessage,
  confidenceLevel,
  formatNumber,
  LABEL_COLORS,
  orderCards,
  refreshSnapshotCaveat,
  slugify,
  stanceKind,
} from './viewModel'

function card(overrides: Partial<Card>): Card {
  return {
    source: 'wikipedia',
    title: 'Wikipedia',
    message: null,
    disabled: false,
    snapshot: false,
    badge: null,
    label: null,
    direction: null,
    confidence: null,
    rule: null,
    rule_text: null,
    reason: null,
    window_text: null,
    evidence: [],
    skipped: [],
    change_my_mind: [],
    caveats: [],
    wiki: null,
    narration: null,
    chart: null,
    ...overrides,
  }
}

describe('verdict colours', () => {
  it('match the brief: up green, down red, fluke orange, seasonal violet, no change blue, inconclusive grey', () => {
    expect(LABEL_COLORS).toEqual({
      TREND_up: 'green',
      TREND_down: 'red',
      FLUKE: 'orange',
      SEASONAL: 'violet',
      NO_CHANGE: 'blue',
      INCONCLUSIVE: 'gray',
    })
  })
})

describe('stanceKind', () => {
  it('reads the supported label and treats anything else as neutral', () => {
    expect(stanceKind('supports_trend')).toBe('trend')
    expect(stanceKind('supports_no_change')).toBe('no_change')
    expect(stanceKind('supports_seasonal')).toBe('seasonal')
    expect(stanceKind('neutral')).toBe('neutral')
    expect(stanceKind('supports_something_else')).toBe('neutral')
  })
})

describe('confidenceLevel', () => {
  it('maps confidence to a 0-3 meter', () => {
    expect([confidenceLevel('low'), confidenceLevel('medium'), confidenceLevel('high')]).toEqual([1, 2, 3])
    expect(confidenceLevel(null)).toBe(0)
    expect(confidenceLevel('unknown')).toBe(0)
  })
})

describe('refreshSnapshotCaveat', () => {
  const caveat = 'Cached snapshot fetched on 2026-10-01 (0 day(s) ago); it may be out of date.'

  it('recomputes the snapshot age for today', () => {
    expect(refreshSnapshotCaveat(caveat, new Date('2026-10-08T23:00:00Z'))).toBe(
      'Cached snapshot fetched on 2026-10-01 (7 day(s) ago); it may be out of date.',
    )
  })

  it('never goes negative and leaves other caveats untouched', () => {
    expect(refreshSnapshotCaveat(caveat, new Date('2026-09-01T00:00:00Z'))).toContain('(0 day(s) ago)')
    expect(refreshSnapshotCaveat('Google samples this data.', new Date())).toBe('Google samples this data.')
  })
})

describe('slugify', () => {
  it('matches signalcheck.adapters.base.slugify', () => {
    expect(slugify('  Rust   programming ')).toBe('rust-programming')
    expect(slugify('Perplexity AI')).toBe('perplexity-ai')
    expect(slugify('C++ / Go!')).toBe('c-go')
  })
})

describe('formatNumber', () => {
  it('formats compactly', () => {
    expect(formatNumber(0)).toBe('0')
    expect(formatNumber(3.14159)).toBe('3.142')
    expect(formatNumber(1234)).toBe('1,234')
    expect(formatNumber(12_345)).toBe('12.3k')
    expect(formatNumber(2_500_000)).toBe('2.5M')
    expect(formatNumber(-4_200_000_000)).toBe('-4.2B')
  })
})

describe('orderCards', () => {
  it('puts analysed cards first, then failures, then disabled sources (stable)', () => {
    const badge = { text: 'Trend ↑', color: 'green' as const, key: 'TREND_up' as const }
    const cards = [
      card({ source: 'x', disabled: true, message: 'X disabled (needs credentials)' }),
      card({ source: 'hackernews', message: "Couldn't fetch Hacker News: timeout" }),
      card({ source: 'wikipedia', badge }),
      card({ source: 'reddit', disabled: true }),
      card({ source: 'google_trends', badge }),
    ]
    expect(orderCards(cards).map((c) => c.source)).toEqual(['wikipedia', 'google_trends', 'hackernews', 'x', 'reddit'])
  })
})

describe('cardMessage', () => {
  it('splits a disabled message into a title and the reason', () => {
    expect(
      cardMessage(card({ disabled: true, message: 'Reddit disabled (needs server-side credentials; not available on the static demo)' })),
    ).toEqual({
      title: 'Not available on the static demo',
      detail: 'needs server-side credentials; not available on the static demo',
    })
  })

  it('passes failure messages through unchanged', () => {
    expect(cardMessage(card({ message: "Couldn't fetch Wikipedia: HTTP 503" }))).toEqual({
      title: "Couldn't fetch Wikipedia: HTTP 503",
      detail: null,
    })
    expect(cardMessage(card({}))).toEqual({ title: '', detail: null })
  })
})
