import { describe, expect, it } from 'vitest'

import type { Annotation, ChartData } from '../types'
import { annotationItems, buildChartModel, dataPeak, periodEnd, pointGroup, toTime } from './chartModel'

const DAY = 86_400_000

function chart(overrides: Partial<ChartData> = {}): ChartData {
  return {
    freq: 'D',
    scale: 'pageviews',
    observed: [
      ['2026-01-01', 10],
      ['2026-01-02', 12],
      ['2026-01-03', null],
      ['2026-01-04', 14],
    ],
    imputed: [],
    dropped_partial: null,
    window: {
      baseline_start: '2026-01-01',
      baseline_end: '2026-01-02',
      recent_start: '2026-01-03',
      recent_end: '2026-01-04',
    },
    annotations: [],
    ...overrides,
  }
}

describe('periodEnd', () => {
  it('ends a day, week or calendar month after the period start (exclusive)', () => {
    expect(periodEnd('2026-01-31', 'D')).toBe(toTime('2026-02-01'))
    expect(periodEnd('2026-01-26', 'W')).toBe(toTime('2026-02-02'))
    expect(periodEnd('2026-01-01', 'M')).toBe(toTime('2026-02-01'))
    expect(periodEnd('2026-12-01', 'M')).toBe(toTime('2027-01-01'))
  })

  it('ignores a time component', () => {
    expect(toTime('2026-03-04T12:34:56')).toBe(toTime('2026-03-04'))
    expect(periodEnd('2026-03-04', 'D') - toTime('2026-03-04')).toBe(DAY)
  })
})

describe('annotationItems', () => {
  it('merges one kind across evidence and drops exact duplicates regardless of key order', () => {
    const a: Annotation[] = [
      { spans: [{ label: 'recent window', start: '2026-01-03', end: '2026-01-04' }] },
      { spans: [{ end: '2026-01-04', start: '2026-01-03', label: 'recent window' }] },
      { spans: [{ label: 'other', start: '2026-01-01', end: '2026-01-02' }], points: [] },
      {},
    ]
    expect(annotationItems(a, 'spans').map((s) => s.label)).toEqual(['recent window', 'other'])
    expect(annotationItems(a, 'points')).toEqual([])
  })
})

describe('pointGroup', () => {
  it('folds "top N excess" labels into one legend group', () => {
    expect(pointGroup('top 1 excess')).toBe('largest excess')
    expect(pointGroup('top 12 excess')).toBe('largest excess')
    expect(pointGroup('outlier')).toBe('outlier')
  })
})

describe('dataPeak', () => {
  it('considers observed, imputed, points, segments, lines and only the baseline band upper', () => {
    const c = chart({
      imputed: [['2026-01-03', 20]],
      annotations: [
        { bands: [{ label: 'outlier threshold', start: '2026-01-01', end: '2026-01-04', lower: 0, upper: 999 }] },
        { bands: [{ label: 'baseline band', start: '2026-01-01', end: '2026-01-02', lower: 5, upper: 25 }] },
      ],
    })
    expect(dataPeak(c)).toBe(25)
    expect(dataPeak(chart({ observed: [['2026-01-01', null]] }))).toBeNull()
  })
})

describe('buildChartModel', () => {
  it('maps the observed series with gaps and hollow imputed points', () => {
    const m = buildChartModel(chart({ imputed: [['2026-01-03', 11]], scale: 'relative_0_100' }))
    expect(m.series.map((p) => p.v)).toEqual([10, 12, null, 14])
    expect(m.imputed).toEqual([{ t: toTime('2026-01-03'), v: 11 }])
    expect(m.seriesLabel).toBe('relative 0 100')
    expect(m.legend.map((e) => e.id)).toContain('imputed')
  })

  it('falls back to the verdict window for the recent-window span', () => {
    const m = buildChartModel(chart())
    expect(m.spans).toEqual([
      { group: 'recent window', x0: toTime('2026-01-03'), x1: toTime('2026-01-05') },
    ])
    expect(buildChartModel(chart({ window: null })).spans).toEqual([])
  })

  it('draws the baseline band as a rect, equal bounds dashed, thresholds dotted (no lower at 0)', () => {
    const m = buildChartModel(
      chart({
        annotations: [
          {
            bands: [
              { label: 'baseline band', start: '2026-01-01', end: '2026-01-02', lower: 8, upper: 14 },
              { label: 'baseline median', start: '2026-01-01', end: '2026-01-04', lower: 11, upper: 11 },
              { label: 'outlier threshold', start: '2026-01-01', end: '2026-01-04', lower: 0, upper: 30 },
              { label: 'practical band', start: '2026-01-01', end: '2026-01-04', lower: 2, upper: 20 },
            ],
          },
        ],
      }),
    )
    expect(m.bands).toEqual([
      { kind: 'rect', group: 'baseline band', x0: toTime('2026-01-01'), x1: toTime('2026-01-03'), y0: 8, y1: 14 },
      { kind: 'hline', group: 'baseline median', x0: toTime('2026-01-01'), x1: toTime('2026-01-05'), y: 11, dash: 'dash' },
      { kind: 'hline', group: 'outlier threshold', x0: toTime('2026-01-01'), x1: toTime('2026-01-05'), y: 30, dash: 'dot' },
      { kind: 'hline', group: 'practical band', x0: toTime('2026-01-01'), x1: toTime('2026-01-05'), y: 20, dash: 'dot' },
      { kind: 'hline', group: 'practical band', x0: toTime('2026-01-01'), x1: toTime('2026-01-05'), y: 2, dash: 'dot' },
    ])
  })

  it('omits a threshold far above the data (headroom ceiling) and its legend entry', () => {
    const c = chart({
      annotations: [
        { bands: [{ label: 'outlier threshold', start: '2026-01-01', end: '2026-01-04', lower: 0, upper: 500 }] },
      ],
    })
    const capped = buildChartModel(c, 3)
    expect(capped.peak).toBe(14)
    expect(capped.ceiling).toBe(42)
    expect(capped.bands).toEqual([])
    expect(capped.legend.some((e) => e.id === 'band:outlier threshold')).toBe(false)
    expect(capped.yDomain[1]).toBeLessThan(100)

    const uncapped = buildChartModel(c, null)
    expect(uncapped.bands).toHaveLength(1)
    expect(uncapped.yDomain[1]).toBeGreaterThan(500)
  })

  it('groups highlighted points by legend group with their styles', () => {
    const m = buildChartModel(
      chart({
        annotations: [
          {
            points: [
              { label: 'top 1 excess', ts: '2026-01-04', value: 14 },
              { label: 'top 2 excess', ts: '2026-01-02', value: 12 },
              { label: 'outlier', ts: '2026-01-04', value: 14 },
              { label: 'outlier', ts: '2026-01-01', value: Number.NaN },
            ],
          },
        ],
      }),
    )
    expect(m.points.map((g) => [g.group, g.points.length])).toEqual([
      ['largest excess', 2],
      ['outlier', 1],
    ])
    expect(m.points[0]?.points[1]?.label).toBe('top 2 excess')
    expect(m.points[0]?.symbol).toBe('diamond')
  })

  it('keeps mean-rate segments hidden by default but visible trend fits', () => {
    const m = buildChartModel(
      chart({
        annotations: [
          {
            segments: [
              { label: 'Theil-Sen trend', start: '2026-01-01', end: '2026-01-04', start_value: 10, end_value: 14 },
              { label: 'baseline mean rate', start: '2026-01-01', end: '2026-01-02', start_value: 11, end_value: 11 },
            ],
          },
        ],
      }),
    )
    const legend = Object.fromEntries(m.legend.map((e) => [e.id, e.visible]))
    expect(legend['segment:Theil-Sen trend']).toBe(true)
    expect(legend['segment:baseline mean rate']).toBe(false)
    expect(m.segments.map((s) => s.from)).toEqual([
      { t: toTime('2026-01-01'), v: 10 },
      { t: toTime('2026-01-01'), v: 11 },
    ])
  })

  it('draws change-point lines and seasonal reference lines', () => {
    const m = buildChartModel(
      chart({
        annotations: [
          { vlines: [{ label: 'change point', ts: '2026-01-03' }] },
          { lines: [{ label: 'baseline + annual pattern', points: [{ ts: '2026-01-03', value: 9 }, { ts: '2026-01-04', value: 9.5 }] }] },
          { lines: [{ label: 'empty', points: [] }] },
        ],
      }),
    )
    expect(m.vlines).toEqual([{ group: 'change point', t: toTime('2026-01-03') }])
    expect(m.lines.map((l) => l.group)).toEqual(['baseline + annual pattern'])
  })

  it('links the greyed dropped partial period from the last observed value', () => {
    const m = buildChartModel(chart({ dropped_partial: { ts: '2026-01-05', value: 3 } }))
    expect(m.partial).toEqual({
      from: { t: toTime('2026-01-04'), v: 14 },
      to: { t: toTime('2026-01-05'), v: 3 },
    })
    expect(m.xDomain[1]).toBe(toTime('2026-01-05'))
    expect(m.legend.at(-1)?.id).toBe('partial')
  })

  it('starts the y axis at zero for positive data and pads below zero otherwise', () => {
    expect(buildChartModel(chart()).yDomain[0]).toBe(0)
    const neg = buildChartModel(chart({ observed: [['2026-01-01', -5], ['2026-01-02', 5]] }))
    expect(neg.yDomain[0]).toBeLessThan(-5)
    expect(neg.yDomain[1]).toBeGreaterThan(5)
  })

  it('handles an empty chart', () => {
    const m = buildChartModel(chart({ observed: [], window: null }))
    expect(m.xDomain).toEqual([0, 1])
    expect(m.yDomain[0]).toBe(0)
    expect(m.peak).toBeNull()
  })
})
