// Contract test over the precomputed sample results written by
// `python -m scripts.build_web` (run it before `npm test`; CI does).

import { describe, expect, it } from 'vitest'

import type { Result, Site } from '../types'
import { buildChartModel } from './chartModel'
import { LABEL_COLORS } from './viewModel'

const results = import.meta.glob<Result>('../../public/results/*.json', { eager: true, import: 'default' })
const sites = import.meta.glob<Site>('../../public/site.json', { eager: true, import: 'default' })
const site = Object.values(sites)[0]

describe('precomputed samples', () => {
  it('exist (run `python -m scripts.build_web` first)', () => {
    expect(site).toBeDefined()
    expect(Object.keys(results).length).toBeGreaterThan(0)
    expect(Object.keys(results).length).toBe(site?.samples.length)
  })

  for (const [path, result] of Object.entries(results)) {
    describe(path.split('/').pop() ?? path, () => {
      it('follows the JSON contract', () => {
        expect(result.schema).toBe(1)
        expect(result.sample).toBe(true)
        expect(result.cards.length).toBeGreaterThan(0)
      })

      it('only shows the static-demo reason for credentialed sources', () => {
        for (const c of result.cards.filter((c) => c.source === 'reddit' || c.source === 'x')) {
          expect(c.disabled).toBe(true)
          expect(c.message).toMatch(/disabled \(needs server-side credentials; not available on the static demo\)$/)
          expect(c.chart).toBeNull()
        }
      })

      it('records when each analysed source was fetched', () => {
        for (const c of result.cards.filter((c) => c.badge)) {
          expect(c.fetched_at).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$/)
        }
      })

      it('gives every analysed card a consistent badge, a template narration and a drawable chart', () => {
        for (const c of result.cards.filter((c) => c.badge)) {
          const badge = c.badge!
          expect(LABEL_COLORS[badge.key]).toBe(badge.color)
          expect(c.narration?.label).toBe(site?.ui.template_label)
          expect(c.chart).not.toBeNull()
          const m = buildChartModel(c.chart!, site?.ui.threshold_headroom ?? null)
          expect(m.series.length).toBeGreaterThan(0)
          expect(m.spans.length).toBeGreaterThan(0)
          for (const v of [...m.xDomain, ...m.yDomain]) expect(Number.isFinite(v)).toBe(true)
          expect(m.yDomain[1]).toBeGreaterThan(m.yDomain[0])
        }
      })
    })
  }
})
