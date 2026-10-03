import { describe, expect, it } from 'vitest'

import { partitionRows } from './cache'

describe('partitionRows', () => {
  it('keeps unexpired and non-expiring rows and expires the rest (6h TTL rows)', () => {
    const now = 1_000_000
    const { live, expired } = partitionRows(
      [
        ['fresh', [now + 6 * 3600, '{"a":1}']],
        ['forever', [null, '"x"']],
        ['stale', [now - 1, '{}']],
        ['exactly-now', [now, '{}']],
      ],
      now,
    )
    expect(live).toEqual([
      ['fresh', now + 6 * 3600, '{"a":1}'],
      ['forever', null, '"x"'],
    ])
    expect(expired).toEqual(['stale', 'exactly-now'])
  })

  it('drops malformed rows instead of throwing', () => {
    const { live, expired } = partitionRows(
      [
        [42, [null, '{}']],
        ['not-array', 'oops'],
        ['wrong-length', [null]],
        ['non-string', [null, { a: 1 }]],
      ],
      0,
    )
    expect(live).toEqual([])
    expect(expired).toEqual([42, 'not-array', 'wrong-length', 'non-string'])
  })
})
