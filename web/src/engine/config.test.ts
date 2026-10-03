import { describe, expect, it } from 'vitest'

import pyodidePackage from '../../node_modules/pyodide/package.json'
import { PYODIDE_CDN, PYODIDE_VERSION } from './config'

describe('Pyodide config', () => {
  it('loads the CDN runtime matching the pinned npm package (used by the parity script)', () => {
    expect(PYODIDE_VERSION).toBe(pyodidePackage.version)
    expect(PYODIDE_CDN).toBe(`https://cdn.jsdelivr.net/pyodide/v${pyodidePackage.version}/full/`)
  })
})
