// Pyodide release that runs the engine (Python 3.13, numpy 2.2, pandas 2.3,
// scipy 1.14, statsmodels 0.14). Keep in sync with the `pyodide` devDependency
// (checked by config.test.ts) — scripts/pyodide-parity.mjs runs that version.
export const PYODIDE_VERSION = '0.29.5'
export const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`
