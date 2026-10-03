// Runs the shipped signalcheck bundle inside Pyodide (Node) over the bundled
// samples and diffs the output against the CPython-precomputed JSON.
import { readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { loadPyodide } from 'pyodide'

const here = dirname(fileURLToPath(import.meta.url))
const pub = join(here, '..', 'public')
const manifest = JSON.parse(readFileSync(join(pub, 'py', 'manifest.json'), 'utf8'))
const site = JSON.parse(readFileSync(join(pub, 'site.json'), 'utf8'))

const t0 = performance.now()
const py = await loadPyodide({
  packageCacheDir: join(tmpdir(), 'signalcheck-pyodide-cache'),
  packageBaseUrl: `https://cdn.jsdelivr.net/pyodide/v${(await import('pyodide/package.json', { with: { type: 'json' } })).default.version}/full/`,
})
await py.loadPackage(manifest.pyodide_packages, { messageCallback: () => {} })
for (const wheel of manifest.wheels) {
  const name = wheel
  py.FS.writeFile(`/tmp/${name}`, readFileSync(join(pub, 'py', name)))
  await py.runPythonAsync(`import micropip; await micropip.install("emfs:/tmp/${name}", deps=False)`)
}
py.FS.writeFile('/tmp/bundle.zip', readFileSync(join(pub, 'py', manifest.bundle)))
py.runPython(`
import sys, zipfile
zipfile.ZipFile('/tmp/bundle.zip').extractall('/home/pyodide/app')
sys.path.insert(0, '/home/pyodide/app')
import time
_t = time.time(); time.sleep(0.2); print('sleep(0.2) took', round(time.time() - _t, 3), 's')
import sys; print('python', sys.version.split()[0])
from signalcheck.web import worker
print(worker.boot('[]', None))
`)
console.log(`boot ${(performance.now() - t0).toFixed(0)} ms`)

const strip = (r) => {
  const c = structuredClone(r)
  delete c.generated_at
  for (const card of c.cards ?? []) card.caveats = (card.caveats ?? []).filter((s) => !s.startsWith('Cached snapshot fetched'))
  return c
}
let failures = 0
for (const sample of site.samples) {
  const t = performance.now()
  py.globals.set('q', sample.query)
  const out = JSON.parse(py.runPython('worker.run_sample_json(q, None)'))
  const ref = JSON.parse(readFileSync(join(pub, 'results', `${sample.slug}.json`), 'utf8'))
  const a = JSON.stringify(strip(out))
  const b = JSON.stringify(strip(ref))
  const same = a === b
  if (!same) {
    failures++
    for (let i = 0; i < Math.max(a.length, b.length); i++) {
      if (a[i] !== b[i]) {
        console.log('  first diff:\n   pyodide:', a.slice(Math.max(0, i - 200), i + 200), '\n   cpython:', b.slice(Math.max(0, i - 200), i + 200))
        break
      }
    }
  }
  console.log(`${sample.slug}: ${same ? 'identical' : 'DIFFERENT'} (${(performance.now() - t).toFixed(0)} ms)`)
}
process.exit(failures ? 1 : 0)
