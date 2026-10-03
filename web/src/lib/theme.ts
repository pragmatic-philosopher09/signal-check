import { useCallback, useSyncExternalStore } from 'react'

const KEY = 'signalcheck:theme'

function subscribe(cb: () => void): () => void {
  const observer = new MutationObserver(cb)
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
  return () => observer.disconnect()
}

const isDark = () => document.documentElement.classList.contains('dark')

/** Light/dark mode: follows the system until the user picks one (saved locally). */
export function useTheme(): { dark: boolean; toggle: () => void } {
  const dark = useSyncExternalStore(subscribe, isDark, () => false)
  const toggle = useCallback(() => {
    const next = !isDark()
    document.documentElement.classList.toggle('dark', next)
    try {
      localStorage.setItem(KEY, next ? 'dark' : 'light')
    } catch {
      // storage unavailable (private mode): the choice lasts for this page only
    }
  }, [])
  return { dark, toggle }
}
