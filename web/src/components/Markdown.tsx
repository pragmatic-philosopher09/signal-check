import { marked } from 'marked'
import { useMemo } from 'react'

marked.setOptions({ gfm: true, breaks: false })

/** Render trusted Markdown generated at build time from the shipped config. */
export function Markdown({ text, className = '' }: { text: string; className?: string }) {
  const html = useMemo(() => marked.parse(text, { async: false }), [text])
  return <div className={`prose-sc ${className}`} dangerouslySetInnerHTML={{ __html: html }} />
}
