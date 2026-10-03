import { bisector } from 'd3-array'
import { scaleLinear, scaleUtc } from 'd3-scale'
import { line as d3line } from 'd3-shape'
import { utcFormat } from 'd3-time-format'
import { useEffect, useId, useMemo, useRef, useState, type PointerEvent } from 'react'
import { buildChartModel, legendId, type Dash, type LegendEntry, type MarkerSymbol } from '../lib/chartModel'
import { formatDay, formatNumber } from '../lib/viewModel'
import type { ChartData } from '../types'

const DASH: Record<Dash, string | undefined> = { solid: undefined, dash: '6 4', dot: '2 3' }
const M = { top: 10, right: 14, bottom: 26, left: 46 }
const DAY = 86_400_000

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => setWidth(Math.floor(entry?.contentRect.width ?? 0)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, width]
}

function Marker({ x, y, symbol, size, color, hollow = false }: {
  x: number
  y: number
  symbol: MarkerSymbol
  size: number
  color: string
  hollow?: boolean
}) {
  const r = size / 2
  const fill = hollow ? 'var(--chart-bg)' : color
  const common = { stroke: color, strokeWidth: hollow ? 1.5 : 1, fill }
  if (symbol === 'square') return <rect x={x - r} y={y - r} width={size} height={size} {...common} />
  if (symbol === 'diamond') return <path d={`M${x},${y - r * 1.2}L${x + r},${y}L${x},${y + r * 1.2}L${x - r},${y}Z`} {...common} />
  if (symbol === 'x') {
    const k = r * 0.8
    return <path d={`M${x - k},${y - k}L${x + k},${y + k}M${x - k},${y + k}L${x + k},${y - k}`} stroke={color} strokeWidth={2.4} strokeLinecap="round" />
  }
  return <circle cx={x} cy={y} r={r} {...common} />
}

function Swatch({ entry }: { entry: LegendEntry }) {
  const c = entry.color
  switch (entry.kind) {
    case 'span':
      return <span className="h-3 w-4 rounded-sm bg-amber-300/40 dark:bg-amber-300/25" />
    case 'band':
      return <span className="h-3 w-4 rounded-sm bg-sky-600/20 dark:bg-sky-400/25" />
    case 'points':
    case 'imputed':
      return (
        <svg width="14" height="14" aria-hidden>
          <Marker x={7} y={7} symbol={entry.symbol ?? 'circle'} size={entry.kind === 'imputed' ? 8 : 9} color={c ?? 'var(--chart-series)'} hollow={entry.kind === 'imputed'} />
        </svg>
      )
    default: {
      const stroke =
        entry.kind === 'series'
          ? 'var(--chart-series)'
          : entry.kind === 'vline'
            ? 'var(--chart-cp)'
            : entry.kind === 'threshold' || entry.kind === 'partial'
              ? '#9e9e9e'
              : (c ?? '#616161')
      const dash = entry.kind === 'vline' ? 'dash' : entry.kind === 'partial' ? 'dot' : (entry.dash ?? 'solid')
      return (
        <svg width="18" height="8" aria-hidden>
          <line x1="1" x2="17" y1="4" y2="4" stroke={stroke} strokeWidth={2} strokeDasharray={DASH[dash]} />
        </svg>
      )
    }
  }
}

interface Hover {
  px: number
  t: number
  v: number | null
}

export function Chart({ chart, headroom, height = 300, title }: { chart: ChartData; headroom: number | null; height?: number; title: string }) {
  const model = useMemo(() => buildChartModel(chart, headroom), [chart, headroom])
  const [ref, width] = useWidth<HTMLDivElement>()
  const [hidden, setHidden] = useState<Set<string>>(() => new Set(model.legend.filter((e) => !e.visible).map((e) => e.id)))
  const [hover, setHover] = useState<Hover | null>(null)
  const clipId = useId()

  const h = width < 480 ? Math.round(height * 0.82) : height
  const x = useMemo(() => scaleUtc().domain(model.xDomain).range([M.left, Math.max(M.left + 1, width - M.right)]), [model, width])
  const y = useMemo(() => scaleLinear().domain(model.yDomain).nice(5).range([h - M.bottom, M.top]), [model, h])
  const show = (id: string) => !hidden.has(id)

  const seriesPath = useMemo(
    () =>
      d3line<{ t: number; v: number | null }>()
        .defined((p) => p.v !== null)
        .x((p) => x(p.t))
        .y((p) => y(p.v as number))(model.series) ?? '',
    [model, x, y],
  )

  const spanYears = (model.xDomain[1] - model.xDomain[0]) / (365 * DAY)
  const tickFmt = utcFormat(spanYears > 1.2 ? '%b %Y' : '%-d %b')
  const xTicks = width ? x.ticks(width < 480 ? 4 : 7) : []
  const yTicks = y.ticks(5)

  const bisect = useMemo(() => bisector<{ t: number }, number>((p) => p.t).center, [])
  const onMove = (e: PointerEvent<SVGRectElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const t = x.invert(e.clientX - rect.left + M.left).getTime()
    const i = bisect(model.series, t)
    const p = model.series[i]
    if (p) setHover({ px: x(p.t), t: p.t, v: p.v })
  }

  const hoverNotes = useMemo(() => {
    if (!hover) return []
    const notes: string[] = []
    for (const g of model.points) for (const p of g.points) if (p.t === hover.t && show(legendId.points(g.group))) notes.push(p.label)
    if (model.imputed.some((p) => p.t === hover.t)) notes.push('imputed (not observed)')
    for (const v of model.vlines) if (v.t === hover.t) notes.push(v.group)
    return notes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hover, model, hidden])

  const toggle = (id: string) =>
    setHidden((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  const first = model.series[0]
  const last = model.series[model.series.length - 1]
  const aria = `${title}: ${model.seriesLabel} from ${first ? formatDay(first.t) : ''} to ${last ? formatDay(last.t) : ''}` +
    (model.spans.length ? '; recent window shaded' : '') +
    (model.vlines.length ? `; change point on ${formatDay(model.vlines[0]!.t)}` : '')

  return (
    <figure className="chart-root">
      <div ref={ref} className="relative w-full" style={{ height: h }}>
        {width > 0 && (
          <svg width={width} height={h} role="img" aria-label={aria} className="block select-none">
            <defs>
              <clipPath id={clipId}>
                <rect x={M.left - 6} y={M.top - 6} width={Math.max(0, width - M.left - M.right + 12)} height={h - M.top - M.bottom + 12} />
              </clipPath>
            </defs>
            {/* grid + y axis */}
            {yTicks.map((tv) => (
              <g key={`y${tv}`}>
                <line x1={M.left} x2={width - M.right} y1={y(tv)} y2={y(tv)} className="stroke-slate-200 dark:stroke-slate-700/70" strokeWidth={1} />
                <text x={M.left - 8} y={y(tv)} dy="0.32em" textAnchor="end" className="fill-slate-500 text-[10.5px] tabular-nums dark:fill-slate-400">
                  {formatNumber(tv)}
                </text>
              </g>
            ))}
            {xTicks.map((tv) => (
              <text
                key={`x${tv.getTime()}`}
                x={x(tv)}
                y={h - 8}
                textAnchor={x(tv) > width - M.right - 24 ? 'end' : x(tv) < M.left + 24 ? 'start' : 'middle'}
                className="fill-slate-500 text-[10.5px] dark:fill-slate-400"
              >
                {tickFmt(tv)}
              </text>
            ))}
            <g clipPath={`url(#${clipId})`}>
              {model.spans.map((s, i) =>
                show(legendId.span(s.group)) ? (
                  <rect key={`s${i}`} x={x(s.x0)} width={Math.max(1, x(s.x1) - x(s.x0))} y={M.top - 6} height={h - M.top - M.bottom + 6} fill="var(--chart-recent)" />
                ) : null,
              )}
              {model.bands.map((b, i) => {
                if (!show(legendId.band(b.group))) return null
                if (b.kind === 'rect') {
                  return <rect key={`b${i}`} x={x(b.x0)} width={Math.max(1, x(b.x1) - x(b.x0))} y={y(b.y1)} height={Math.max(1, y(b.y0) - y(b.y1))} fill="var(--chart-band)" />
                }
                return <line key={`b${i}`} x1={x(b.x0)} x2={x(b.x1)} y1={y(b.y)} y2={y(b.y)} stroke="#9e9e9e" strokeWidth={1} strokeDasharray={DASH[b.dash]} />
              })}
              {model.lines.map((l, i) =>
                show(legendId.line(l.group)) ? (
                  <path key={`l${i}`} d={d3line<{ t: number; v: number }>().x((p) => x(p.t)).y((p) => y(p.v))(l.points) ?? ''} fill="none" stroke="var(--chart-seasonal)" strokeWidth={1.5} strokeDasharray={DASH.dot} />
                ) : null,
              )}
              {show('series') && <path d={seriesPath} fill="none" stroke="var(--chart-series)" strokeWidth={1.6} strokeLinejoin="round" strokeLinecap="round" />}
              {show('imputed') && model.imputed.map((p, i) => <Marker key={`i${i}`} x={x(p.t)} y={y(p.v)} symbol="circle" size={7} color="var(--chart-series)" hollow />)}
              {model.segments.map((s, i) =>
                show(legendId.segment(s.group)) ? (
                  <line key={`g${i}`} x1={x(s.from.t)} y1={y(s.from.v)} x2={x(s.to.t)} y2={y(s.to.v)} stroke={s.color} strokeWidth={2} strokeDasharray={DASH[s.dash]} strokeLinecap="round" className="seg" />
                ) : null,
              )}
              {model.points.map((g) =>
                show(legendId.points(g.group))
                  ? g.points.map((p, i) => <Marker key={`${g.group}${i}`} x={x(p.t)} y={y(p.v)} symbol={g.symbol} size={g.size} color={g.color} />)
                  : null,
              )}
              {model.vlines.map((v, i) =>
                show(legendId.vline(v.group)) ? (
                  <line key={`v${i}`} x1={x(v.t)} x2={x(v.t)} y1={M.top - 6} y2={h - M.bottom} stroke="var(--chart-cp)" strokeWidth={1.5} strokeDasharray={DASH.dash} />
                ) : null,
              )}
              {model.partial && show('partial') && (
                <g>
                  {model.partial.from && (
                    <line x1={x(model.partial.from.t)} y1={y(model.partial.from.v)} x2={x(model.partial.to.t)} y2={y(model.partial.to.v)} stroke="#9e9e9e" strokeWidth={1.2} strokeDasharray={DASH.dot} />
                  )}
                  <Marker x={x(model.partial.to.t)} y={y(model.partial.to.v)} symbol="circle" size={8} color="#9e9e9e" />
                </g>
              )}
            </g>
            {hover && (
              <g pointerEvents="none">
                <line x1={hover.px} x2={hover.px} y1={M.top - 6} y2={h - M.bottom} className="stroke-slate-400/70 dark:stroke-slate-500" strokeWidth={1} />
                {hover.v !== null && <circle cx={hover.px} cy={y(hover.v)} r={4} fill="var(--chart-series)" stroke="var(--chart-bg)" strokeWidth={2} />}
              </g>
            )}
            <rect
              x={M.left}
              y={0}
              width={Math.max(0, width - M.left - M.right)}
              height={h - M.bottom}
              fill="transparent"
              onPointerMove={onMove}
              onPointerDown={onMove}
              onPointerLeave={() => setHover(null)}
              style={{ touchAction: 'pan-y' }}
            />
          </svg>
        )}
        {hover && width > 0 && (
          <div
            className="pointer-events-none absolute top-1 z-10 rounded-lg border border-slate-200 bg-white/95 px-2.5 py-1.5 text-xs shadow-lg backdrop-blur dark:border-slate-700 dark:bg-slate-900/95"
            style={hover.px > width / 2 ? { right: width - hover.px + 10 } : { left: hover.px + 10 }}
          >
            <div className="font-medium text-slate-900 dark:text-slate-100">{formatDay(hover.t)}</div>
            <div className="tabular-nums text-slate-600 dark:text-slate-300">
              {hover.v === null ? 'missing (gap)' : `${formatNumber(hover.v)} ${model.seriesLabel}`}
            </div>
            {hoverNotes.map((n) => (
              <div key={n} className="text-slate-500 dark:text-slate-400">
                {n}
              </div>
            ))}
          </div>
        )}
      </div>
      <figcaption className="mt-2 flex flex-wrap gap-x-1 gap-y-1">
        {model.legend.map((entry) => (
          <button
            key={entry.id}
            type="button"
            onClick={() => toggle(entry.id)}
            aria-pressed={show(entry.id)}
            className={`inline-flex items-center gap-1.5 rounded-md px-1.5 py-0.5 text-[11px] text-slate-600 transition hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800 ${show(entry.id) ? '' : 'opacity-40 line-through'}`}
          >
            <Swatch entry={entry} />
            {entry.label}
          </button>
        ))}
      </figcaption>
    </figure>
  )
}
