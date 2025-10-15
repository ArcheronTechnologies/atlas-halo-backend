import React from 'react'

type Point = { x: number; y: number }

export default function Chart({
  data,
  width = 480,
  height = 180,
  stroke = '#64b5f6',
}: { data: number[] | Point[]; width?: number; height?: number; stroke?: string }) {
  if (!data || data.length === 0) return <svg width={width} height={height} />
  const pts: Point[] = (data as any[]).map((v: any, i) => typeof v === 'number' ? ({ x: i, y: v }) : v)
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y)
  const minX = Math.min(...xs), maxX = Math.max(...xs)
  const minY = Math.min(...ys), maxY = Math.max(...ys)
  const pad = 10
  const sx = (x: number) => pad + ((x - minX) / (maxX - minX || 1)) * (width - pad * 2)
  const sy = (y: number) => height - pad - ((y - minY) / (maxY - minY || 1)) * (height - pad * 2)
  const d = pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x)},${sy(p.y)}`).join(' ')
  return (
    <svg width={width} height={height}>
      <rect x={0} y={0} width={width} height={height} fill="#0f1a31" />
      <path d={d} fill="none" stroke={stroke} strokeWidth={2} />
    </svg>
  )
}

