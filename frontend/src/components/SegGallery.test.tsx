import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { SegSeed } from '@/api/types'
import { SegGallery } from './SegGallery'

const seeds: SegSeed[] = [
  {
    url: '/api/file?path=a.png',
    source: 'a.png',
    status: 'kept',
    confidence: 0.82,
    method: 'rembg',
    reason: '',
  },
  {
    url: '/api/file?path=b.png',
    source: 'b.png',
    status: 'kept',
    confidence: 0.4,
    method: 'otsu',
    reason: '',
  },
  {
    url: '/api/file?path=c.png',
    source: 'c.png',
    status: 'skipped',
    confidence: 0.2,
    method: '-',
    reason: 'low_confidence',
  },
]

describe('SegGallery', () => {
  it('labels the filter tabs with kept/skipped counts', () => {
    render(<SegGallery seeds={seeds} thresh={0.55} />)
    expect(screen.getByRole('tab', { name: 'All 3' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Kept 2' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Skipped 1' })).toBeInTheDocument()
  })

  it('shows a per-seed confidence on every card', () => {
    render(<SegGallery seeds={seeds} thresh={0.55} />)
    expect(screen.getByText('conf 0.82')).toBeInTheDocument()
    expect(screen.getByText('conf 0.20')).toBeInTheDocument()
  })

  it('filters to only skipped seeds when the Skipped tab is selected', () => {
    render(<SegGallery seeds={seeds} thresh={0.55} />)
    fireEvent.click(screen.getByRole('tab', { name: 'Skipped 1' }))
    expect(screen.getByText('low_confidence')).toBeInTheDocument()
    expect(screen.queryByText('rembg')).toBeNull() // a kept card is hidden
  })
})
