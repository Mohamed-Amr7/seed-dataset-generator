import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

// Stub the schema query (for data_root) and the fs listing endpoint.
vi.mock('@/api/queries', () => ({ useSchema: () => ({ data: { data_root: '/data' } }) }))
vi.mock('@/api/endpoints', () => ({
  fsList: vi.fn().mockResolvedValue({
    path: '/data',
    parent: null,
    data_root: '/data',
    entries: [
      { name: 'alpha', path: '/data/alpha', is_dir: true },
      { name: 'beta', path: '/data/beta', is_dir: true },
      { name: 'gamma', path: '/data/gamma', is_dir: true },
    ],
  }),
}))

import { usePickerStore } from '@/store/pickerStore'
import { PathPicker } from './PathPicker'

beforeEach(() => {
  act(() => usePickerStore.setState({ open: false, onSelect: null }))
})

describe('PathPicker', () => {
  it('renders the sandboxed listing and moves focus with arrow keys', async () => {
    render(<PathPicker />)
    act(() => usePickerStore.getState().openPicker({ kind: 'dir', onSelect: vi.fn() }))

    await waitFor(() => expect(screen.getByText('alpha')).toBeInTheDocument())

    const rows = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-fs-row]'))
    expect(rows).toHaveLength(3)

    rows[0].focus()
    expect(document.activeElement).toBe(rows[0])

    fireEvent.keyDown(rows[0].parentElement!, { key: 'ArrowDown' })
    expect(document.activeElement).toBe(rows[1])

    fireEvent.keyDown(rows[1].parentElement!, { key: 'ArrowUp' })
    expect(document.activeElement).toBe(rows[0])
  })

  it('selects a directory via the confirm button', async () => {
    const onSelect = vi.fn()
    render(<PathPicker />)
    act(() => usePickerStore.getState().openPicker({ kind: 'dir', onSelect }))
    await waitFor(() => expect(screen.getByText('alpha')).toBeInTheDocument())

    fireEvent.click(screen.getByRole('button', { name: /select this folder/i }))
    expect(onSelect).toHaveBeenCalledWith('/data')
  })
})
