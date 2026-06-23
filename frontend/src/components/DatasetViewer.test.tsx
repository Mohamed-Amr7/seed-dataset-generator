import { render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/api/endpoints', () => ({ datasetImageUrl: () => '/api/datasets/image?x' }))

import { DatasetViewer } from './DatasetViewer'

const createObjectURL = vi.fn(() => 'blob:fake-1')
const revokeObjectURL = vi.fn()

beforeEach(() => {
  createObjectURL.mockClear()
  revokeObjectURL.mockClear()
  vi.stubGlobal('URL', { createObjectURL, revokeObjectURL })
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: (k: string) => (k === 'X-Total' ? '3' : 'a.jpg') },
      blob: () => Promise.resolve(new Blob()),
    }),
  )
})

afterEach(() => vi.unstubAllGlobals())

const noop = () => {}

describe('DatasetViewer', () => {
  it('creates an object URL for the image and revokes it on unmount (no leak)', async () => {
    const { unmount } = render(
      <DatasetViewer
        path="/d"
        split="train"
        idx={0}
        total={3}
        name=""
        onMeta={noop}
        onPrev={noop}
        onNext={noop}
      />,
    )

    await waitFor(() => expect(createObjectURL).toHaveBeenCalled())
    unmount()
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:fake-1')
  })
})
