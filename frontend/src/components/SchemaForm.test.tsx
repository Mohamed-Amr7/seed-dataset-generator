import { act, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import type { FormSchema } from '@/api/types'
import { useConfigStore } from '@/store/configStore'
import { SchemaForm } from './SchemaForm'

const schema: FormSchema = {
  groups: [
    {
      title: 'I/O',
      fields: [
        {
          name: 'sources',
          default: '',
          widget: 'text',
          choices: null,
          help: '',
          flag: '',
          path_kind: 'dir',
        },
        {
          name: 'segment',
          default: 'auto',
          widget: 'choice',
          choices: ['auto', 'grabcut', 'otsu'],
          help: '',
          flag: '',
          path_kind: null,
        },
      ],
    },
    {
      title: 'segmentation',
      fields: [
        {
          name: 'grabcut_iters',
          default: 5,
          widget: 'int',
          choices: null,
          help: '',
          flag: '',
          path_kind: null,
        },
        {
          name: 'otsu_blur_kernel',
          default: 0,
          widget: 'int',
          choices: null,
          help: '',
          flag: '',
          path_kind: null,
        },
      ],
    },
  ],
  recommended: {},
  presets: {},
  data_root: '/data',
  project_root: '/project',
}

beforeEach(() => {
  act(() => useConfigStore.setState({ config: { segment: 'auto' }, seeded: true }))
})

describe('SchemaForm', () => {
  it('renders the fields from each group', () => {
    render(<SchemaForm schema={schema} />)
    expect(screen.getByText(/sources/)).toBeInTheDocument()
    expect(screen.getByText('segment')).toBeInTheDocument()
  })

  it('hides per-method segmentation fields that do not match the selected method', () => {
    render(<SchemaForm schema={schema} />)
    expect(screen.queryByText('grabcut_iters')).toBeNull()
    expect(screen.queryByText('otsu_blur_kernel')).toBeNull()
  })

  it('reveals a method field when the segment method changes to it', () => {
    render(<SchemaForm schema={schema} />)
    act(() => useConfigStore.setState({ config: { segment: 'grabcut' } }))
    expect(screen.getByText('grabcut_iters')).toBeInTheDocument()
    expect(screen.queryByText('otsu_blur_kernel')).toBeNull()
  })
})
