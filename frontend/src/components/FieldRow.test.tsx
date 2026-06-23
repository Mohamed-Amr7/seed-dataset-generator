import { act, fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import type { SchemaField } from '@/api/types'
import { useConfigStore } from '@/store/configStore'
import { usePickerStore } from '@/store/pickerStore'
import { FieldRow } from './FieldRow'

const pathField: SchemaField = {
  name: 'textures',
  default: '',
  widget: 'text',
  choices: null,
  help: '',
  flag: '',
  path_kind: 'dir',
}

beforeEach(() => {
  act(() => {
    useConfigStore.setState({ config: {}, seeded: true })
    usePickerStore.setState({ open: false, onSelect: null })
  })
})

describe('FieldRow path picker', () => {
  it('reflects a picked path in both the input and the config store', () => {
    render(<FieldRow field={pathField} />)

    // Open the picker, then invoke the onSelect it registered (what PathPicker does on pick).
    fireEvent.click(screen.getByRole('button', { name: /browse for textures/i }))
    act(() => usePickerStore.getState().onSelect?.('/data/textures/grass'))

    expect(screen.getByRole('textbox')).toHaveValue('/data/textures/grass')
    expect(useConfigStore.getState().config.textures).toBe('/data/textures/grass')
  })
})
