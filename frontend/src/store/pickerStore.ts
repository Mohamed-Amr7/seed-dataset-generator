import { create } from 'zustand'

export type PickerKind = 'dir' | 'file'

// A single shared filesystem-picker modal, triggerable from any path field (mirrors the
// original single-modal design). The opener supplies the kind + an onSelect callback.
interface PickerState {
  open: boolean
  kind: PickerKind
  fieldName: string
  initialPath: string | null
  onSelect: ((path: string) => void) | null
  openPicker: (opts: {
    kind: PickerKind
    fieldName?: string
    initialPath?: string | null
    onSelect: (path: string) => void
  }) => void
  close: () => void
}

export const usePickerStore = create<PickerState>((set) => ({
  open: false,
  kind: 'dir',
  fieldName: 'path',
  initialPath: null,
  onSelect: null,
  openPicker: ({ kind, fieldName = 'path', initialPath = null, onSelect }) =>
    set({ open: true, kind, fieldName, initialPath, onSelect }),
  close: () => set({ open: false, onSelect: null }),
}))
