import { create } from 'zustand'

// One-shot channel for "open this dataset in the Browse tab". The Run tab sets a path when
// a generation finishes; App switches to the Browse tab; the Browse tab consumes it. Kept
// out of the URL/config so it's a transient hand-off, not persisted state.
interface BrowseState {
  requestedPath: string | null
  openInBrowser: (path: string) => void
  consume: () => void
}

export const useBrowseStore = create<BrowseState>((set) => ({
  requestedPath: null,
  openInBrowser: (path) => set({ requestedPath: path }),
  consume: () => set({ requestedPath: null }),
}))
