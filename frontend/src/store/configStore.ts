import { create } from 'zustand'
import { createJSONStorage, persist, type StateStorage } from 'zustand/middleware'
import type { ConfigDict, ConfigValue } from '@/api/types'

// localStorage can be absent or disabled (private mode, sandboxed iframes, some test
// environments). Fall back to an in-memory map so persistence degrades gracefully instead
// of throwing.
const memory = new Map<string, string>()
const safeStorage: StateStorage = {
  getItem: (key) => {
    try {
      return localStorage.getItem(key)
    } catch {
      return memory.get(key) ?? null
    }
  },
  setItem: (key, value) => {
    try {
      localStorage.setItem(key, value)
    } catch {
      memory.set(key, value)
    }
  },
  removeItem: (key) => {
    try {
      localStorage.removeItem(key)
    } catch {
      memory.delete(key)
    }
  },
}

// The single working config, shared across the Run form, the Segmentation tuner, and the
// Config tab (mirrors the old global `STATE`). Persisted to localStorage so a refresh
// doesn't wipe an in-progress setup.
interface ConfigState {
  config: ConfigDict
  /** True once the recommended defaults (or a persisted config) have been applied. */
  seeded: boolean
  setField: (name: string, value: ConfigValue) => void
  setMany: (values: ConfigDict) => void
  /** Replace the whole config (preset switch / load-from-file). */
  replace: (values: ConfigDict) => void
  markSeeded: () => void
}

export const useConfigStore = create<ConfigState>()(
  persist(
    (set) => ({
      config: {},
      seeded: false,
      setField: (name, value) => set((s) => ({ config: { ...s.config, [name]: value } })),
      setMany: (values) => set((s) => ({ config: { ...s.config, ...values } })),
      replace: (values) => set({ config: { ...values }, seeded: true }),
      markSeeded: () => set({ seeded: true }),
    }),
    {
      name: 'msg-config',
      storage: createJSONStorage(() => safeStorage),
      partialize: (s) => ({ config: s.config, seeded: s.seeded }),
    },
  ),
)

/** Drop empty / null values so optional fields fall back to backend defaults. */
export function cleanConfig(config: ConfigDict): ConfigDict {
  const out: ConfigDict = {}
  for (const [k, v] of Object.entries(config)) {
    if (v === null || v === undefined) continue
    if (typeof v === 'string' && v.trim() === '') continue
    if (Array.isArray(v) && v.length === 0) continue
    out[k] = v
  }
  return out
}
