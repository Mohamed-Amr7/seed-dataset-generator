import { create } from 'zustand'

export type ToastVariant = 'default' | 'success' | 'error'

export interface Toast {
  id: number
  message: string
  variant: ToastVariant
}

interface ToastState {
  toasts: Toast[]
  push: (message: string, variant?: ToastVariant) => void
  dismiss: (id: number) => void
}

let counter = 0

export const useToastStore = create<ToastState>((set) => ({
  toasts: [],
  push: (message, variant = 'default') => {
    const id = ++counter
    set((s) => ({ toasts: [...s.toasts, { id, message, variant }] }))
    // auto-dismiss after a few seconds
    setTimeout(() => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })), 5000)
  },
  dismiss: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}))

/** Convenience helpers for non-component code. */
export const toast = {
  success: (m: string) => useToastStore.getState().push(m, 'success'),
  error: (m: string) => useToastStore.getState().push(m, 'error'),
  info: (m: string) => useToastStore.getState().push(m, 'default'),
}
