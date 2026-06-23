import { X } from 'lucide-react'
import { useToastStore } from '@/store/toastStore'
import { cn } from '@/lib/utils'

/** Renders the global toast queue (see store/toastStore). Mount once near the app root. */
export function Toaster() {
  const toasts = useToastStore((s) => s.toasts)
  const dismiss = useToastStore((s) => s.dismiss)
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-[100] flex w-80 max-w-[calc(100vw-2rem)] flex-col gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          role={t.variant === 'error' ? 'alert' : 'status'}
          className={cn(
            'pointer-events-auto flex items-start justify-between gap-3 rounded-md border bg-card px-3 py-2 text-sm text-foreground shadow-lg',
            t.variant === 'success' && 'border-success/40',
            t.variant === 'error' && 'border-destructive/40',
            t.variant === 'default' && 'border-border',
          )}
        >
          <span className="whitespace-pre-wrap break-words">{t.message}</span>
          <button
            type="button"
            onClick={() => dismiss(t.id)}
            className="shrink-0 opacity-60 hover:opacity-100"
            aria-label="Dismiss notification"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      ))}
    </div>
  )
}
