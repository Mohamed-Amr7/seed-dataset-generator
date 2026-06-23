import { useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'

export function RunLog({ lines, className }: { lines: string[]; className?: string }) {
  const ref = useRef<HTMLPreElement>(null)
  // Keep the view pinned to the newest output as lines stream in.
  useEffect(() => {
    const el = ref.current
    if (el) el.scrollTop = el.scrollHeight
  }, [lines])
  return (
    <pre
      ref={ref}
      role="log"
      aria-live="polite"
      aria-label="Run log"
      className={cn(
        'h-full overflow-auto whitespace-pre-wrap break-words rounded-md border border-border bg-background p-3 font-mono text-xs leading-relaxed',
        className,
      )}
    >
      {lines.length
        ? lines.join('\n')
        : 'No output yet. Set parameters and click Generate to start a run.'}
    </pre>
  )
}
