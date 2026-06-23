import { Badge, type BadgeVariant } from '@/components/ui/badge'
import type { RunPhase } from '@/hooks/useRunSocket'

const MAP: Record<Exclude<RunPhase, 'idle'>, { variant: BadgeVariant; label: string }> = {
  running: { variant: 'warning', label: 'running' },
  done: { variant: 'success', label: 'done' },
  error: { variant: 'destructive', label: 'error' },
}

export function StatusBadge({ status }: { status: RunPhase }) {
  if (status === 'idle') return null
  const { variant, label } = MAP[status]
  return <Badge variant={variant}>{label}</Badge>
}
