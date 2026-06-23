import { RefreshCw } from 'lucide-react'
import { useRuns } from '@/api/queries'
import type { RunStatus } from '@/api/types'
import { Badge, type BadgeVariant } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/common'
import { cn } from '@/lib/utils'

const VARIANT: Record<RunStatus, BadgeVariant> = {
  running: 'warning',
  done: 'success',
  error: 'destructive',
}

export function RunHistory({
  activeId,
  poll,
  onSelect,
}: {
  activeId: string | null
  poll: boolean
  onSelect: (id: string) => void
}) {
  const { data, isLoading, refetch, isFetching } = useRuns(poll ? 2000 : false)
  const runs = data?.runs ?? []
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Recent runs</h3>
        <Button variant="ghost" size="icon" aria-label="Refresh run list" onClick={() => refetch()}>
          {isFetching ? <Spinner /> : <RefreshCw />}
        </Button>
      </div>
      {isLoading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : !runs.length ? (
        <p className="text-sm text-muted-foreground">No runs yet.</p>
      ) : (
        <ul className="flex max-h-40 flex-col gap-1 overflow-auto">
          {runs.map((r) => (
            <li key={r.id}>
              <button
                type="button"
                onClick={() => onSelect(r.id)}
                title={r.out}
                className={cn(
                  'flex w-full items-center justify-between gap-2 rounded-md border px-2 py-1.5 text-left text-xs hover:bg-accent',
                  r.id === activeId ? 'border-primary' : 'border-border',
                )}
              >
                <span className="truncate font-mono">{r.id}</span>
                <Badge variant={VARIANT[r.status]}>{r.status}</Badge>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
