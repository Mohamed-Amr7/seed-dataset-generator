import { FolderSearch, Play, Square } from 'lucide-react'
import { useEffect, useMemo } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { ApiError } from '@/api/client'
import { startRun, stopRun } from '@/api/endpoints'
import { qk, useRuns } from '@/api/queries'
import type { FormSchema } from '@/api/types'
import type { RunSocket } from '@/hooks/useRunSocket'
import { useBrowseStore } from '@/store/browseStore'
import { cleanConfig, useConfigStore } from '@/store/configStore'
import { toast } from '@/store/toastStore'
import { validateRunConfig } from '@/lib/validation'
import { Button } from '@/components/ui/button'
import { Spinner, ErrorText } from '@/components/common'
import { PresetSelect } from '@/components/PresetSelect'
import { SchemaForm } from '@/components/SchemaForm'
import { Dropzone } from '@/components/Dropzone'
import { RunHistory } from '@/components/RunHistory'
import { RunLog } from '@/components/RunLog'
import { StatusBadge } from '@/components/StatusBadge'

export function RunTab({ schema, socket }: { schema: FormSchema; socket: RunSocket }) {
  const config = useConfigStore((s) => s.config)
  const setField = useConfigStore((s) => s.setField)
  const openInBrowser = useBrowseStore((s) => s.openInBrowser)
  const queryClient = useQueryClient()
  const validation = validateRunConfig(config)
  const running = socket.status === 'running'

  // The active run's output dir, looked up from the run list (single source of truth for
  // `out`). Drives the "Browse dataset" hand-off once a generation finishes.
  const { data: runsData } = useRuns(running ? 2000 : false)
  const activeOut = useMemo(
    () => runsData?.runs.find((r) => r.id === socket.runId)?.out ?? null,
    [runsData, socket.runId],
  )

  const startMut = useMutation({
    mutationFn: () => startRun(cleanConfig(config)),
    onSuccess: (res) => {
      socket.clear()
      socket.connect(res.run_id)
      void queryClient.invalidateQueries({ queryKey: qk.runs })
    },
    onError: (e) => toast.error(e instanceof ApiError ? e.message : 'Failed to start run'),
  })

  const stopMut = useMutation({
    mutationFn: (rid: string) => stopRun(rid),
    onError: (e) => toast.error(e instanceof ApiError ? e.message : 'Failed to stop run'),
  })

  // Refresh the run list once a run reaches a terminal state.
  useEffect(() => {
    if (socket.status === 'done' || socket.status === 'error') {
      void queryClient.invalidateQueries({ queryKey: qk.runs })
    }
  }, [socket.status, queryClient])

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(360px,460px)_1fr]">
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-1">
          <PresetSelect schema={schema} />
          <p className="text-xs text-muted-foreground">
            Recommended defaults are pre-filled — set <code className="font-mono">sources</code> and{' '}
            <code className="font-mono">out</code>, then Generate.
          </p>
        </div>

        <details className="rounded-md border border-border p-2">
          <summary className="cursor-pointer text-sm font-medium">Upload seed images</summary>
          <div className="pt-2">
            <Dropzone
              defaultDest={`${schema.data_root}/seeds/uploaded`}
              onUploaded={(dir) => setField('sources', dir)}
            />
          </div>
        </details>

        <SchemaForm schema={schema} errors={validation.errors} />

        <div className="sticky bottom-0 z-10 flex flex-col gap-2 border-t border-border bg-background/95 py-3 backdrop-blur">
          <div className="flex gap-2">
            <Button
              onClick={() => startMut.mutate()}
              disabled={!validation.ok || running || startMut.isPending}
            >
              {startMut.isPending || running ? <Spinner /> : <Play />} Generate dataset
            </Button>
            <Button
              variant="destructive"
              disabled={!running || !socket.runId}
              onClick={() => socket.runId && stopMut.mutate(socket.runId)}
            >
              <Square /> Stop
            </Button>
          </div>
          {!validation.ok && (
            <ErrorText>Required: {Object.values(validation.errors).join(', ')}</ErrorText>
          )}
        </div>
      </div>

      <div className="flex min-h-[60vh] flex-col gap-3">
        <RunHistory
          activeId={socket.runId}
          poll={running}
          onSelect={(id) => {
            socket.clear()
            socket.connect(id)
          }}
        />
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold">Run log</h2>
          <StatusBadge status={socket.status} />
          {socket.status === 'done' && activeOut && (
            <Button
              variant="outline"
              size="sm"
              className="ml-auto"
              onClick={() => openInBrowser(activeOut)}
            >
              <FolderSearch /> Browse dataset
            </Button>
          )}
        </div>
        <RunLog lines={socket.lines} className="min-h-[50vh] flex-1" />
      </div>
    </div>
  )
}
