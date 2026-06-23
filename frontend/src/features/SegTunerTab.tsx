import { FolderOpen, ScanSearch, Square } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { ApiError } from '@/api/client'
import { getSegResult, segCheck, stopRun } from '@/api/endpoints'
import { qk, useSchema } from '@/api/queries'
import type { ConfigDict, ConfigValue, SchemaField } from '@/api/types'
import { useRunSocket } from '@/hooks/useRunSocket'
import { TUNER_FIELDS } from '@/lib/segFields'
import { cleanConfig, useConfigStore } from '@/store/configStore'
import { usePickerStore } from '@/store/pickerStore'
import { toast } from '@/store/toastStore'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Spinner, EmptyState } from '@/components/common'
import { FieldRow } from '@/components/FieldRow'
import { RunLog } from '@/components/RunLog'
import { StatusBadge } from '@/components/StatusBadge'
import { SegReport } from '@/components/SegReport'
import { SegGallery } from '@/components/SegGallery'

const METHODS = ['auto', 'rembg', 'otsu', 'grabcut', 'threshold', 'alpha', 'sam']

const asStr = (v: ConfigValue | undefined) => (typeof v === 'string' ? v : '')
const asNum = (v: ConfigValue | undefined, d: number) => (typeof v === 'number' ? v : d)

export function SegTunerTab() {
  const config = useConfigStore((s) => s.config)
  const openPicker = usePickerStore((s) => s.openPicker)
  const { data: schema } = useSchema()
  const socket = useRunSocket() // private to the tuner (separate from the Run tab's socket)

  const [sources, setSources] = useState(() => asStr(config.sources))
  const [method, setMethod] = useState(() => asStr(config.segment) || 'auto')
  const [n, setN] = useState('24')
  const [params, setParams] = useState<ConfigDict>({})
  const [out, setOut] = useState<string | null>(null)

  // Flatten the schema into name -> field so per-method knobs render with their real
  // widget / help / default. Values are kept LOCAL (params); they never touch the store.
  const fieldMap = useMemo(() => {
    const m: Record<string, SchemaField> = {}
    for (const g of schema?.groups ?? []) for (const f of g.fields) m[f.name] = f
    return m
  }, [schema])

  // Seed local params from the current config (falling back to schema defaults) once.
  const seeded = useRef(false)
  useEffect(() => {
    if (!schema || seeded.current) return
    const init: ConfigDict = {}
    for (const name of new Set(Object.values(TUNER_FIELDS).flat())) {
      init[name] = (config[name] ?? fieldMap[name]?.default ?? null) as ConfigValue
    }
    setParams(init)
    seeded.current = true
  }, [schema, fieldMap, config])

  const methodFields = (TUNER_FIELDS[method] ?? []).map((nm) => fieldMap[nm]).filter(Boolean)

  const thresh =
    typeof params.seg_conf_thresh === 'number'
      ? params.seg_conf_thresh
      : asNum(config.seg_conf_thresh, 0.55)

  const buildPayload = (): ConfigDict => {
    const mp: ConfigDict = {}
    for (const name of TUNER_FIELDS[method] ?? []) {
      if (params[name] != null) mp[name] = params[name]
    }
    return { ...cleanConfig(config), sources, segment: method, ...cleanConfig(mp) }
  }

  const checkMut = useMutation({
    mutationFn: () => segCheck(buildPayload(), Math.max(1, parseInt(n, 10) || 24)),
    onSuccess: (res) => {
      setOut(res.out)
      socket.clear()
      socket.connect(res.run_id)
    },
    onError: (e) => toast.error(e instanceof ApiError ? e.message : 'Segmentation check failed'),
  })

  const stopMut = useMutation({
    mutationFn: (rid: string) => stopRun(rid),
    onError: (e) => toast.error(e instanceof ApiError ? e.message : 'Failed to stop check'),
  })

  // Fetch the report + per-seed manifest once the check reaches a terminal state. Keyed by
  // run id so each re-check refetches even though the output dir is reused.
  const terminal = socket.status === 'done' || socket.status === 'error'
  const result = useQuery({
    queryKey: qk.segResult(socket.runId ?? 'none'),
    queryFn: () => getSegResult(out as string),
    enabled: !!out && !!socket.runId && terminal,
  })

  const runCheck = () => {
    if (!sources.trim()) {
      toast.error('Set a sources folder to check.')
      return
    }
    checkMut.mutate()
  }

  const running = socket.status === 'running'
  const pending = running || checkMut.isPending

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-base font-semibold">Segmentation tuner</h2>
        <p className="text-sm text-muted-foreground">
          Segments a sample of N sources and renders each cut-out on a checkerboard, with
          per-seed confidence and skip reasons. Tune the method &amp; its settings, then re-check
          before a full run. Stop anytime; results stream live.
        </p>
      </div>

      <div className="grid items-end gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <div className="flex flex-col gap-1">
          <Label htmlFor="seg-sources">Sources</Label>
          <div className="flex gap-2">
            <Input
              id="seg-sources"
              value={sources}
              onChange={(e) => setSources(e.target.value)}
              placeholder="seeds/Test/images"
            />
            <Button
              type="button"
              variant="outline"
              size="icon"
              aria-label="Browse for sources folder"
              onClick={() =>
                openPicker({
                  kind: 'dir',
                  fieldName: 'sources',
                  initialPath: sources || null,
                  onSelect: setSources,
                })
              }
            >
              <FolderOpen />
            </Button>
          </div>
        </div>

        <div className="flex flex-col gap-1">
          <Label htmlFor="seg-method">Method</Label>
          <Select value={method} onValueChange={setMethod}>
            <SelectTrigger id="seg-method">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {METHODS.map((m) => (
                <SelectItem key={m} value={m}>
                  {m}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-1">
          <Label htmlFor="seg-n"># seeds</Label>
          <Input
            id="seg-n"
            type="number"
            min={1}
            value={n}
            onChange={(e) => setN(e.target.value)}
          />
          <span className="text-[11px] leading-tight text-muted-foreground">
            samples this many sources (round-robin across classes); counts reflect exactly these
          </span>
        </div>
      </div>

      <details className="rounded-lg border border-border p-3" open>
        <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {method} settings
        </summary>
        <div className="mt-2 flex flex-col divide-y divide-border">
          {methodFields.length ? (
            methodFields.map((f) => (
              <FieldRow
                key={f.name}
                field={f}
                value={(params[f.name] ?? f.default) as ConfigValue}
                onChange={(v) => setParams((p) => ({ ...p, [f.name]: v }))}
              />
            ))
          ) : (
            <p className="py-2 text-sm text-muted-foreground">
              No tunable settings for this method.
            </p>
          )}
        </div>
      </details>

      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={runCheck} disabled={pending}>
          {pending ? <Spinner /> : <ScanSearch />} Check
        </Button>
        <Button
          variant="destructive"
          disabled={!running || !socket.runId}
          onClick={() => socket.runId && stopMut.mutate(socket.runId)}
        >
          <Square /> Stop
        </Button>
        <StatusBadge status={socket.status} />
      </div>

      {socket.lines.length > 0 && <RunLog lines={socket.lines} className="max-h-48" />}

      {running ? (
        <EmptyState>Checking segmentation… (live log above)</EmptyState>
      ) : result.isFetching ? (
        <EmptyState>Loading results…</EmptyState>
      ) : result.data ? (
        <div className="flex flex-col gap-4">
          <SegReport report={result.data.report} />
          <SegGallery seeds={result.data.seeds} thresh={thresh} />
        </div>
      ) : socket.status === 'idle' ? (
        <EmptyState>Run a check to preview how seeds are segmented.</EmptyState>
      ) : null}
    </div>
  )
}
