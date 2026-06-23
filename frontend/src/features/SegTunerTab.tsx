import { FolderOpen, ScanSearch } from 'lucide-react'
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { ApiError } from '@/api/client'
import { segCheck } from '@/api/endpoints'
import type { ConfigValue } from '@/api/types'
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
import { SegReport } from '@/components/SegReport'
import { ImageGrid } from '@/components/ImageGrid'

const METHODS = ['auto', 'rembg', 'otsu', 'grabcut', 'threshold', 'alpha']
const PROVIDERS = ['auto', 'cuda', 'cpu']

const asStr = (v: ConfigValue | undefined) => (typeof v === 'string' ? v : '')
const asNum = (v: ConfigValue | undefined, d: number) => (typeof v === 'number' ? v : d)

export function SegTunerTab() {
  const config = useConfigStore((s) => s.config)
  const openPicker = usePickerStore((s) => s.openPicker)

  const [sources, setSources] = useState(() => asStr(config.sources))
  const [method, setMethod] = useState(() => asStr(config.segment) || 'auto')
  const [thresh, setThresh] = useState(() => String(asNum(config.seg_conf_thresh, 0.55)))
  const [provider, setProvider] = useState(() => asStr(config.rembg_provider) || 'auto')
  const [n, setN] = useState('24')

  const mut = useMutation({
    mutationFn: () =>
      segCheck(
        {
          ...cleanConfig(config),
          sources,
          segment: method,
          seg_conf_thresh: parseFloat(thresh) || 0.55,
          rembg_provider: provider,
        },
        parseInt(n, 10) || 24,
      ),
    onError: (e) => toast.error(e instanceof ApiError ? e.message : 'Segmentation check failed'),
  })

  const runCheck = () => {
    if (!sources.trim()) {
      toast.error('Set a sources folder to check.')
      return
    }
    mut.mutate()
  }

  const result = mut.data

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h2 className="text-base font-semibold">Segmentation tuner</h2>
        <p className="text-sm text-muted-foreground">
          Renders cut-outs on a checkerboard and reports per-source confidence + skip reasons.
          Adjust the method / threshold / provider and re-check before a full run.
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
          <Label htmlFor="seg-thresh">conf-thresh</Label>
          <Input
            id="seg-thresh"
            type="number"
            step="any"
            value={thresh}
            onChange={(e) => setThresh(e.target.value)}
          />
        </div>

        <div className="flex flex-col gap-1">
          <Label htmlFor="seg-provider">rembg provider</Label>
          <Select value={provider} onValueChange={setProvider}>
            <SelectTrigger id="seg-provider">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PROVIDERS.map((p) => (
                <SelectItem key={p} value={p}>
                  {p}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-1">
          <Label htmlFor="seg-n"># seeds</Label>
          <Input id="seg-n" type="number" value={n} onChange={(e) => setN(e.target.value)} />
        </div>

        <Button onClick={runCheck} disabled={mut.isPending}>
          {mut.isPending ? <Spinner /> : <ScanSearch />} Check
        </Button>
      </div>

      {mut.isPending ? (
        <EmptyState>Checking segmentation…</EmptyState>
      ) : result ? (
        <div className="flex flex-col gap-4">
          <SegReport report={result.report} />
          <ImageGrid urls={result.images} emptyText="No cut-outs were produced." />
          {result.log && (
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded-md border border-border bg-background p-3 font-mono text-xs">
              {result.log}
            </pre>
          )}
        </div>
      ) : (
        <EmptyState>Run a check to preview how seeds are segmented.</EmptyState>
      )}
    </div>
  )
}
