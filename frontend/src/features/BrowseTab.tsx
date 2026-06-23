import { Download, FolderOpen, Search } from 'lucide-react'
import { useCallback, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { ApiError } from '@/api/client'
import { datasetDownloadUrl, datasetInfo, listDatasets } from '@/api/endpoints'
import type { Split } from '@/api/types'
import { usePickerStore } from '@/store/pickerStore'
import { toast } from '@/store/toastStore'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { EmptyState, Spinner } from '@/components/common'
import { DatasetViewer } from '@/components/DatasetViewer'
import { ImageGrid } from '@/components/ImageGrid'
import { SegReport } from '@/components/SegReport'

export function BrowseTab() {
  const openPicker = usePickerStore((s) => s.openPicker)
  const [root, setRoot] = useState('')
  const [selected, setSelected] = useState('')
  const [split, setSplit] = useState<Split>('train')
  const [idx, setIdx] = useState(0)
  const [total, setTotal] = useState(0)
  const [name, setName] = useState('')

  // Selecting a dataset or split resets paging to the first image.
  const selectDataset = (d: string) => {
    setSelected(d)
    setIdx(0)
  }
  const selectSplit = (s: Split) => {
    setSplit(s)
    setIdx(0)
  }

  const scanMut = useMutation({
    mutationFn: () => listDatasets(root.trim() || undefined),
    onSuccess: (r) => {
      if (r.datasets.length) selectDataset(r.datasets[0])
      else toast.info('No datasets found under that root.')
    },
    onError: (e) => toast.error(e instanceof ApiError ? e.message : 'Scan failed'),
  })

  const info = useQuery({
    queryKey: ['datasetInfo', selected],
    queryFn: () => datasetInfo(selected),
    enabled: !!selected,
  })

  const onMeta = useCallback((t: number, nm: string) => {
    setTotal(t)
    setName(nm)
  }, [])

  const datasets = scanMut.data?.datasets ?? []

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-base font-semibold">Dataset browser</h2>

      <div className="grid items-end gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <div className="flex flex-col gap-1">
          <Label htmlFor="ds-root">Datasets root</Label>
          <div className="flex gap-2">
            <Input
              id="ds-root"
              value={root}
              onChange={(e) => setRoot(e.target.value)}
              placeholder="(data root)"
            />
            <Button
              type="button"
              variant="outline"
              size="icon"
              aria-label="Browse for datasets root"
              onClick={() =>
                openPicker({
                  kind: 'dir',
                  fieldName: 'datasets root',
                  initialPath: root || null,
                  onSelect: setRoot,
                })
              }
            >
              <FolderOpen />
            </Button>
          </div>
        </div>

        <Button onClick={() => scanMut.mutate()} disabled={scanMut.isPending}>
          {scanMut.isPending ? <Spinner /> : <Search />} Scan
        </Button>

        <div className="flex flex-col gap-1">
          <Label htmlFor="ds-select">Dataset</Label>
          <Select value={selected} onValueChange={selectDataset}>
            <SelectTrigger id="ds-select">
              <SelectValue placeholder="Scan to discover…" />
            </SelectTrigger>
            <SelectContent>
              {datasets.map((d) => (
                <SelectItem key={d} value={d}>
                  {d}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-1">
          <Label htmlFor="ds-split">Split</Label>
          <Select value={split} onValueChange={(v) => selectSplit(v as Split)}>
            <SelectTrigger id="ds-split">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="train">train</SelectItem>
              <SelectItem value="val">val</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {!selected ? (
        <EmptyState>Scan a root folder, then pick a dataset to inspect.</EmptyState>
      ) : info.isLoading ? (
        <EmptyState>Loading dataset…</EmptyState>
      ) : info.isError || !info.data ? (
        <EmptyState>Could not load dataset info.</EmptyState>
      ) : (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="font-mono text-muted-foreground">{info.data.path}</span>
            <Button asChild variant="outline" size="sm">
              <a href={datasetDownloadUrl(selected)} download>
                <Download /> Download .zip
              </a>
            </Button>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span>splits:</span>
            {Object.entries(info.data.splits).map(([k, v]) => (
              <Badge key={k} variant="secondary">
                {k}={v}
              </Badge>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-1 text-sm">
            <span>classes ({info.data.classes.length}):</span>
            {info.data.classes.slice(0, 30).map((c) => (
              <Badge key={c}>{c}</Badge>
            ))}
          </div>

          <DatasetViewer
            path={selected}
            split={split}
            idx={idx}
            total={total}
            name={name}
            onMeta={onMeta}
            onPrev={() => setIdx((i) => Math.max(0, i - 1))}
            onNext={() => setIdx((i) => (total > 0 ? Math.min(total - 1, i + 1) : i + 1))}
          />

          {info.data.qa.length > 0 && (
            <div className="flex flex-col gap-2">
              <h3 className="text-sm font-semibold">QA charts</h3>
              <ImageGrid urls={info.data.qa} />
            </div>
          )}

          {info.data.seg_report && (
            <div className="flex flex-col gap-2">
              <h3 className="text-sm font-semibold">Segmentation report</h3>
              <SegReport report={info.data.seg_report} compact />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
