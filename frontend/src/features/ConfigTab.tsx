import { FolderOpen, FolderUp, Save } from 'lucide-react'
import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { ApiError } from '@/api/client'
import { loadConfig, saveConfig } from '@/api/endpoints'
import { useSchema } from '@/api/queries'
import { cleanConfig, useConfigStore } from '@/store/configStore'
import { usePickerStore } from '@/store/pickerStore'
import { toast } from '@/store/toastStore'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Spinner } from '@/components/common'

export function ConfigTab() {
  const config = useConfigStore((s) => s.config)
  const setMany = useConfigStore((s) => s.setMany)
  const openPicker = usePickerStore((s) => s.openPicker)
  const { data: schema } = useSchema()
  const [path, setPath] = useState('')
  // Config files live with the project (configs/*.yaml), not the data root - so default the
  // browser there rather than letting it fall back to DATA_ROOT (often /tmp). fs/list climbs
  // to the project root if `configs/` doesn't exist.
  const configDir = schema ? `${schema.project_root}/configs` : null

  const saveMut = useMutation({
    mutationFn: () => saveConfig(path.trim(), cleanConfig(config)),
    onSuccess: (r) => toast.success(`Saved → ${r.saved}`),
    onError: (e) => toast.error(e instanceof ApiError ? e.message : 'Save failed'),
  })

  const loadMut = useMutation({
    mutationFn: () => loadConfig(path.trim()),
    onSuccess: (v) => {
      setMany(v)
      toast.success(`Loaded ${Object.keys(v).length} keys into the Run form.`)
    },
    onError: (e) => toast.error(e instanceof ApiError ? e.message : 'Load failed'),
  })

  const noPath = !path.trim()

  return (
    <div className="flex max-w-2xl flex-col gap-4">
      <div>
        <h2 className="text-base font-semibold">Config save / load</h2>
        <p className="text-sm text-muted-foreground">
          Round-trips with the CLI&apos;s <code className="font-mono">--config</code>. Save writes
          the current Run form; Load fills it from a JSON/YAML config.
        </p>
      </div>

      <div className="flex flex-col gap-1">
        <Label htmlFor="cfg-path">Path</Label>
        <div className="flex gap-2">
          <Input
            id="cfg-path"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="configs/run_config.yaml"
          />
          <Button
            type="button"
            variant="outline"
            size="icon"
            aria-label="Browse for config file"
            onClick={() =>
              openPicker({
                kind: 'file',
                fieldName: 'config path',
                initialPath: path || configDir,
                onSelect: setPath,
              })
            }
          >
            <FolderOpen />
          </Button>
        </div>
      </div>

      <div className="flex gap-2">
        <Button onClick={() => saveMut.mutate()} disabled={noPath || saveMut.isPending}>
          {saveMut.isPending ? <Spinner /> : <Save />} Save current form
        </Button>
        <Button
          variant="secondary"
          onClick={() => loadMut.mutate()}
          disabled={noPath || loadMut.isPending}
        >
          {loadMut.isPending ? <Spinner /> : <FolderUp />} Load into form
        </Button>
      </div>
    </div>
  )
}
