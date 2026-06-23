import { FolderOpen, FolderSearch, Upload } from 'lucide-react'
import type * as React from 'react'
import { useRef, useState } from 'react'
import { ApiError } from '@/api/client'
import { uploadSeeds } from '@/api/endpoints'
import { usePickerStore } from '@/store/pickerStore'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Spinner } from '@/components/common'
import { toast } from '@/store/toastStore'
import { cn } from '@/lib/utils'

const IMG_RE = /\.(png|jpe?g)$/i

/**
 * Two ways to point `sources` at seeds, since the backend works on server-side paths:
 *   1. Upload from your machine — native file picker + drag-drop -> POST /api/seeds/upload
 *      into a server folder you choose (typed or Browsed).
 *   2. Use an existing server folder — open the sandboxed picker and set `sources` directly,
 *      no upload, for seeds already on the server.
 */
export function Dropzone({
  defaultDest,
  onUploaded,
}: {
  defaultDest: string
  onUploaded: (dir: string) => void
}) {
  const openPicker = usePickerStore((s) => s.openPicker)
  const [dest, setDest] = useState(defaultDest)
  const [files, setFiles] = useState<File[]>([])
  const [busy, setBusy] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const addFiles = (list: FileList | null) => {
    if (!list) return
    setFiles(Array.from(list).filter((f) => IMG_RE.test(f.name)))
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    addFiles(e.dataTransfer.files)
  }

  const upload = async () => {
    if (!dest.trim() || !files.length) {
      toast.error('Pick a destination folder and at least one image.')
      return
    }
    setBusy(true)
    try {
      const r = await uploadSeeds(dest.trim(), files)
      toast.success(`Uploaded ${r.count} image(s) → ${r.saved_dir}`)
      onUploaded(r.saved_dir)
      setFiles([])
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : 'Upload failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Option 1: upload from this machine */}
      <div
        onDrop={onDrop}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        className={cn(
          'flex flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed p-4 text-center text-sm',
          dragOver ? 'border-primary bg-accent' : 'border-border',
        )}
      >
        <Upload className="h-5 w-5 text-muted-foreground" />
        <p className="text-muted-foreground">Drop seed images here, or</p>
        <Button type="button" variant="outline" size="sm" onClick={() => inputRef.current?.click()}>
          Choose files…
        </Button>
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg"
          multiple
          hidden
          onChange={(e) => addFiles(e.target.files)}
        />
        {files.length > 0 && (
          <p className="text-xs text-muted-foreground">{files.length} file(s) selected</p>
        )}
      </div>

      <div className="flex flex-col gap-1">
        <Label htmlFor="upload-dest">Destination folder (on the server)</Label>
        <div className="flex gap-2">
          <Input
            id="upload-dest"
            value={dest}
            onChange={(e) => setDest(e.target.value)}
            placeholder="destination folder (under the data root)"
          />
          <Button
            type="button"
            variant="outline"
            size="icon"
            aria-label="Browse for destination folder"
            onClick={() =>
              openPicker({
                kind: 'dir',
                fieldName: 'upload destination',
                initialPath: dest || null,
                onSelect: setDest,
              })
            }
          >
            <FolderOpen />
          </Button>
        </div>
      </div>

      <Button onClick={upload} disabled={busy}>
        {busy ? <Spinner /> : <Upload />} Upload to seeds folder
      </Button>

      {/* Option 2: skip the upload entirely and point sources at an existing server folder */}
      <div className="flex items-center gap-2 pt-1">
        <span className="h-px flex-1 bg-border" />
        <span className="text-xs text-muted-foreground">or</span>
        <span className="h-px flex-1 bg-border" />
      </div>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={() =>
          openPicker({
            kind: 'dir',
            fieldName: 'sources',
            initialPath: dest || null,
            onSelect: onUploaded,
          })
        }
      >
        <FolderSearch /> Use an existing server folder as sources
      </Button>
    </div>
  )
}
