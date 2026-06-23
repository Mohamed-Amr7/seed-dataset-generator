import { Upload } from 'lucide-react'
import type * as React from 'react'
import { useRef, useState } from 'react'
import { ApiError } from '@/api/client'
import { uploadSeeds } from '@/api/endpoints'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Spinner } from '@/components/common'
import { toast } from '@/store/toastStore'
import { cn } from '@/lib/utils'

const IMG_RE = /\.(png|jpe?g)$/i

/** Drag-and-drop / pick seed images and POST them to /api/seeds/upload; on success the
 *  caller wires the returned folder into `sources`. */
export function Dropzone({
  defaultDest,
  onUploaded,
}: {
  defaultDest: string
  onUploaded: (dir: string) => void
}) {
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
    <div className="flex flex-col gap-2">
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
      <Input
        value={dest}
        onChange={(e) => setDest(e.target.value)}
        placeholder="destination folder (under the data root)"
        aria-label="Upload destination folder"
      />
      <Button onClick={upload} disabled={busy}>
        {busy ? <Spinner /> : <Upload />} Upload to seeds folder
      </Button>
    </div>
  )
}
