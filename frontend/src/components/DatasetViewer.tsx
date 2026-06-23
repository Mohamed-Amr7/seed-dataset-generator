import { ChevronLeft, ChevronRight } from 'lucide-react'
import type * as React from 'react'
import { useEffect, useState } from 'react'
import { datasetImageUrl } from '@/api/endpoints'
import type { Split } from '@/api/types'
import { Button } from '@/components/ui/button'
import { Spinner } from '@/components/common'

interface Props {
  path: string
  split: Split
  idx: number
  total: number
  name: string
  onMeta: (total: number, name: string) => void
  onPrev: () => void
  onNext: () => void
}

/** Annotated-image viewer. Fetches the PNG (to read X-Total / X-Name headers), shows it via
 *  an object URL, and revokes the previous URL on every change/unmount (the leak the old UI
 *  had). Arrow keys page through the split. */
export function DatasetViewer({ path, split, idx, total, name, onMeta, onPrev, onNext }: Props) {
  const [src, setSrc] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!path) return
    let objectUrl: string | null = null
    let cancelled = false
    const ac = new AbortController()
    // Fetch-on-change effect (network + blob lifecycle is exactly what effects are for).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true)
    setError(null)
    fetch(datasetImageUrl(path, split, idx), { signal: ac.signal })
      .then(async (r) => {
        if (!r.ok) throw new Error('no image')
        const t = parseInt(r.headers.get('X-Total') ?? '0', 10)
        const nm = r.headers.get('X-Name') ?? ''
        const blob = await r.blob()
        if (cancelled) return
        objectUrl = URL.createObjectURL(blob)
        setSrc(objectUrl)
        onMeta(t, nm)
      })
      .catch((e: unknown) => {
        if (!cancelled && (e as Error).name !== 'AbortError') setError('Could not load image')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
      ac.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [path, split, idx, onMeta])

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') onPrev()
    else if (e.key === 'ArrowRight') onNext()
  }

  return (
    <div className="flex flex-col items-center gap-2">
      {/* Arrow-key paging on an aria-labelled group; the Prev/Next buttons remain the
          primary, fully accessible control. Block-disable survives Prettier re-wrapping. */}
      {/* eslint-disable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */}
      <div
        role="group"
        aria-label="Dataset image navigation"
        tabIndex={0}
        onKeyDown={onKeyDown}
        className="flex w-full items-center justify-center gap-3"
      >
        <Button
          variant="outline"
          size="icon"
          aria-label="Previous image"
          onClick={onPrev}
          disabled={idx <= 0}
        >
          <ChevronLeft />
        </Button>
        <div className="flex min-h-[200px] flex-1 items-center justify-center rounded-md border border-border bg-black/40">
          {loading ? (
            <Spinner className="h-6 w-6" />
          ) : error ? (
            <span className="p-4 text-sm text-muted-foreground">{error}</span>
          ) : src ? (
            <img
              src={src}
              alt={name || 'dataset sample with annotations'}
              className="max-h-[64vh] max-w-full object-contain"
            />
          ) : (
            <span className="p-4 text-sm text-muted-foreground">No image</span>
          )}
        </div>
        <Button
          variant="outline"
          size="icon"
          aria-label="Next image"
          onClick={onNext}
          disabled={total > 0 && idx >= total - 1}
        >
          <ChevronRight />
        </Button>
      </div>
      {/* eslint-enable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */}
      {name && (
        <p className="text-xs text-muted-foreground">
          {name} ({idx + 1} / {total})
        </p>
      )}
    </div>
  )
}
