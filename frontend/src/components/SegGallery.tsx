import { useMemo, useState } from 'react'
import type { SegSeed } from '@/api/types'
import { Badge, type BadgeVariant } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

type Filter = 'all' | 'kept' | 'skipped'

function baseName(source: string | null, url: string): string {
  const raw = source ?? url.split('/').pop() ?? ''
  try {
    return decodeURIComponent(raw)
  } catch {
    return raw
  }
}

/** Confidence badge: kept→green (or amber if under threshold), skipped→red, unknown→grey. */
function confBadge(s: SegSeed, thresh: number): { variant: BadgeVariant; label: string } {
  if (s.confidence == null) return { variant: 'secondary', label: 'conf n/a' }
  const c = s.confidence.toFixed(2)
  if (s.status === 'skipped') return { variant: 'destructive', label: `conf ${c}` }
  return { variant: s.confidence >= thresh ? 'success' : 'warning', label: `conf ${c}` }
}

/**
 * Per-seed segmentation preview with a kept/skipped filter. Kept cut-outs render on a
 * checkerboard; skipped sources show their original with the skip reason. Each card shows
 * the confidence, colour-coded against the tuner's confidence threshold.
 */
export function SegGallery({ seeds, thresh }: { seeds: SegSeed[]; thresh: number }) {
  const [filter, setFilter] = useState<Filter>('all')

  const counts = useMemo(
    () => ({
      all: seeds.length,
      kept: seeds.filter((s) => s.status === 'kept').length,
      skipped: seeds.filter((s) => s.status === 'skipped').length,
    }),
    [seeds],
  )

  const shown = useMemo(
    () => (filter === 'all' ? seeds : seeds.filter((s) => s.status === filter)),
    [seeds, filter],
  )

  if (!seeds.length) {
    return <p className="text-sm text-muted-foreground">No cut-outs were produced.</p>
  }

  const tabs: { key: Filter; label: string }[] = [
    { key: 'all', label: `All ${counts.all}` },
    { key: 'kept', label: `Kept ${counts.kept}` },
    { key: 'skipped', label: `Skipped ${counts.skipped}` },
  ]

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-1" role="tablist" aria-label="Filter cut-outs">
        {tabs.map((t) => (
          <button
            key={t.key}
            type="button"
            role="tab"
            aria-selected={filter === t.key}
            onClick={() => setFilter(t.key)}
            className={cn(
              'rounded-md border px-3 py-1 text-xs font-medium transition-colors',
              filter === t.key
                ? 'border-primary bg-primary/10 text-primary'
                : 'border-border text-muted-foreground hover:bg-accent',
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {!shown.length ? (
        <p className="text-sm text-muted-foreground">Nothing matches this filter.</p>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-3">
          {shown.map((s) => {
            const name = baseName(s.source, s.url)
            const badge = confBadge(s, thresh)
            return (
              <figure
                key={s.url}
                className={cn(
                  'flex flex-col overflow-hidden rounded-md border',
                  s.status === 'skipped' ? 'border-destructive/40' : 'border-border',
                )}
              >
                <img
                  src={s.url}
                  alt={name}
                  loading="lazy"
                  className="h-32 w-full bg-black/30 object-contain"
                />
                <figcaption className="flex flex-col gap-1 p-1.5">
                  <div className="flex items-center justify-between gap-1">
                    <Badge variant={badge.variant}>{badge.label}</Badge>
                    <span className="truncate text-[11px] text-muted-foreground" title={name}>
                      {s.status === 'skipped' ? (s.reason ?? 'skipped') : (s.method ?? '')}
                    </span>
                  </div>
                  <span className="truncate text-[11px] text-muted-foreground" title={name}>
                    {name}
                  </span>
                </figcaption>
              </figure>
            )
          })}
        </div>
      )}
    </div>
  )
}
