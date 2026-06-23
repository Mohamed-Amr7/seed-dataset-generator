import type { SegReportData } from '@/api/types'
import { Badge } from '@/components/ui/badge'

export function SegReport({
  report,
  compact = false,
}: {
  report: SegReportData | null
  compact?: boolean
}) {
  if (!report) {
    return <p className="text-sm text-muted-foreground">No segmentation report produced.</p>
  }
  const s = report.summary ?? {}
  return (
    <div className="flex flex-col gap-2 text-sm">
      {!compact && (
        <p className="text-muted-foreground">
          scanned {s.scanned ?? '?'} · kept {s.kept ?? '?'} · skipped {s.skipped ?? '?'} · cache
          hits {s.cache_hits ?? 0}
        </p>
      )}
      {s.method_counts && Object.keys(s.method_counts).length > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          <span className="font-medium">kept by method:</span>
          {Object.entries(s.method_counts).map(([k, v]) => (
            <Badge key={k} variant="secondary">
              {k}: {v}
            </Badge>
          ))}
        </div>
      )}
      {s.skip_counts && Object.keys(s.skip_counts).length > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          <span className="font-medium">skipped by reason:</span>
          {Object.entries(s.skip_counts).map(([k, v]) => (
            <Badge key={k} variant="warning">
              {k}: {v}
            </Badge>
          ))}
        </div>
      )}
      {!compact && s.confidence && (
        <p className="text-muted-foreground">
          confidence min/median/mean: {s.confidence.min} / {s.confidence.median} /{' '}
          {s.confidence.mean}
        </p>
      )}
      {!compact && report.skipped && report.skipped.length > 0 && (
        <div className="overflow-auto rounded-md border border-border">
          <table className="w-full border-collapse text-xs">
            <thead>
              <tr className="text-left text-muted-foreground">
                <th className="p-2">source</th>
                <th className="p-2">reason</th>
                <th className="p-2">conf</th>
              </tr>
            </thead>
            <tbody>
              {report.skipped.slice(0, 40).map((row, i) => (
                <tr key={`${row.source}-${i}`} className="border-t border-border">
                  <td className="p-2 font-mono">{row.source}</td>
                  <td className="p-2">{row.reason}</td>
                  <td className="p-2">{row.confidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
