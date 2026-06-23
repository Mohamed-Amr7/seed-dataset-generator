import { z } from 'zod'
import type { ConfigDict } from '@/api/types'

// The backend (CLI) is the source of truth for full config validation; the UI only
// enforces the two fields a run cannot start without, so we can fail fast in the form
// instead of round-tripping to a 400.
const runConfigSchema = z.object({
  sources: z.string().min(1, 'sources is required'),
  out: z.string().min(1, 'out is required'),
})

export interface ValidationResult {
  ok: boolean
  errors: Record<string, string>
}

export function validateRunConfig(config: ConfigDict): ValidationResult {
  const candidate = {
    sources: typeof config.sources === 'string' ? config.sources : '',
    out: typeof config.out === 'string' ? config.out : '',
  }
  const result = runConfigSchema.safeParse(candidate)
  if (result.success) return { ok: true, errors: {} }
  const errors: Record<string, string> = {}
  for (const issue of result.error.issues) {
    const key = String(issue.path[0])
    if (!errors[key]) errors[key] = issue.message
  }
  return { ok: false, errors }
}
