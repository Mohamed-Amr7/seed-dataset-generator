// Thin fetch wrapper that normalizes FastAPI errors into a single ApiError with a
// human-readable message. FastAPI returns `{detail: string}` for raised HTTPExceptions
// and `{detail: [{loc, msg}, ...]}` for 422 validation errors — both are flattened here
// so the UI never has to dump raw JSON at the user.

export class ApiError extends Error {
  readonly status: number
  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

interface PydanticError {
  loc?: unknown[]
  msg?: unknown
}

/** Pull a readable message out of a FastAPI error body, or null if it isn't shaped like one. */
export function extractDetail(body: unknown): string | null {
  if (!body || typeof body !== 'object' || !('detail' in body)) return null
  const detail = (body as { detail: unknown }).detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const parts = detail.map((entry) => {
      if (entry && typeof entry === 'object' && 'msg' in entry) {
        const e = entry as PydanticError
        const loc = Array.isArray(e.loc) ? e.loc.join('.') : ''
        return loc ? `${loc}: ${String(e.msg)}` : String(e.msg)
      }
      return JSON.stringify(entry)
    })
    return parts.join('; ')
  }
  return JSON.stringify(detail)
}

export async function apiFetch<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch {
    throw new ApiError(`Network error: cannot reach ${url}`, 0)
  }

  const contentType = res.headers.get('content-type') ?? ''
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`.trim()
    if (contentType.includes('application/json')) {
      const body = await res.json().catch(() => null)
      const detail = extractDetail(body)
      if (detail) message = detail
    } else {
      const text = await res.text().catch(() => '')
      if (text) message = text
    }
    throw new ApiError(message, res.status)
  }

  if (contentType.includes('application/json')) {
    return (await res.json()) as T
  }
  return (await res.text()) as unknown as T
}
