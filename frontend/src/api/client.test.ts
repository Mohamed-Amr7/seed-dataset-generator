import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, apiFetch, extractDetail } from './client'

function jsonResponse(status: number, body: unknown) {
  return {
    ok: status < 400,
    status,
    statusText: 'Status',
    headers: { get: () => 'application/json' },
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  } as unknown as Response
}

describe('extractDetail', () => {
  it('returns a string detail unchanged', () => {
    expect(extractDetail({ detail: 'sources is required' })).toBe('sources is required')
  })

  it('flattens a pydantic 422 detail array to loc: msg', () => {
    expect(extractDetail({ detail: [{ loc: ['body', 'config'], msg: 'must be a dict' }] })).toBe(
      'body.config: must be a dict',
    )
  })

  it('returns null when the body is not a FastAPI error', () => {
    expect(extractDetail({ foo: 'bar' })).toBeNull()
  })
})

describe('apiFetch', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('parses JSON on success', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse(200, { run_id: 'run-1' })))
    await expect(apiFetch('/api/runs')).resolves.toEqual({ run_id: 'run-1' })
  })

  it('throws an ApiError carrying the backend detail on 400', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse(400, { detail: 'missing sources' })),
    )
    await expect(apiFetch('/api/runs')).rejects.toMatchObject({
      name: 'ApiError',
      message: 'missing sources',
      status: 400,
    })
  })

  it('throws an ApiError on network failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('boom')))
    const err = await apiFetch('/api/schema').catch((e: unknown) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect((err as ApiError).status).toBe(0)
  })
})
