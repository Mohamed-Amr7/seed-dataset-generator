import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { appendBounded, parseWsMessage, useRunSocket } from './useRunSocket'

describe('appendBounded', () => {
  it('appends while under capacity', () => {
    expect(appendBounded(['a'], 'b', 5)).toEqual(['a', 'b'])
  })

  it('drops the oldest line once at capacity', () => {
    expect(appendBounded(['a', 'b', 'c'], 'd', 3)).toEqual(['b', 'c', 'd'])
  })
})

describe('parseWsMessage', () => {
  it('parses a line frame', () => {
    expect(parseWsMessage('{"type":"line","line":"hello"}')).toEqual({
      type: 'line',
      line: 'hello',
    })
  })

  it('parses a done frame', () => {
    expect(parseWsMessage('{"type":"done","returncode":0,"status":"done"}')).toMatchObject({
      type: 'done',
      status: 'done',
    })
  })

  it('returns null on malformed input', () => {
    expect(parseWsMessage('not-json')).toBeNull()
  })
})

// A minimal WebSocket stand-in we can drive synchronously from tests.
class MockWebSocket {
  static last: MockWebSocket | null = null
  onmessage: ((ev: { data: string }) => void) | null = null
  onerror: (() => void) | null = null
  close = vi.fn()
  constructor() {
    MockWebSocket.last = this
  }
}

describe('useRunSocket', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('accumulates lines and reaches a terminal status', () => {
    vi.stubGlobal('WebSocket', MockWebSocket as unknown as typeof WebSocket)
    const { result } = renderHook(() => useRunSocket())

    act(() => result.current.connect('run-1'))
    expect(result.current.status).toBe('running')

    const ws = MockWebSocket.last!
    act(() => ws.onmessage?.({ data: '{"type":"line","line":"step 1"}' }))
    act(() => ws.onmessage?.({ data: '{"type":"done","returncode":0,"status":"done"}' }))

    expect(result.current.status).toBe('done')
    expect(result.current.lines[0]).toBe('step 1')
    expect(result.current.lines.some((l) => l.includes('finished (exit 0)'))).toBe(true)
  })

  it('flips to error on an error frame', () => {
    vi.stubGlobal('WebSocket', MockWebSocket as unknown as typeof WebSocket)
    const { result } = renderHook(() => useRunSocket())
    act(() => result.current.connect('run-2'))
    act(() => MockWebSocket.last!.onmessage?.({ data: '{"type":"error","msg":"no such run"}' }))
    expect(result.current.status).toBe('error')
  })
})
