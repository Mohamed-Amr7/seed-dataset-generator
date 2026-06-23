import { useCallback, useEffect, useRef, useState } from 'react'
import { wsUrl } from '@/api/endpoints'
import type { RunStatus, WsMessage } from '@/api/types'

export const LOG_MAX_LINES = 5000

/** Append a line, keeping at most `max` lines (drops oldest). Pure for easy testing. */
export function appendBounded(prev: string[], line: string, max: number): string[] {
  const next = prev.length >= max ? prev.slice(prev.length - max + 1) : prev.slice()
  next.push(line)
  return next
}

/** Parse a WS frame into a typed message, or null if malformed. */
export function parseWsMessage(data: string): WsMessage | null {
  try {
    const m: unknown = JSON.parse(data)
    if (m && typeof m === 'object' && 'type' in m) return m as WsMessage
    return null
  } catch {
    return null
  }
}

export type RunPhase = RunStatus | 'idle'

export interface RunSocket {
  lines: string[]
  status: RunPhase
  runId: string | null
  connect: (rid: string) => void
  clear: () => void
}

/** Subscribe to /ws/runs/{rid}; streams a bounded line buffer + terminal status. */
export function useRunSocket(): RunSocket {
  const [lines, setLines] = useState<string[]>([])
  const [status, setStatus] = useState<RunPhase>('idle')
  const [runId, setRunId] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  const clear = useCallback(() => setLines([]), [])

  const connect = useCallback((rid: string) => {
    wsRef.current?.close()
    setRunId(rid)
    setStatus('running')
    const ws = new WebSocket(wsUrl(rid))
    wsRef.current = ws
    ws.onmessage = (ev: MessageEvent) => {
      const m = parseWsMessage(typeof ev.data === 'string' ? ev.data : '')
      if (!m) return
      if (m.type === 'line') {
        setLines((prev) => appendBounded(prev, m.line, LOG_MAX_LINES))
      } else if (m.type === 'done') {
        setStatus(m.status)
        setLines((prev) =>
          appendBounded(prev, `\n--- finished (exit ${m.returncode}) ---`, LOG_MAX_LINES),
        )
      } else {
        setStatus('error')
        setLines((prev) => appendBounded(prev, `\n--- error: ${m.msg} ---`, LOG_MAX_LINES))
      }
    }
    ws.onerror = () => setStatus('error')
  }, [])

  useEffect(() => () => wsRef.current?.close(), [])

  return { lines, status, runId, connect, clear }
}
