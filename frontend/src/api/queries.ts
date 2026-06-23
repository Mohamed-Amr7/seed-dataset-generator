import { useQuery } from '@tanstack/react-query'
import { getSchema, listRuns } from './endpoints'

export const qk = {
  schema: ['schema'] as const,
  runs: ['runs'] as const,
}

/** The form descriptor rarely changes within a session, so cache it for the session. */
export function useSchema() {
  return useQuery({ queryKey: qk.schema, queryFn: getSchema, staleTime: Infinity })
}

/** Run list; pass a refetchInterval to poll while a generation is in flight. */
export function useRuns(refetchInterval: number | false = false) {
  return useQuery({ queryKey: qk.runs, queryFn: listRuns, refetchInterval })
}
