// Typed function per backend route. URLs are same-origin: in dev the Vite proxy forwards
// /api and /ws to the FastAPI backend on :8000; in prod FastAPI serves both the SPA and
// the API from one origin.
import { apiFetch } from './client'
import type {
  ConfigDict,
  ConfigSaveResponse,
  DatasetInfo,
  DatasetsList,
  FormSchema,
  FsList,
  RunDetail,
  RunsList,
  SegCheckResponse,
  Split,
  StartRunResponse,
  UploadResponse,
} from './types'

const JSON_HEADERS = { 'Content-Type': 'application/json' }
const enc = encodeURIComponent

export const getSchema = () => apiFetch<FormSchema>('/api/schema')

export const loadConfig = (path: string) =>
  apiFetch<ConfigDict>(`/api/config/load?path=${enc(path)}`)

export const saveConfig = (path: string, config: ConfigDict) =>
  apiFetch<ConfigSaveResponse>('/api/config/save', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ path, config }),
  })

export const startRun = (config: ConfigDict) =>
  apiFetch<StartRunResponse>('/api/runs', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ config }),
  })

export const listRuns = () => apiFetch<RunsList>('/api/runs')

export const getRun = (rid: string) => apiFetch<RunDetail>(`/api/runs/${enc(rid)}`)

export const stopRun = (rid: string) =>
  apiFetch<{ stopped: string; already_exited?: boolean }>(`/api/runs/${enc(rid)}/stop`, {
    method: 'POST',
  })

export const segCheck = (config: ConfigDict, n: number, out?: string) =>
  apiFetch<SegCheckResponse>('/api/seg/check', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ config, n, out }),
  })

export const listDatasets = (root?: string) =>
  apiFetch<DatasetsList>('/api/datasets' + (root ? `?root=${enc(root)}` : ''))

export const datasetInfo = (path: string) =>
  apiFetch<DatasetInfo>(`/api/datasets/info?path=${enc(path)}`)

export const fsList = (path: string | null, kind: 'dir' | 'file') =>
  apiFetch<FsList>(`/api/fs/list?kind=${kind}` + (path ? `&path=${enc(path)}` : ''))

export const uploadSeeds = (dest: string, files: File[]) => {
  const form = new FormData()
  form.append('dest', dest)
  for (const f of files) form.append('files', f)
  return apiFetch<UploadResponse>('/api/seeds/upload', { method: 'POST', body: form })
}

// URL builders for endpoints consumed directly by <img> / browser download, not fetch().
export const datasetImageUrl = (path: string, split: Split, idx: number) =>
  `/api/datasets/image?path=${enc(path)}&split=${split}&idx=${idx}`

export const datasetDownloadUrl = (path: string, split?: Split) =>
  `/api/datasets/download?path=${enc(path)}` + (split ? `&split=${split}` : '')

export const fileUrl = (path: string) => `/api/file?path=${enc(path)}`

export const wsUrl = (rid: string) => {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}/ws/runs/${enc(rid)}`
}
