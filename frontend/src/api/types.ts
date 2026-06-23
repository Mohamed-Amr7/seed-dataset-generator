// Wire types for the MultiSeedGen FastAPI backend. These mirror the Pydantic models /
// route return shapes in multiseedgen/web/{app,schemas,services}.py. The `config` dict is
// intentionally loose (string-keyed scalars) — the CLI is the source of truth for
// validating individual keys, exactly as the backend documents.

export type Widget = 'bool' | 'choice' | 'int' | 'float' | 'text'
export type PathKind = 'dir' | 'file' | null

export interface SchemaField {
  name: string
  default: unknown
  widget: Widget
  choices: string[] | null
  help: string
  flag: string
  path_kind: PathKind
}

export interface SchemaGroup {
  title: string
  fields: SchemaField[]
}

export interface FormSchema {
  groups: SchemaGroup[]
  recommended: ConfigDict
  presets: Record<string, ConfigDict>
  data_root: string
  project_root: string
}

export type ConfigValue = string | number | boolean | string[] | null
export type ConfigDict = Record<string, ConfigValue>

export interface StartRunResponse {
  run_id: string
  out: string
}

export type RunStatus = 'running' | 'done' | 'error'

export interface RunSummary {
  id: string
  status: RunStatus
  out: string
  returncode: number | null
}

export interface RunDetail extends RunSummary {
  log: string[]
}

export interface RunsList {
  runs: RunSummary[]
}

export type WsMessage =
  | { type: 'line'; line: string }
  | { type: 'done'; returncode: number; status: RunStatus }
  | { type: 'error'; msg: string }

export interface SegSummary {
  scanned?: number
  kept?: number
  skipped?: number
  cache_hits?: number
  method_counts?: Record<string, number>
  skip_counts?: Record<string, number>
  confidence?: { min: number; median: number; mean: number }
}

export interface SegSkipped {
  source: string
  reason: string
  confidence: number
}

export interface SegReportData {
  summary?: SegSummary
  skipped?: SegSkipped[]
}

export interface SegCheckResponse {
  returncode: number
  report: SegReportData | null
  images: string[]
  log: string
  out: string
}

export interface DatasetsList {
  root: string
  datasets: string[]
}

export interface DatasetInfo {
  path: string
  splits: Record<string, number>
  classes: string[]
  qa: string[]
  seg_report: SegReportData | null
  run_config?: string
}

export interface FsEntry {
  name: string
  path: string
  is_dir: boolean
}

export interface FsList {
  path: string
  parent: string | null
  entries: FsEntry[]
  data_root: string
}

export interface ConfigSaveResponse {
  saved: string
  format: 'yaml' | 'json'
}

export interface UploadResponse {
  saved_dir: string
  files: string[]
  count: number
}

export type Split = 'train' | 'val'
