import { FolderOpen } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { ConfigValue, SchemaField } from '@/api/types'
import { useConfigStore } from '@/store/configStore'
import { usePickerStore } from '@/store/pickerStore'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { cn } from '@/lib/utils'

function toDisplay(value: ConfigValue): string {
  if (value == null) return ''
  if (Array.isArray(value)) return value.join(',')
  return String(value)
}

const REQUIRED = new Set(['sources', 'out'])

/**
 * One schema-driven field. Text/number/array widgets keep a local string mirror so partial
 * input (e.g. "0." while typing a float) survives, while still re-syncing when the value is
 * changed externally (preset switch, load-from-file, Browse). bool -> Switch, choice -> Select.
 *
 * By default the field is bound to the shared config store. Pass `value` + `onChange` to drive
 * it from external (e.g. component-local) state instead — used by the segmentation tuner, whose
 * per-method knobs are intentionally kept out of the run config.
 */
export function FieldRow({
  field,
  error,
  value: controlledValue,
  onChange,
}: {
  field: SchemaField
  error?: string
  value?: ConfigValue
  onChange?: (value: ConfigValue) => void
}) {
  const storeValue = useConfigStore((s) => s.config[field.name])
  const setField = useConfigStore((s) => s.setField)
  const openPicker = usePickerStore((s) => s.openPicker)

  const controlled = onChange !== undefined
  const value = controlled ? (controlledValue ?? null) : storeValue

  const id = `field-${field.name}`
  const isArrayField = field.name === 'backgrounds'
  const isRequired = REQUIRED.has(field.name)

  const [text, setText] = useState(() => toDisplay(value))
  const lastPushed = useRef<ConfigValue>(value)

  // Re-sync the local mirror only on external changes (value differs from what we last pushed).
  useEffect(() => {
    if (value !== lastPushed.current) {
      setText(toDisplay(value))
      lastPushed.current = value
    }
  }, [value])

  const push = (next: ConfigValue) => {
    lastPushed.current = next
    if (controlled) onChange(next)
    else setField(field.name, next)
  }

  const onTextChange = (raw: string) => {
    setText(raw)
    if (field.widget === 'int' || field.widget === 'float') {
      if (raw.trim() === '') return push(null)
      const n = field.widget === 'int' ? parseInt(raw, 10) : parseFloat(raw)
      return push(Number.isNaN(n) ? null : n)
    }
    if (isArrayField) {
      return push(
        raw
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
      )
    }
    return push(raw)
  }

  const label = (
    <label htmlFor={id} className="flex items-center gap-1 font-mono text-xs">
      {field.name}
      {isRequired && (
        <span className="text-destructive" aria-hidden>
          *
        </span>
      )}
    </label>
  )

  let control: React.ReactNode
  if (field.widget === 'bool') {
    control = (
      <Switch id={id} checked={Boolean(value)} onCheckedChange={(checked) => push(checked)} />
    )
  } else if (field.widget === 'choice' && field.choices) {
    control = (
      <Select value={value == null ? '' : String(value)} onValueChange={(v) => push(v)}>
        <SelectTrigger id={id} className="w-full">
          <SelectValue placeholder="Select…" />
        </SelectTrigger>
        <SelectContent>
          {field.choices.map((c) => (
            <SelectItem key={c} value={c}>
              {c}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    )
  } else {
    const isPath = field.path_kind && field.widget === 'text'
    control = (
      <div className="flex gap-2">
        <Input
          id={id}
          type={field.widget === 'int' || field.widget === 'float' ? 'number' : 'text'}
          step={field.widget === 'float' ? 'any' : undefined}
          value={text}
          aria-invalid={error ? true : undefined}
          onChange={(e) => onTextChange(e.target.value)}
          className={cn(error && 'border-destructive focus-visible:ring-destructive')}
        />
        {isPath && (
          <Button
            type="button"
            variant="outline"
            size="icon"
            aria-label={`Browse for ${field.name}`}
            onClick={() =>
              openPicker({
                kind: field.path_kind === 'file' ? 'file' : 'dir',
                fieldName: field.name,
                initialPath: typeof value === 'string' && value ? value : null,
                // The picker sets the value externally (it never goes through the Input's
                // onChange), so sync the local text mirror too — otherwise push() updates
                // lastPushed.current and the re-sync effect skips it, leaving a stale input.
                onSelect: (p) => {
                  setText(toDisplay(p))
                  push(p)
                },
              })
            }
          >
            <FolderOpen />
          </Button>
        )}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 items-center gap-1 py-2 sm:grid-cols-[minmax(140px,38%)_1fr]">
      <div className="flex flex-col gap-0.5">
        {label}
        {field.help && (
          <span className="text-[11px] leading-tight text-muted-foreground">{field.help}</span>
        )}
      </div>
      <div className="flex flex-col gap-1">
        {control}
        {error && (
          <span role="alert" className="text-xs text-destructive">
            {error}
          </span>
        )}
      </div>
    </div>
  )
}
