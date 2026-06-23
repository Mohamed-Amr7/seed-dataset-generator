import type { FormSchema } from '@/api/types'
import { useConfigStore } from '@/store/configStore'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

/** Switch the form to a named recipe. Merges onto the current config (like the old UI),
 *  so fields absent from a preset keep their recommended values. */
export function PresetSelect({ schema }: { schema: FormSchema }) {
  const setMany = useConfigStore((s) => s.setMany)
  const names = Object.keys(schema.presets)
  if (!names.length) return null
  return (
    <Select
      onValueChange={(name) => {
        const preset = schema.presets[name]
        if (preset) setMany({ ...preset })
      }}
    >
      <SelectTrigger className="w-full">
        <SelectValue placeholder="Start from a preset…" />
      </SelectTrigger>
      <SelectContent>
        {names.map((n) => (
          <SelectItem key={n} value={n}>
            {n}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
