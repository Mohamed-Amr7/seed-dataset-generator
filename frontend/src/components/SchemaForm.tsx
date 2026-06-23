import type { FormSchema } from '@/api/types'
import { useConfigStore } from '@/store/configStore'
import { isFieldVisible } from '@/lib/segFields'
import { FieldRow } from './FieldRow'

export function SchemaForm({
  schema,
  errors,
}: {
  schema: FormSchema
  errors?: Record<string, string>
}) {
  const segment = useConfigStore((s) =>
    typeof s.config.segment === 'string' ? s.config.segment : 'auto',
  )

  return (
    <div className="flex flex-col gap-4">
      {schema.groups.map((group) => {
        const fields = group.fields.filter((f) => isFieldVisible(f.name, segment))
        if (!fields.length) return null
        return (
          <fieldset key={group.title} className="rounded-lg border border-border p-3">
            <legend className="px-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {group.title}
            </legend>
            <div className="flex flex-col divide-y divide-border">
              {fields.map((f) => (
                <FieldRow key={f.name} field={f} error={errors?.[f.name]} />
              ))}
            </div>
          </fieldset>
        )
      })}
    </div>
  )
}
