import type { ConfigDict, ConfigValue, FormSchema } from '@/api/types'

/** Flatten every field's default into a config dict (the form's starting point). */
export function schemaDefaults(schema: FormSchema): ConfigDict {
  const out: ConfigDict = {}
  for (const group of schema.groups) {
    for (const field of group.fields) {
      out[field.name] = (field.default ?? null) as ConfigValue
    }
  }
  return out
}
