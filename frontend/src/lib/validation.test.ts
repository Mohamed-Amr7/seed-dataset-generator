import { describe, expect, it } from 'vitest'
import { validateRunConfig } from './validation'

describe('validateRunConfig', () => {
  it('fails and reports both fields when sources/out are missing', () => {
    const r = validateRunConfig({})
    expect(r.ok).toBe(false)
    expect(r.errors.sources).toBeDefined()
    expect(r.errors.out).toBeDefined()
  })

  it('treats an empty string as missing', () => {
    expect(validateRunConfig({ sources: '', out: 'o' }).ok).toBe(false)
  })

  it('passes when both are present', () => {
    expect(validateRunConfig({ sources: 'seeds', out: 'datasets/x' })).toEqual({
      ok: true,
      errors: {},
    })
  })
})
