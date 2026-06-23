import { ArrowUp, File as FileIcon, Folder } from 'lucide-react'
import type * as React from 'react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '@/api/client'
import { fsList } from '@/api/endpoints'
import { useSchema } from '@/api/queries'
import type { FsList } from '@/api/types'
import { usePickerStore } from '@/store/pickerStore'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { ErrorText, Spinner } from '@/components/common'

/** The shared, sandbox-aware filesystem picker. Radix Dialog gives focus-trap + Esc;
 *  list rows are buttons (Tab/Enter) with ArrowUp/Down roving for keyboard navigation. */
export function PathPicker() {
  const { open, kind, fieldName, initialPath, onSelect, close } = usePickerStore()
  const dataRoot = useSchema().data?.data_root ?? null

  const [listing, setListing] = useState<FsList | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [newName, setNewName] = useState('')
  const listRef = useRef<HTMLDivElement>(null)

  const load = useCallback(
    async (path: string | null) => {
      setLoading(true)
      setError(null)
      try {
        setListing(await fsList(path, kind))
      } catch (e) {
        setError(e instanceof ApiError ? e.message : 'Failed to list directory')
      } finally {
        setLoading(false)
      }
    },
    [kind],
  )

  useEffect(() => {
    // Fetch-on-open effect: load() drives its own loading/error state via the network.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (open) void load(initialPath ?? dataRoot)
  }, [open, initialPath, dataRoot, load])

  if (!open) return null

  // Resets clear the typed name so it never leaks into the next open.
  const reset = () => {
    setNewName('')
    close()
  }

  const pick = (path: string) => {
    onSelect?.(path)
    setNewName('')
    close()
  }

  const confirm = () => {
    if (!listing) return
    const extra = newName.trim().replace(/^\/+|\/+$/g, '')
    if (kind === 'file' && !extra) return // file mode needs a typed name or a clicked file
    const base = listing.path.replace(/\/+$/, '')
    pick(extra ? `${base}/${extra}` : base)
  }

  const onListKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
    const rows = Array.from(
      listRef.current?.querySelectorAll<HTMLButtonElement>('[data-fs-row]') ?? [],
    )
    if (!rows.length) return
    e.preventDefault()
    const idx = rows.indexOf(document.activeElement as HTMLButtonElement)
    const next = e.key === 'ArrowDown' ? Math.min(rows.length - 1, idx + 1) : Math.max(0, idx - 1)
    rows[next]?.focus()
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (!o) reset()
      }}
    >
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle>
            Select a {kind === 'file' ? 'file' : 'folder'} for{' '}
            <code className="font-mono text-primary">{fieldName}</code>
          </DialogTitle>
          <DialogDescription>
            Browsing is sandboxed to the project and data roots.
          </DialogDescription>
        </DialogHeader>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={!listing?.parent}
            onClick={() => load(listing?.parent ?? null)}
          >
            <ArrowUp /> Up
          </Button>
          <span className="truncate font-mono text-xs text-muted-foreground" title={listing?.path}>
            {listing?.path ?? '…'}
          </span>
        </div>

        {error && <ErrorText>{error}</ErrorText>}

        {/* eslint-disable-next-line jsx-a11y/no-static-element-interactions */}
        <div
          ref={listRef}
          onKeyDown={onListKeyDown}
          className="max-h-[40vh] min-h-[8rem] overflow-auto rounded-md border border-border"
        >
          {loading ? (
            <div className="flex items-center gap-2 p-3 text-sm text-muted-foreground">
              <Spinner /> loading…
            </div>
          ) : !listing?.entries.length ? (
            <div className="p-3 text-sm text-muted-foreground">(empty)</div>
          ) : (
            listing.entries.map((entry) => (
              <button
                key={entry.path}
                type="button"
                data-fs-row
                onClick={() => (entry.is_dir ? load(entry.path) : pick(entry.path))}
                title={entry.path}
                className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm hover:bg-accent focus:bg-accent focus:outline-none"
              >
                {entry.is_dir ? (
                  <Folder className="h-4 w-4 shrink-0 text-primary" />
                ) : (
                  <FileIcon className="h-4 w-4 shrink-0 opacity-70" />
                )}
                <span className="truncate">{entry.name}</span>
              </button>
            ))
          )}
        </div>

        <div className="flex flex-col gap-1">
          <label htmlFor="fs-newname" className="text-xs text-muted-foreground">
            {kind === 'file' ? 'New file name' : 'New subfolder'} (optional)
          </label>
          <Input
            id="fs-newname"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder={kind === 'file' ? 'e.g. run_config.yaml' : 'e.g. soybean_healthy'}
            onKeyDown={(e) => {
              if (e.key === 'Enter') confirm()
            }}
          />
        </div>

        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={reset}>
            Cancel
          </Button>
          <Button onClick={confirm}>
            {kind === 'file' ? 'Use this name' : 'Select this folder'}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
