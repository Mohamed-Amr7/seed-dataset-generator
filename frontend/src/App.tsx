import { useEffect, useState } from 'react'
import { useSchema } from '@/api/queries'
import { useConfigStore } from '@/store/configStore'
import { schemaDefaults } from '@/lib/schema'
import { useRunSocket } from '@/hooks/useRunSocket'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Spinner } from '@/components/common'
import { ThemeToggle } from '@/components/ThemeToggle'
import { PathPicker } from '@/components/PathPicker'
import { Toaster } from '@/components/ui/toaster'
import { RunTab } from '@/features/RunTab'
import { SegTunerTab } from '@/features/SegTunerTab'
import { BrowseTab } from '@/features/BrowseTab'
import { ConfigTab } from '@/features/ConfigTab'

const TAB_IDS = ['run', 'seg', 'browse', 'config'] as const
type TabId = (typeof TAB_IDS)[number]

function readHash(): TabId {
  const h = window.location.hash.replace('#', '')
  return (TAB_IDS as readonly string[]).includes(h) ? (h as TabId) : 'run'
}

export default function App() {
  const { data: schema, isLoading, isError, error } = useSchema()
  const seeded = useConfigStore((s) => s.seeded)
  const replace = useConfigStore((s) => s.replace)
  const [tab, setTab] = useState<TabId>(readHash)
  const socket = useRunSocket()

  // Seed the form with schema defaults + recommended overlay on first load (unless a
  // persisted config already rehydrated from localStorage).
  useEffect(() => {
    if (schema && !seeded) replace({ ...schemaDefaults(schema), ...schema.recommended })
  }, [schema, seeded, replace])

  // Keep the URL hash in sync so tabs are deep-linkable.
  useEffect(() => {
    window.location.hash = tab
  }, [tab])

  return (
    <div className="mx-auto flex min-h-screen max-w-[1400px] flex-col gap-4 p-4 sm:p-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold">MultiSeedGen</h1>
          <p className="text-xs text-muted-foreground">
            Synthetic multi-seed detection dataset generator
          </p>
        </div>
        <ThemeToggle />
      </header>

      {isLoading ? (
        <div className="flex items-center gap-2 p-8 text-muted-foreground">
          <Spinner /> Loading schema…
        </div>
      ) : isError || !schema ? (
        <div role="alert" className="rounded-md border border-destructive/40 bg-card p-4 text-sm">
          <p className="font-semibold text-destructive">Could not load the form schema.</p>
          <p className="mt-1 text-muted-foreground">
            {error instanceof Error ? error.message : 'Is the backend running on :8000?'}
          </p>
        </div>
      ) : (
        <Tabs value={tab} onValueChange={(v) => setTab(v as TabId)} className="flex flex-col gap-4">
          <TabsList>
            <TabsTrigger value="run">Run</TabsTrigger>
            <TabsTrigger value="seg">Segmentation tuner</TabsTrigger>
            <TabsTrigger value="browse">Dataset browser</TabsTrigger>
            <TabsTrigger value="config">Config</TabsTrigger>
          </TabsList>
          <TabsContent value="run">
            <RunTab schema={schema} socket={socket} />
          </TabsContent>
          <TabsContent value="seg">
            <SegTunerTab />
          </TabsContent>
          <TabsContent value="browse">
            <BrowseTab />
          </TabsContent>
          <TabsContent value="config">
            <ConfigTab />
          </TabsContent>
        </Tabs>
      )}

      <PathPicker />
      <Toaster />
    </div>
  )
}
