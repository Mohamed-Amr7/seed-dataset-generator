function caption(url: string): string {
  const last = url.split('/').pop() ?? ''
  try {
    return decodeURIComponent(last)
  } catch {
    return last
  }
}

export function ImageGrid({ urls, emptyText }: { urls: string[]; emptyText?: string }) {
  if (!urls.length) {
    return emptyText ? <p className="text-sm text-muted-foreground">{emptyText}</p> : null
  }
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-3">
      {urls.map((url) => (
        <figure key={url} className="overflow-hidden rounded-md border border-border">
          <img
            src={url}
            alt={caption(url)}
            loading="lazy"
            className="h-32 w-full bg-black/30 object-contain"
          />
          <figcaption
            className="truncate p-1 text-center text-[11px] text-muted-foreground"
            title={caption(url)}
          >
            {caption(url)}
          </figcaption>
        </figure>
      ))}
    </div>
  )
}
