import type * as React from 'react'
import { cn } from '@/lib/utils'

export type BadgeVariant =
  | 'default'
  | 'secondary'
  | 'success'
  | 'destructive'
  | 'warning'
  | 'outline'

const variants: Record<BadgeVariant, string> = {
  default: 'bg-primary/15 text-primary border-primary/30',
  secondary: 'bg-secondary text-secondary-foreground border-border',
  success: 'bg-success/15 text-success border-success/30',
  destructive: 'bg-destructive/15 text-destructive border-destructive/30',
  warning: 'bg-warning/15 text-warning border-warning/30',
  outline: 'text-foreground border-border',
}

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant
}

export function Badge({ className, variant = 'default', ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium',
        variants[variant],
        className,
      )}
      {...props}
    />
  )
}
