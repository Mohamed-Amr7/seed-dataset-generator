import { Moon, Sun } from 'lucide-react'
import { useState } from 'react'
import { Button } from '@/components/ui/button'

function isDark() {
  return document.documentElement.classList.contains('dark')
}

export function ThemeToggle() {
  const [dark, setDark] = useState(isDark)
  const toggle = () => {
    const next = !dark
    document.documentElement.classList.toggle('dark', next)
    try {
      localStorage.setItem('msg-theme', next ? 'dark' : 'light')
    } catch {
      /* localStorage may be unavailable; ignore */
    }
    setDark(next)
  }
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={toggle}
      aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
    >
      {dark ? <Sun /> : <Moon />}
    </Button>
  )
}
