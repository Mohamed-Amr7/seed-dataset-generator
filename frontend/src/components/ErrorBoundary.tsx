import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}
interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('UI render error:', error, info)
  }

  render() {
    if (this.state.error) {
      return (
        <div
          role="alert"
          className="m-6 rounded-md border border-destructive/40 bg-card p-4 text-sm"
        >
          <p className="font-semibold text-destructive">Something went wrong rendering the UI.</p>
          <pre className="mt-2 overflow-auto whitespace-pre-wrap text-xs text-muted-foreground">
            {this.state.error.message}
          </pre>
        </div>
      )
    }
    return this.props.children
  }
}
