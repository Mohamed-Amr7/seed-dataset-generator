# MultiSeedGen web UI (frontend)

The source for MultiSeedGen's single-page web UI: **React + TypeScript + Vite + Tailwind v4 +
Radix UI**, with TanStack Query (server state), Zustand (UI state), Zod (validation) and Vitest
(tests). It builds into **`../multiseedgen/web/static/`**, a committed bundle the FastAPI backend
serves — so the Python package runs with no Node toolchain.

## Quick start

Node **22 LTS** (`.nvmrc`; `engines` requires ≥ 20). Run the backend and Vite dev server together:

```bash
multiseedgen-web                 # FastAPI backend on :8000 (separate terminal)
cd frontend && npm ci && npm run dev   # UI on :5173, hot-reload
```

Vite proxies `/api` and `/ws` to `:8000`, so there's no CORS to configure. From the repo root,
`make dev` starts both sides for you.

## Scripts

| Script | What it does |
|--------|--------------|
| `npm run dev` | Vite dev server (`:5173`, HMR) |
| `npm run build` | `tsc -b && vite build` → writes the committed `../multiseedgen/web/static/` bundle |
| `npm run typecheck` | `tsc -b --noEmit` (strict) |
| `npm run lint` | ESLint (typescript-eslint, react-hooks, jsx-a11y) |
| `npm run test` / `test:watch` / `coverage` | Vitest + Testing Library |
| `npm run format` / `format:check` | Prettier write / check |
| `npm run preview` | Serve a production build locally |

## The committed-bundle rule

The packaged app serves `../multiseedgen/web/static/`, **not** `frontend/src`. After any UI change,
run `npm run build` and **commit the regenerated `../multiseedgen/web/static/` assets** alongside
your source change — a source-only edit ships stale JS, and CI fails on the resulting bundle drift.

## More

The full dev loop, the regression gate (lint / typecheck / test / build), and project conventions
live in **[`../docs/contributing.md`](../docs/contributing.md)** (Frontend section). The HTTP/WS API
this UI talks to is documented in **[`../docs/api.md`](../docs/api.md)**.
