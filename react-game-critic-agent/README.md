# react-game-critic-agent

An AI critic agent that playtests a React game repo in a real browser and reports whether
anything looks broken.

Given a path to a React game repo, it:

1. **Reads the repo** — `package.json`, README, and a shallow source listing — to figure out
   how to start it (npm/yarn/pnpm, dev vs. start script, framework-based port guess) and to give
   Claude enough context to plan a sensible playtest.
2. **Starts the dev server** (or reuses a URL you already have running).
3. **Attaches a persistent Chrome session** via [Playwright](https://playwright.dev)'s
   `launchPersistentContext`, using a fixed on-disk profile directory so cookies, localStorage,
   and other page state survive across separate runs of the agent — this is the "the same
   browser session" a human tester would keep between sessions, not a fresh throwaway browser
   each time.
4. **Asks Claude to design a test plan**: a short JSON sequence of actions (navigate, wait,
   keypress, click, screenshot, evaluate) tailored to what the source/README suggest about the
   game's controls and goal.
5. **Executes the plan against the raw Chrome DevTools Protocol** — keyboard/mouse input via
   `Input.dispatchKeyEvent` / `Input.dispatchMouseEvent`, screenshots via `Page.captureScreenshot`
   — rather than Playwright's higher-level locator API, so the agent is driving the page exactly
   the way a real player's input would arrive. Console errors, uncaught exceptions, and failed
   network requests are collected continuously via the CDP `Runtime`/`Network` domains and
   attached to whichever screenshot follows them.
6. **Hands each screenshot to Claude as a VLM critic**, along with the action that preceded it
   and any console/network issues captured since the last screenshot, and asks for a structured
   severity judgment (`none` / `low` / `medium` / `high`) plus a description.
7. **Writes a report** (`report.md` + `report.json`) with every screenshot, the critic's verdict
   on each, the raw console/network diagnostics, and an overall severity rollup.

## Setup

```bash
cd react-game-critic-agent
npm install
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

By default the agent launches your system-installed Chrome (`channel: "chrome"`) so the
persistent profile behaves like your everyday browser; set `CRITIC_CHROME_CHANNEL=chromium` in
`.env` to use Playwright's bundled Chromium instead (run `npx playwright install chromium` first
if you haven't already).

## Usage

```bash
# Point it at a repo; it will run `npm run dev` (or yarn/pnpm) itself
npm run dev -- --repo ../path/to/some-react-game

# Or test a URL that's already running
npm run dev -- --repo ../path/to/some-react-game --url http://localhost:5173

# Give the planner extra hints about controls/goal
npm run dev -- --repo ../path/to/some-react-game --context "Controls are WASD, goal is to reach the green flag"

# Headless, custom output dir
npm run dev -- --repo ../path/to/some-react-game --headless --out ./reports
```

Try it against the bundled fixture game:

```bash
npm run dev -- --repo ./examples/sample-react-game
```

Each run writes to `.critic-agent/reports/<timestamp>/`:

- `screenshots/*.png` — every screenshot the plan captured
- `report.md` — human-readable report with embedded screenshots and verdicts
- `report.json` — the same data, structured

The Chrome profile persists at `.critic-agent/profile/` across runs (override with `--profile`).

## CLI options

| Flag | Description |
| --- | --- |
| `--repo <path>` | **Required.** Path to the React game repo to test |
| `--url <url>` | Skip auto-starting a dev server and test this URL directly |
| `--port <port>` | Port to run/expect the dev server on |
| `--headless` | Run Chrome headless |
| `--profile <dir>` | Persistent Chrome profile directory (default `.critic-agent/profile`) |
| `--out <dir>` | Output directory for screenshots/reports (default `.critic-agent/reports`) |
| `--context <text>` | Extra hints for the planner (controls, goal, known trouble spots) |

## Project layout

```
src/
  cli.ts             CLI entry point — wires everything together
  repoAnalyzer.ts     inspects the target repo (package.json, README, file tree)
  devServer.ts        spawns the repo's dev script, waits for it to serve
  browserSession.ts   launches the persistent Chrome context + CDP session
  planner.ts           Claude call that turns repo context into a JSON test plan
  executor.ts          runs the plan's steps against the browser via CDP
  critic.ts             Claude vision call that judges each screenshot
  report.ts             assembles the markdown/JSON report
  cdp/
    input.ts            raw Input.dispatchKeyEvent / dispatchMouseEvent helpers
    capture.ts           raw Page.captureScreenshot helper
    diagnostics.ts        Runtime/Network domain listeners for console + network issues
examples/
  sample-react-game/    tiny arrow-key game fixture for trying the agent end-to-end
```

## Limitations / possible extensions

- The planner and critic are both single-pass Claude calls; there's no retry/re-plan loop if a
  step fails (e.g. a selector-free click misses because the layout differs from what the planner
  assumed) — a step failing just logs a warning and the plan continues.
- Diagnostics are best-effort: `Network.loadingFailed` doesn't include the URL directly in this
  implementation (CDP gives it by `requestId`), so failed *requests* are reported by request ID
  rather than URL; non-2xx *responses* do include the URL.
- No accessibility- or performance-focused checks — this is purely "does it look and behave
  correctly," judged visually plus console/network signals.
- Headed mode (the default) is meant for local development where you can watch the agent play;
  use `--headless` for CI-style runs.
