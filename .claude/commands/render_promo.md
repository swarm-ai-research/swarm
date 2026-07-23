# /render_promo

Render the SWARM Remotion promo video to MP4, GIF, or still frame.

> **Note**: The `promo/` source code has been moved to [`swarm-ai-research/swarm-artifacts`](https://github.com/swarm-ai-research/swarm-artifacts). Clone that repo first if `promo/` is not present locally.

## Usage

```
/render_promo [--format mp4|gif|still] [--frame N] [--open]
```

Examples:
- `/render_promo` (renders MP4, opens it)
- `/render_promo --format gif`
- `/render_promo --format still --frame 60`

## Behavior

### Step 0: Check promo/ exists

If `promo/` is not present, instruct the user:
```
promo/ not found. Clone it from the artifacts repo:
  git clone https://github.com/swarm-ai-research/swarm-artifacts.git /tmp/swarm-artifacts
  cp -R /tmp/swarm-artifacts/promo ./promo
```

### Step 1: Install dependencies (if needed)

```bash
cd promo && npm ls remotion 2>/dev/null || npm install
```

Only install if `node_modules` is missing or incomplete.

### Step 2: Type-check

```bash
cd promo && node_modules/.bin/tsc --noEmit
```

If type errors exist, fix them before rendering. Common issues:
- Missing `typescript` dev dependency: `npm install typescript --save-dev`
- Unicode escape sequences in JSX: use actual characters instead of `\uXXXX`

### Step 3: Render

**MP4 (default):**
```bash
cd promo && npx remotion render SwarmPromo out/swarm-promo.mp4
```

**GIF:**
```bash
cd promo && npx remotion render SwarmPromo out/swarm-promo.gif --image-format=png --every-nth-frame=2
```

**Still frame:**
```bash
cd promo && npx remotion still SwarmPromo --frame=<N> out/thumbnail.png
```

Default frame for stills: 60 (shows the SWARM title fully animated). Frame 0 is too dark.

### Step 4: Report and open

```
Rendered:
  Format:   <mp4|gif|still>
  Output:   promo/out/<filename>
  Size:     <file size>
  Duration: <seconds if video>
```

Then open the output:
```bash
open promo/out/<filename>
```

## Project Structure

The Remotion project lives in `promo/`:
- `promo/src/Root.tsx` — Composition definition (1170 frames, 30fps, 1920x1080)
- `promo/src/SwarmPromo.tsx` — Scene sequencing via TransitionSeries
- `promo/src/scenes/*.tsx` — Individual scenes
- `promo/src/theme.ts` — Shared colors and fonts
- `promo/out/` — Render output (gitignored)

## Constraints

- Always run from the repo root, using `cd promo &&` prefix.
- Do not modify scene files unless the user explicitly asks for content changes.
- The `promo/out/` directory is gitignored — renders are not committed.
- If `npx remotion` is not found, ensure `@remotion/cli` is in dependencies.
