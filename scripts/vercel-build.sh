#!/usr/bin/env bash
set -euo pipefail

PY=3.12
VENV=.venv-build

uv python install "$PY"
uv venv --python "$PY" "$VENV"
# mkdocs pinned below 2.0: the Material team's advisory (printed in every
# build) says MkDocs 2.0 removes the plugin system and theming our site
# depends on, with no migration path. Keep 1.x until the ecosystem settles.
uv pip install --python "$VENV/bin/python" \
  'mkdocs>=1.6,<2' \
  mkdocs-material \
  'mkdocstrings[python]' \
  pymdown-extensions \
  mkdocs-git-revision-date-localized-plugin \
  mkdocs-rss-plugin

(cd viz && npm install && npm run build:deploy)

# Intentionally NOT --strict here: link/nav integrity is gated by CI
# (.github/workflows/ci.yml, `mkdocs build --strict`) before code reaches main,
# so production deploys stay resilient and aren't blocked by a late non-fatal
# warning. Keep the gate in CI, keep deploys lenient.
"$VENV/bin/mkdocs" build

# gitlawb dashboard backfill snapshot.
# Regenerate the scored snapshot at build time (stdlib only, no swarm install).
# A scheduled Vercel deploy hook (.github/workflows/gitlawb-snapshot.yml) re-runs
# this build to keep it fresh without pushing to a protected branch. Fail-safe:
# a node outage writes an empty snapshot rather than failing the build.
"$VENV/bin/python" scripts/gen_gitlawb_snapshot.py site/bridges/gitlawb_snapshot.json
