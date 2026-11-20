#!/usr/bin/env bash
# Render the Graphviz architecture diagram to PNG and SVG.
# Requires Graphviz (`sudo apt-get install graphviz` or `brew install graphviz`).
set -euo pipefail
cd "$(dirname "$0")"
dot -Tpng -Gdpi=150 architecture.dot -o architecture.png
dot -Tsvg architecture.dot -o architecture.svg
echo "Rendered architecture.png and architecture.svg"
