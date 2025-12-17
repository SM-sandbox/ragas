#!/bin/bash
# Quick launch for MkDocs documentation
# Usage: ./docs_serve.sh

cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
echo "Starting docs at http://localhost:8000"
echo "Press Ctrl+C to stop"
mkdocs serve
