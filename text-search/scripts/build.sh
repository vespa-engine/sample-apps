#!/usr/bin/env bash
#

set -euo pipefail
if [[  -n "${DEBUG:-}" ]]; then
  set -x
fi


main() {
  local APP_DIR

  APP_DIR="$(realpath "$(dirname "$0")/..")"

  echo "### Setup '${APP_DIR}'  ###"
  pushd "$APP_DIR" >/dev/null

  mkdir -p dist/
  echo "Generating:"

  # Convert the documents to JSONL format
  echo '  * dist/documents.jsonl'
  bash scripts/convert-msmarco.sh
  mkdir -p dataset/
  cp dataset/documents.jsonl dist/documents.jsonl

  (
    echo '  * dist/application.zip'
    cp README.md app/
    cd app && zip --recurse-paths --quiet ../dist/application.zip ./
  )

  popd >/dev/null
  echo "### Finished building sample application ###"
}

main "$@"
