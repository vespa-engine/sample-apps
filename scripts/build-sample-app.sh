#!/usr/bin/env bash
#

usage() {
  echo "Usage: $0 app-name"
  echo
  echo "Creates the application package and documents.jsonl file for the specified sample application."
  exit 1
}

set -euo pipefail
if [[  -n "${DEBUG:-}" ]]; then
  set -x
fi
if [[ $# -ne 1 ]]; then
  usage
fi


main() {
  local SAMPLE_APP_DIR="$(realpath "$(dirname "$0")/..")"
  local APP_NAME="$1"

  pushd "$SAMPLE_APP_DIR"  >/dev/null

  # Ensure that the APP_NAME is a subdirectory of the current directory to prevent directory traversal attacks
  if [[ "$(realpath "$APP_NAME" )" != "$(realpath .)/$APP_NAME" ]]; then
    echo "Invalid application name. Please provide a valid subdirectory name."
    exit 2
  fi

  # Start wotking in the specified application
  echo "### Setup '${APP_NAME}'  ###"
  pushd "./${APP_NAME}" >/dev/null || (
    echo "The specified application directory '${APP_NAME}' does not exist."
    exit 3
  )

  if [[ ! -d "app" ]]; then
    echo "The application directory 'app' does not exist in '${APP_NAME}'."
    exit 4
  fi

  if [[ -d "dist" ]]; then
    echo "Cleaning up the previous build..."
    rm -rf dist
  fi

  mkdir -p dist

  if [[ -x "./scripts/build.sh" ]]; then
    echo "Found application-specific build script. Running it..."
    bash ./scripts/build.sh
  else
    echo "No build script found. Using default build process."
    echo "Generating:"
    (
      echo '  * dist/application.zip'
      cp README.md app/
      cd app && zip --recurse-paths --quiet ../dist/application.zip ./
    )

    echo '  * dist/documents.jsonl'
    {
      for f in dataset/*.json; do
        jq -c . "$f"
      done
    } > dist/documents.jsonl
fi

  echo
  echo "Build completed successfully."
  echo "The application package and documents.jsonl file are located in the '${APP_NAME}/dist' directory."
  echo

  popd  >/dev/null # application directory
  popd  >/dev/null # back to the initial directory
}

main "$@"
