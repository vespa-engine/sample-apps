#!/usr/bin/env bash
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly resource_dir="${script_dir}/src/main/application/lucene-linguistics"
readonly temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/german-decompounder.XXXXXX")"
trap 'rm -rf "${temporary_dir}"' EXIT

mkdir -p "${resource_dir}"
git clone --depth 1 https://github.com/uschindler/german-decompounder.git "${temporary_dir}/repository"
cp "${temporary_dir}/repository/dictionary-de.txt" "${resource_dir}/"
cp "${temporary_dir}/repository/de_DR.xml" "${resource_dir}/"
