#!/usr/bin/env bash
set -e

SAMPLE_APP_ROOT=$(dirname $0)/..
curl --key ${SAMPLE_APP_ROOT}/pki/vespa/host.key --cert ${SAMPLE_APP_ROOT}/pki/vespa/host.pem --cacert ${SAMPLE_APP_ROOT}/pki/vespa/ca-vespa.pem "$@"