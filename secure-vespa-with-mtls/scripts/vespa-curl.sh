#!/usr/bin/env bash
# Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -e

SAMPLE_APP_ROOT=$(dirname $0)/..
curl --key ${SAMPLE_APP_ROOT}/pki/vespa/host.key --cert ${SAMPLE_APP_ROOT}/pki/vespa/host.pem --cacert ${SAMPLE_APP_ROOT}/pki/vespa/ca-vespa.pem "$@"