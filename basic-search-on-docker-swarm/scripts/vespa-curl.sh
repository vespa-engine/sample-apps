#!/bin/bash
curl --key $(dirname $0)/../pki/vespa/host.key --cert $(dirname $0)/../pki/vespa/host.pem --cacert $(dirname $0)/../pki/vespa/ca-internal.pem "$@"