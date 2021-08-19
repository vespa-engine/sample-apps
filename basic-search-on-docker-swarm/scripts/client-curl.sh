#!/bin/bash
curl --key $(dirname $0)/../pki/client/host.key --cert $(dirname $0)/../pki/client/host.pem --cacert $(dirname $0)/../pki/vespa/ca-internal.pem "$@"