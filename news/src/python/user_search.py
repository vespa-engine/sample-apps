#! /usr/bin/env python3
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import sys
import json

from vespa_client import connect, query_news, query_user_embedding


def main():
    user_id = sys.argv[1]
    hits = sys.argv[2] if len(sys.argv) > 2 else 10
    filter = sys.argv[3] if len(sys.argv) > 3 else ""

    url, cert = connect()

    user_vector = query_user_embedding(user_id, url, cert)
    result = query_news(user_vector, int(hits), filter, url, cert)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
