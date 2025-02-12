#!/usr/bin/env python3
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import yaml
import argparse
import re


def get_app(endpoint):
    splits = re.split(r'/|\.', endpoint)
    return splits[3] + '.' + splits[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.yaml_config, "r"))
    urls = [get_app(entry["url"]) + "," + entry["url"] for entry in config.get("search", {}).get("feed_endpoints", []) if "url" in entry]
    print("\n".join(urls))


if __name__ == "__main__":
    main()
