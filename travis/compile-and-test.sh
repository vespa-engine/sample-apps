#!/usr/bin/env bash

# Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

set -e

source /etc/profile.d/jdk-env.sh

# Compile and test the sample apps
mvn -V --batch-mode --no-snapshot-updates install

