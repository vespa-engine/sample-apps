#!/usr/bin/env bash

# Copyright 2020 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

set -e

readonly CERT_FILE=data-plane-public-cert.pem
readonly PRIVATE_KEY_FILE=data-plane-private-key.pem
readonly DEPLOY_KEY_FILE=deploy-key.pem
readonly VESPA_CLOUD_ENDPOINT="https://api.vespa-external-cd.aws.oath.cloud:4443"

source /etc/profile.d/jdk-env.sh

yum install -y openssl

cd vespa-cloud/album-recommendation-searcher/

# Generate self-signed x509 certificate for data plane security
openssl req -x509 -nodes -days 14 -newkey rsa:4096 \
  -subj "/C=NO/ST=Trondheim/L=Trondheim/O=Vespa/OU=Vespa/CN=client.vespa.ai" \
  -keyout ${PRIVATE_KEY_FILE} -out ${CERT_FILE}

# Copy certificate to src/main/security/
mkdir -p src/main/application/security && cp ${CERT_FILE} src/main/application/security/clients.pem

# Set upgrade policy to canary
sed -i'' -e '/<deployment[^/>]*>/ s,$,\n  <upgrade policy="canary" />,' src/main/application/deployment.xml

# Write deploy key to file
# Deploy key must be encoded as single-line base64 (e.g using 'openssl base64 -A -a')
echo ${ALBUM_RECOMMENDATION_JAVA_DEPLOY_KEY} | openssl base64 -A -a -d -out ${DEPLOY_KEY_FILE}

# Retrieve Vespa version
mvn \
  -Dtenant=vespa \
  -Dapplication=album-recommendation \
  -Dendpoint=${VESPA_CLOUD_ENDPOINT} \
  -DapiKeyFile=${DEPLOY_KEY_FILE} \
  clean vespa:compileVersion

# Package and submit application
mvn -Dvespaversion=$(cat target/vespa.compile.version) \
  -Dtenant=vespa \
  -Dapplication=album-recommendation \
  -Dendpoint=${VESPA_CLOUD_ENDPOINT} \
  -Drepository=$(git config --get remote.origin.url) \
  -Dbranch=$(git rev-parse --abbrev-ref HEAD) \
  -Dcommit=$(git rev-parse HEAD) \
  -DauthorEmail=$(git log -1 --format=%aE) \
  -DapiKeyFile=${DEPLOY_KEY_FILE} \
  package vespa:submit
