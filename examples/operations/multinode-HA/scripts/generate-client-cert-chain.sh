#!/usr/bin/env bash
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -e

OUTPUT_DIRECTORY=$(dirname $0)/../pki/client

cat > ${OUTPUT_DIRECTORY}/cert-exts.cnf << EOF
[req]
distinguished_name=req

[ca_exts]
basicConstraints       = critical, CA:TRUE
keyUsage               = critical, digitalSignature, cRLSign, keyCertSign
subjectKeyIdentifier   = hash
subjectAltName         = email:foo-ca@example.com

[host_exts]
basicConstraints       = critical, CA:FALSE
keyUsage               = critical, digitalSignature, keyAgreement, keyEncipherment
extendedKeyUsage       = serverAuth, clientAuth
subjectKeyIdentifier   = hash
authorityKeyIdentifier = keyid,issuer
subjectAltName         = @host_sans
[host_sans]
DNS.1 = client.hostname
EOF

# Self-signed CA
openssl genrsa -out ${OUTPUT_DIRECTORY}/ca-client.key 2048

openssl req -x509 -new \
    -key ${OUTPUT_DIRECTORY}/ca-client.key \
    -out ${OUTPUT_DIRECTORY}/ca-client.pem \
    -subj '/C=US/L=California/O=ACME Inc/OU=Vespa Sample App Client CA/CN=acme-client-ca.example.com' \
    -config ${OUTPUT_DIRECTORY}/cert-exts.cnf \
    -extensions ca_exts \
    -sha256 \
    -days 10000

# Create private key, CSR and certificate for host. Certificate has DNS SANs for all provided hostnames
openssl genrsa -out ${OUTPUT_DIRECTORY}/client.key 2048

openssl req -new -key ${OUTPUT_DIRECTORY}/client.key -out ${OUTPUT_DIRECTORY}/client.csr \
    -subj '/C=US/L=California/O=ACME Inc/OU=Vespa Sample App Client' \
    -sha256

openssl x509 -req -in ${OUTPUT_DIRECTORY}/client.csr \
    -CA ${OUTPUT_DIRECTORY}/ca-client.pem \
    -CAkey ${OUTPUT_DIRECTORY}/ca-client.key \
    -CAcreateserial \
    -CAserial ${OUTPUT_DIRECTORY}/serial.srl \
    -out ${OUTPUT_DIRECTORY}/client.pem \
    -extfile ${OUTPUT_DIRECTORY}/cert-exts.cnf \
    -extensions host_exts \
    -days 10000 \
    -sha256
