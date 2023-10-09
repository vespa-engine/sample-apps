#!/usr/bin/env bash
# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
set -e

OUTPUT_DIRECTORY=$(dirname $0)/../pki/vespa
VESPA_HOSTNAMES=( $(echo node{0..9}.vespanet) )

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
EOF

for (( i=1; i <= "${#VESPA_HOSTNAMES[@]}"; i++ )); do
  echo "DNS.${i} = ${VESPA_HOSTNAMES[i - 1]}" >> ${OUTPUT_DIRECTORY}/cert-exts.cnf
done
for (( i=0; i <= 9; i++ )); do
  echo "IP.${i} = 10.0.10.1${i}" >> ${OUTPUT_DIRECTORY}/cert-exts.cnf
done
echo "IP.10 = 127.0.0.1" >> ${OUTPUT_DIRECTORY}/cert-exts.cnf
echo "DNS.$((${#VESPA_HOSTNAMES[@]} + 1)) = localhost" >> ${OUTPUT_DIRECTORY}/cert-exts.cnf


# Self-signed CA
openssl genrsa -out ${OUTPUT_DIRECTORY}/ca-vespa.key 2048

openssl req -x509 -new \
    -key ${OUTPUT_DIRECTORY}/ca-vespa.key \
    -out ${OUTPUT_DIRECTORY}/ca-vespa.pem \
    -subj '/C=US/L=California/O=ACME Inc/OU=Vespa Sample App Internal CA/CN=acme-vespa-ca.example.com' \
    -config ${OUTPUT_DIRECTORY}/cert-exts.cnf \
    -extensions ca_exts \
    -sha256 \
    -days 10000

# Create private key, CSR and certificate for host. Certificate has DNS SANs for all provided hostnames
openssl genrsa -out ${OUTPUT_DIRECTORY}/host.key 2048

openssl req -new -key ${OUTPUT_DIRECTORY}/host.key -out ${OUTPUT_DIRECTORY}/host.csr \
    -subj '/C=US/L=California/O=ACME Inc/OU=Vespa Sample Apps' \
    -sha256

openssl x509 -req -in ${OUTPUT_DIRECTORY}/host.csr \
    -CA ${OUTPUT_DIRECTORY}/ca-vespa.pem \
    -CAkey ${OUTPUT_DIRECTORY}/ca-vespa.key \
    -CAcreateserial \
    -CAserial ${OUTPUT_DIRECTORY}/serial.srl \
    -out ${OUTPUT_DIRECTORY}/host.pem \
    -extfile ${OUTPUT_DIRECTORY}/cert-exts.cnf \
    -extensions host_exts \
    -days 10000 \
    -sha256
