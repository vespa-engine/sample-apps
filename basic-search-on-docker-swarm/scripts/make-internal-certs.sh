#!/bin/sh
set -e

if [[ -z "$@" ]];
then
  echo "Usage: $(basename $0) hostname1 hostname2 ... hostnameN"
  exit 1
fi

cat > cert-exts.cnf << EOF
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

for (( i=1; i <= "$#"; i++ )); do
  echo "DNS.${i} = ${!i}" >> cert-exts.cnf
done
for (( i=1; i <= 8; i++ )); do
  echo "IP.${i} = 10.0.10.1${i}" >> cert-exts.cnf
done
echo "IP.9 = 127.0.0.1" >> cert-exts.cnf
echo "DNS.$(($# + 1)) = localhost" >> cert-exts.cnf


# Self-signed CA
openssl genrsa -out ca-internal.key 2048

openssl req -x509 -new \
    -key ca-internal.key \
    -out ca-internal.pem \
    -subj '/C=US/L=California/O=ACME Inc/OU=Vespa Sample App Internal CA/CN=acme-ca.example.com' \
    -config cert-exts.cnf \
    -extensions ca_exts \
    -sha256 \
    -days 10000

# Create private key, CSR and certificate for host. Certificate has DNS SANs for all provided hostnames
openssl genrsa -out host.key 2048

openssl req -new -key host.key -out host.csr \
    -subj '/C=US/L=California/O=ACME Inc/OU=Vespa Sample Apps' \
    -sha256

openssl x509 -req -in host.csr \
    -CA ca-internal.pem \
    -CAkey ca-internal.key \
    -CAcreateserial \
    -out host.pem \
    -extfile cert-exts.cnf \
    -extensions host_exts \
    -days 10000 \
    -sha256

