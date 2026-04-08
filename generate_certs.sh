#!/bin/bash

set -e

# Configuration
SERVICE_NAME="gpu-scheduler-webhook"
NAMESPACE="gpu-scheduler"
SECRET_NAME="gpu-scheduler-tls"
CERT_DIR="./certs"
WEBHOOK_CONFIG="${WEBHOOK_CONFIG:-$(pwd)/scheduler-config.yaml}"

mkdir -p "${CERT_DIR}"
cd "${CERT_DIR}"

echo "[INFO] Generating CA private key..."
openssl genrsa -out ca.key 2048

echo "[INFO] Generating CA certificate..."
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 \
    -out ca.crt \
    -subj "/CN=gpu-scheduler-ca"

echo "[INFO] Generating server private key..."
openssl genrsa -out tls.key 2048

# OpenSSL Config
cat > csr.conf <<EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = req_ext

[dn]
CN = ${SERVICE_NAME}.${NAMESPACE}.svc

[req_ext]
subjectAltName = @alt_names

[alt_names]
DNS.1 = ${SERVICE_NAME}
DNS.2 = ${SERVICE_NAME}.${NAMESPACE}
DNS.3 = ${SERVICE_NAME}.${NAMESPACE}.svc
DNS.4 = ${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local
EOF

echo "[INFO] Generating Certificate Signing Request..."
openssl req -new -key tls.key -out tls.csr -config csr.conf

echo "[INFO] Signing the CSR with the CA..."
openssl x509 -req -in tls.csr \
    -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out tls.crt -days 365 -sha256 \
    -extensions req_ext -extfile csr.conf

# Verify certificate
echo ""
echo "[INFO] Certificate details:"
openssl x509 -in tls.crt -noout -subject -issuer
openssl x509 -in tls.crt -noout -ext subjectAltName

# Create the namespace if it does not exist
echo ""
echo "[INFO] Ensuring namespace ${NAMESPACE} exists..."
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

# Create the Kubernetes Secret
echo "[INFO] Creating Secret ${SECRET_NAME} in namespace ${NAMESPACE}..."
kubectl create secret tls "${SECRET_NAME}" \
    --cert=tls.crt \
    --key=tls.key \
    --namespace="${NAMESPACE}" \
    --dry-run=client -o yaml | kubectl apply -f -

# Inject the CA bundle into the MutatingWebhookConfiguration manifest if possible
echo ""
if [ -f "${WEBHOOK_CONFIG}" ]; then
    echo "[INFO] Injecting CA bundle into ${WEBHOOK_CONFIG}..."
    sed -i.bak "s|caBundle:.*|caBundle: ${CA_BUNDLE}|" "${WEBHOOK_CONFIG}"
    rm -f "${WEBHOOK_CONFIG}.bak"
    echo "[INFO] CA bundle successfully written to ${WEBHOOK_CONFIG}"
else
    echo "[WARNING] ${WEBHOOK_CONFIG} not found — printing CA bundle for manual paste:"
    echo ""
    echo "${CA_BUNDLE}"
fi

echo ""
echo "[INFO] Done. Certificate files are in ${CERT_DIR}/"