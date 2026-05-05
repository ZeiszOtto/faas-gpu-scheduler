#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
info() { echo -e "${YELLOW}[..] $1${NC}"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Root
if [ "$EUID" -ne 0 ]; then
  err "Please use root privilages to run the script: sudo bash $0"
fi

echo ""
echo "============================================="
echo "  Containerd installation and configuration"
echo "============================================="
echo ""

# ===================================================================================================
# 1. Kernel modulok
info "Configuring kernel modules..."

cat <<EOF > /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

modprobe overlay      || err "overlay module failed to load"
modprobe br_netfilter || err "br_netfilter module failed to load"

lsmod | grep -q "^overlay"      || err "overlay module is inactive"
lsmod | grep -q "^br_netfilter" || err "br_netfilter module is inactive"

ok "Kernel modules loaded (overlay, br_netfilter)."


# ===================================================================================================
# 2. Sysctl beállítások
info "Applying sysctl network settings..."

cat <<EOF > /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sysctl --system > /dev/null 2>&1

sysctl net.bridge.bridge-nf-call-iptables  | grep -q "= 1" || err "bridge-nf-call-iptables not 1"
sysctl net.bridge.bridge-nf-call-ip6tables | grep -q "= 1" || err "bridge-nf-call-ip6tables not 1"
sysctl net.ipv4.ip_forward                 | grep -q "= 1" || err "ip_forward not 1"

ok "Sysctl settings applied."


# ===================================================================================================
# Swap kikapcsolása
info "Disabling swap..."

swapoff -a

if grep -q "swap" /etc/fstab; then
  cp /etc/fstab /etc/fstab.bak
  sed -i '/swap/d' /etc/fstab
  ok "Swap removed from /etc/fstab (Recovery: /etc/fstab.bak)"
else
  ok "Swap not found in /etc/fstab-ban, removal not needed"
fi

SWAP_TOTAL=$(free | awk '/^Swap:/ {print $2}')
if [ "$SWAP_TOTAL" -eq 0 ]; then
  ok "Swap disabled."
else
  err "Swap still active, manual check needed."
fi


# ===================================================================================================
# Containerd telepítése
info "Upgrading packages.."
apt-get update -q

info "Installing containerd..."
apt-get install -y -q containerd

ok "Containerd installed successfully."


# ===================================================================================================
# Containerd konfig
info "Configuring containerd..."

mkdir -p /etc/containerd
containerd config default > /etc/containerd/config.toml

if grep -q "SystemdCgroup = false" /etc/containerd/config.toml; then
  sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
  ok "SystemdCgroup = true set."
else
  grep -q "SystemdCgroup = true" /etc/containerd/config.toml \
    && ok "SystemdCgroup already true." \
    || err "SystemdCgroup line not found in the configuration file, manual check needed."
fi


# ===================================================================================================
# Containerd restart
info "Restarting services..."

systemctl restart containerd
systemctl enable containerd > /dev/null 2>&1

sleep 2
systemctl is-active --quiet containerd \
  && ok "Setup complete — containerd active." \
  || err "containerd inactive — run 'journalctl -u containerd' for logs."
