#!/usr/bin/env bash

set -euo pipefail

SAMBA_DOMAIN="${SAMBA_DOMAIN:-ITSM}"
SAMBA_REALM="${SAMBA_REALM:-itsm.local}"
SAMBA_NETBIOS="${SAMBA_NETBIOS:-ITSMDC}"
SAMBA_ADMIN_PASS="${SAMBA_ADMIN_PASS:-ItsmAdminDevOnly#2026}"

echo "======================================"
echo "Samba AD Development Domain Controller"
echo "======================================"
echo "Domain: ${SAMBA_DOMAIN}"
echo "Realm:  ${SAMBA_REALM}"
echo

if [ ! -f /var/lib/samba/private/sam.ldb ]; then
    echo "[Samba] Provisioning domain..."

    rm -f /etc/samba/smb.conf

    samba-tool domain provision \
        --server-role=dc \
        --use-rfc2307 \
        --realm="${SAMBA_REALM^^}" \
        --domain="${SAMBA_DOMAIN^^}" \
        --host-name="${SAMBA_NETBIOS}" \
        --dns-backend=SAMBA_INTERNAL \
        --adminpass="${SAMBA_ADMIN_PASS}"

    echo "[Samba] Domain provisioned."
else
    echo "[Samba] Existing domain detected."
fi

echo "[Samba] Starting Samba..."

samba -D

echo "[Samba] Waiting for LDAP..."

READY=false

for i in $(seq 1 30); do
    if nc -z 127.0.0.1 389; then
        READY=true
        echo "[Samba] LDAP ready."
        break
    fi

    sleep 1
done

if [ "$READY" != "true" ]; then
    echo "[Samba] LDAP failed to start."
    exit 1
fi

echo "[Samba] Creating development users..."

if ! samba-tool user show jdoe >/dev/null 2>&1; then
    samba-tool user create jdoe 'JdoeDev#2026'
fi

if ! samba-tool user show asmith >/dev/null 2>&1; then
    samba-tool user create asmith 'AsmithDev#2026'
fi

if ! samba-tool user show disabled >/dev/null 2>&1; then
    samba-tool user create disabled 'DisabledDev#2026'
    samba-tool user disable disabled
fi

echo "[Samba] Creating groups..."

if ! samba-tool group show "VPN-Users" >/dev/null 2>&1; then
    samba-tool group add "VPN-Users"
fi

if ! samba-tool group show "ITSM-Admins" >/dev/null 2>&1; then
    samba-tool group add "ITSM-Admins"
fi

samba-tool group addmembers "VPN-Users" jdoe >/dev/null 2>&1 || true
samba-tool group addmembers "VPN-Users" asmith >/dev/null 2>&1 || true
samba-tool group addmembers "ITSM-Admins" asmith >/dev/null 2>&1 || true

echo
echo "[Samba] Development AD ready."
echo
echo "Users:"
samba-tool user list

echo
echo "Groups:"
samba-tool group list

echo

exec tail -F /var/log/samba/log.samba