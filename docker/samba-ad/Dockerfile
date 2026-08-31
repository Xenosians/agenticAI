FROM debian:12

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        samba \
        samba-common-bin \
        smbclient \
        winbind \
        krb5-user \
        ldap-utils \
        netcat-openbsd \
        dnsutils \
        procps && \
    rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 389/tcp

ENTRYPOINT ["/entrypoint.sh"]