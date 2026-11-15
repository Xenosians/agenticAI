---
name: proxmox-operator
description: How to read and explain Proxmox VE state, and how guest power actions are approved and confirmed.
triggers: [proxmox, pve, virtual machine, vm, lxc, container host, hypervisor, node storage]
---
When the request is about Proxmox VE:

- Base every statement about the cluster on the Proxmox results in this
  request. If there is no result (not configured, unreachable, refused), say
  that plainly and do not describe the cluster from general knowledge.
- Report what matters first: nodes offline, guests stopped unexpectedly,
  storage above 80% used, failed recent tasks.
- Give numbers with units and the observation time from the result. Memory
  and disk as used / total and percent; uptime in days and hours.
- Name nodes and guests exactly as Proxmox returned them (node name, VM ID
  and name).
- Reading changes nothing. A guest can only be started, shut down, rebooted
  or force-stopped, one exact guest at a time and only after the owner
  approves it. Say a power action happened only when its receipt shows the
  Proxmox task finished and the guest was read back in the expected state;
  an unknown outcome means "check the guest", never "done" or "try again".
  Creating, deleting, migrating or reconfiguring guests, storage or the
  cluster is not available.
- Proxmox is reached over the tailnet (MagicDNS name, HTTPS). Never suggest
  disabling certificate checks or exposing the web UI publicly.
