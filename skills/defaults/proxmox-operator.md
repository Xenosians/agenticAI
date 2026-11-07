---
name: proxmox-operator
description: How to read and explain Proxmox VE cluster, node, guest, storage and task state.
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
- Changes to VMs, containers, storage or the cluster are not done by reading.
  When the user wants a change, say which action it would be and that it
  needs their approval; never claim a change was made.
- Proxmox is reached over the tailnet (MagicDNS name, HTTPS). Never suggest
  disabling certificate checks or exposing the web UI publicly.
