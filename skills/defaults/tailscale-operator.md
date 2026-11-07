---
name: tailscale-operator
description: How to read and explain tailnet devices, DNS and reachability.
triggers: [tailscale, tailnet, magicdns, magic dns, exit node, tailscale serve, subnet router]
---
When the request is about the tailnet:

- Use the tailnet results in this request (devices, online state, DNS). If a
  read failed, say so; do not guess a device's state.
- Refer to devices by their MagicDNS name, not by tailnet IP address, and do
  not repeat full 100.x addresses unless the user asks for them.
- For "can't reach X" questions, check in order: is the device online, does
  its name resolve through MagicDNS, is the service listening, do access rules
  allow it. Say which step the evidence points to.
- Tailscale Serve shares a service inside the tailnet only; Funnel exposes it
  to the public internet. Recommend Serve for internal services; mention
  Funnel only when the user explicitly wants public access, with the risk.
- Never print auth keys, OAuth secrets or node keys.
