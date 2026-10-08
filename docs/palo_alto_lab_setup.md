# PAN-OS lab setup and acceptance

Status on 2026-10-08: adapter implemented; real firewall not provisioned.
The live preflight correctly fails because no host or API key is configured.
Unit-test fixtures are not a firewall and do not satisfy live acceptance.

## Provisioning prerequisite

Obtain an authorized VM-Series image and matching license/evaluation entitlement
through Palo Alto Networks. The KVM deployment documentation explicitly lists a
VM-Series BYOL license. Do not download third-party repackaged images.

Choose the VM model and PAN-OS version before allocating CPU, RAM, disk and NICs;
use that model's current requirements and the vendor compatibility matrix.
The official KVM guide documents Linux hosts. Merely finding `/dev/kvm` in WSL
does not establish that WSL is a supported deployment platform.

Observed local resources: Windows 11 Home, about 16 GB physical RAM; WSL about
7.4 GiB RAM and 2 GiB swap; RTX 4050 with 6 GB VRAM. `/dev/kvm` exists, but
`qemu-system-x86_64` was not found. Windows available physical RAM fell to about
319 MiB during a Hub load. Do not assume a firewall VM plus AI fits comfortably.
Confirm a supported host and resource budget before installing a hypervisor or
booting an image. A separate lab host avoids sharing this laptop's memory budget.

Official deployment references:

- https://docs.paloaltonetworks.com/vm-series/11-1/vm-series-deployment/set-up-the-vm-series-firewall-on-kvm
- https://docs.paloaltonetworks.com/vm-series/deployment/private-cloud/set-up-the-vm-series-firewall-on-kvm/vm-series-on-kvm-requirements-and-prerequisites

## Connect the actual appliance

1. Deploy the entitled image on the selected supported hypervisor according to
   the vendor guide. Use an isolated lab network. The current integration only
   needs management access, not production traffic interception.
2. Configure management addressing, a trusted TLS certificate, and a dedicated
   API account permitted to read system information. Obtain its API key through
   the appliance's documented API authentication procedure. Keep it out of chat,
   source control, command history, and reports.
3. Set these settings in the AI runtime's existing private `.env`, using the
   actual management origin and key. These lines document names, not working
   values; do not paste the placeholders as credentials.

   ```dotenv
   PALO_ALTO_BASE_URL=https://<actual-management-host>
   PALO_ALTO_API_KEY=<private-key>
   PALO_ALTO_VERIFY_TLS=true
   PALO_ALTO_ALLOW_INSECURE_HTTP=false
   PALO_ALTO_TIMEOUT_SECONDS=10
   ```

4. From the AI repository with the `qwen-infra` environment active, run:

   ```bash
   python scripts/palo_alto_preflight.py
   ```

   Require real hostname, model and PAN-OS version. The script also reports
   provider elapsed time, independently of model inference. Resolve certificate,
   routing or permission failures at their source.
5. Start AI and Phoenix using their existing service instructions. Confirm
   authenticated `GET /api/v1/integrations` reflects the configured integration.
   Configuration alone is not a reachability assertion.
6. Run the Palo Alto read case through the real FE -> Phoenix -> AI -> provider
   flow, checking the displayed values against the appliance. Record job/attempt
   evidence. Repository/unit gates do not certify this step or the complete
   V18-03 restart/stale-attempt and governed mutation requirements.

## Provider response contract

The system-info operational command returns `response/result/system`. The parser
requires that structure and nonempty hostname, model and software version; it
does not return success with missing identity fields.

Source: Palo Alto Networks' SDK `show_system_info()` and `_save_system_info()`:
https://github.com/PaloAltoNetworks/pan-os-python/blob/develop/panos/base.py

## Performance diagnosis

```bash
python scripts/inference_latency_probe.py --model-key hub-main --runs 2
```

This diagnostic loads the configured model and measures generation through the
existing scheduler. It does not route prompts or execute tools. It reports load,
queue, generation, token count/rate and CUDA memory, preserving unknown TTFT as
null. It writes no generated text. Compare it with provider preflight timing;
neither replaces complete job-stage timing or semantic acceptance.
