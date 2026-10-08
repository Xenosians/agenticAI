"""Measure real tokenizer cost without loading model weights or printing prompts."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import Settings
from transformers import AutoTokenizer
from learning.training.hub_training_contracts import build_hub_training_environment
from subagents.prompts.prompt_loader import load_prompt


def main():
    settings = Settings()
    env = build_hub_training_environment(agent_directory=ROOT / "subagents/agents")
    legacy = load_prompt("hub_router.txt").replace("{{SPECIALISTS_JSON}}", json.dumps(env["specialists"], indent=2)) + "\n\n" + load_prompt("hub_conditional_workflow.txt")
    tokenizer = AutoTokenizer.from_pretrained(settings.require_model_profile(settings.hub_model_key).model_path, local_files_only=True)
    counts = {name:len(tokenizer.encode(prompt, add_special_tokens=False))
        for name,prompt in [("legacy_tokens", legacy), ("deduplicated_tokens", env["system_prompt"]) ]}
    report = {"schema":"router-prompt-budget.v1", **counts, "specialists":len(env["specialists"]),
        "reduction_fraction":1-counts["deduplicated_tokens"]/counts["legacy_tokens"],
        "capability_catalog_sha256":env["capability_catalog_sha256"], "system_prompt_sha256":env["system_prompt_sha256"]}
    path = ROOT / ".runtime/router-prompt-budget.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
