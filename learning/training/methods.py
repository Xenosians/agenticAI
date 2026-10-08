"""SRS 1.8 method/role compatibility, distinct from available hardware or data."""
METHODS = {
    "causal_adaptation": {"roles": ["hub", "specialist"], "tuning": ["full", "lora", "qlora"], "entrypoint": "scripts/train_recipe.py", "requires": "reviewed licensed text with independent validation"},
    "sft": {"roles": ["hub", "specialist"], "tuning": ["full", "lora", "qlora"], "entrypoint": "scripts/train_recipe.py", "requires": "reviewed messages with assistant targets and independent validation"},
    "dpo": {"roles": ["specialist"], "tuning": ["qlora"], "entrypoint": "scripts/train_preferences.py", "requires": "verified reviewed preference partitions and matching base artifact"},
    "ppo": {"roles": ["hub"], "tuning": ["qlora"], "entrypoint": "scripts/train_ppo.py", "requires": "verified adapter checkpoint and independent simulator episodes"},
    "rag": {"roles": ["hub", "specialist"], "tuning": [], "entrypoint": "services/knowledge/local.py", "requires": "reviewed public knowledge snapshot and hash; no weight training"},
    "continual": {"roles": ["hub", "specialist"], "tuning": [], "entrypoint": "learning/cli/run_continual_cycle.py", "requires": "reviewed evidence, replay, candidate evaluation and separate promotion"},
}


def validate_method_role(method: str, role: str, tuning: str | None = None):
    contract = METHODS.get(method)
    if contract is None or role not in contract["roles"]:
        raise ValueError("Unsupported training method/role combination.")
    if tuning is not None and tuning not in contract["tuning"]:
        raise ValueError("Unsupported method/tuning combination.")
    return contract
