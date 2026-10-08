"""Print supported SRS method contracts; this does not assert dataset/GPU readiness."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.training.methods import METHODS
if __name__ == "__main__":
    print(json.dumps({"schema":"srs18-method-contracts.v1", "methods":METHODS,
        "training_started":False, "readiness":"requires per-run data, dependency and resource preflight"}, indent=2))
