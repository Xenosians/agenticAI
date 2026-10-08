"""Train a reviewed local SFT or causal-adaptation recipe into an isolated candidate."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learning.training.recipes import AdaptationRecipe, admit_records, train_recipe

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--allow-training", action="store_true")
    args = parser.parse_args()
    recipe = AdaptationRecipe.model_validate_json(args.recipe.read_text())
    train, validation = admit_records(recipe)
    if not args.allow_training:
        print(json.dumps({"status": "dataset_admission_passed", "train": len(train), "validation": len(validation), "model_loaded": False}))
        return
    print(train_recipe(recipe, allow_training=True))

if __name__ == "__main__":
    main()
