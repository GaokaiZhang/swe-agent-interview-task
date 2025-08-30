# scripts/select_full10.py
import json, random, os
from datasets import load_dataset

def main():
    os.makedirs("data", exist_ok=True)
    
    ds = load_dataset("princeton-nlp/SWE-bench", split="test")
   
    ds = ds.sort("instance_id")
    ids = [ex["instance_id"] for ex in ds]
    random.seed(42)
    pick = sorted(random.sample(ids, 10)) # random picking 10
    with open("data/ids_full10.json", "w") as f:
        json.dump(pick, f, indent=2)
    print("Wrote data/ids_full10.json with", len(pick), "ids")

if __name__ == "__main__":
    main()
