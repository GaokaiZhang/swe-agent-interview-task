# scripts/run_agent_api.py
import os, json, re, time
from datasets import load_dataset
from tqdm import tqdm
from anthropic import Anthropic, APIStatusError

API_KEY = os.environ.get("CLAUDE_API") or os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    raise SystemExit("ERROR: set CLAUDE_API or ANTHROPIC_API_KEY first.")

MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")
OUT_PATH = "predictions/predictions.jsonl"
LOG_DIR = "logs"

SYSTEM = (
    "You are a senior software engineer. "
    "Generate ONLY a valid unified diff patch that fixes the described issue. "
    "Do not include any commentary outside the diff."
)

USER_TMPL = """\
Instance ID: {iid}
Repository:  {repo}

<issue>
{problem}
</issue>

Constraints:
- Provide a minimal, correct unified diff that will pass tests.
- Do NOT modify tests unless strictly required.
- Ensure file paths in the diff match the repository layout.

Return the diff wrapped in <patch>...</patch>.
"""

def extract_patch(text: str) -> str:
    m = re.search(r"<patch>([\s\S]*?)</patch>", text)
    return m.group(1).strip() if m else text.strip()

def main():
    os.makedirs("predictions", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    ids = json.load(open("data/ids_full10.json"))
    ds = load_dataset("princeton-nlp/SWE-bench", split="test")
    by_id = {ex["instance_id"]: ex for ex in ds if ex["instance_id"] in set(ids)}

    client = Anthropic(api_key=API_KEY)

    with open(OUT_PATH, "w", encoding="utf-8") as fout:
        for iid in tqdm(ids, desc="Generating patches"):
            ex = by_id[iid]
            prompt = USER_TMPL.format(
                iid=iid,
                repo=ex.get("repo", ""),
                problem=ex.get("problem_statement", "")
            )
            try:
                resp = client.messages.create(
                    model=MODEL,
                    system=SYSTEM,
                    max_tokens=4000,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                
                content = "".join([c.text for c in resp.content if c.type == "text"])
                patch = extract_patch(content)
            except APIStatusError as e:
                
                time.sleep(0.5)
                resp = client.messages.create(
                    model=MODEL,
                    system=SYSTEM,
                    max_tokens=4000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = "".join([c.text for c in resp.content if c.type == "text"])
                patch = extract_patch(content)

            # original log
            with open(f"{LOG_DIR}/{iid}.log", "w", encoding="utf-8") as lf:
                lf.write("PROMPT:\n" + prompt + "\n\nRAW_RESPONSE:\n" + content + "\n")

            rec = {
                "instance_id": iid,
                "model_name_or_path": MODEL,
                "model_patch": patch
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Saved ->", OUT_PATH)

if __name__ == "__main__":
    main()
