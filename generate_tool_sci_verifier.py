# -*- coding: utf-8 -*-
# build_chembench4k_reasoning_dataset_chatcompletions_v3_ENPROMPT.py

import os
import json
import re
import time
import ujson
from glob import glob
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI


# -------------------------
# Client (your setting)
# -------------------------
client = OpenAI(
    base_url="http://127.0.0.1:3888/v1",
    api_key="sk-********************"
)


# -------------------------
# Retry
# -------------------------
def try_request_with_retries(request_function, max_retries=6, delay=1, *args, **kwargs):
    for i in range(max_retries):
        try:
            return request_function(*args, **kwargs)
        except Exception as e:
            print(f"[request failed] {type(e).__name__}: {e}")
            if i == max_retries - 1:
                return None
            time.sleep(delay * (i + 1))
    return None


# -------------------------
# JSON extract & cleanup
# -------------------------
def remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)


def extract_json_from_text(text: str):
    # Prefer ```json ... ```
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if m:
        js_str = m.group(1)
    else:
        # fallback: first [...] block
        m2 = re.search(r"\[[\s\S]*\]", text)
        if not m2:
            return False, text
        js_str = m2.group(0)

    try:
        js_str = remove_trailing_commas(js_str)
        obj = ujson.loads(js_str)
        if isinstance(obj, list):
            return True, obj
        return False, text
    except Exception:
        return False, text


# -------------------------
# Prompts (ENGLISH, strict step schema)
# -------------------------
SYSTEM_PROMPT = (
    "You are a rigorous chemistry/cheminformatics researcher. "
    "Given a multiple-choice question asking which compounds are commonly used to synthesize a target SMILES, "
    "you must produce an evidence-supported reasoning chain. "
    "You are allowed to propose tool steps (especially runnable Python/RDKit code), "
    "but you cannot execute tools yourself."
)

USER_PROMPT_TEMPLATE = """
Question:
{question}

Options:
A: {A}
B: {B}
C: {C}
D: {D}

I need an evidence-supported reasoning chain. You may use tools such as Python packages (e.g., RDKit) to support the reasoning.

IMPORTANT tool rule:
- If a tool is needed, put ONLY directly runnable code (or an exact query string) in tool_details.
- Do NOT include any explanation, comments, markdown, or extra text inside tool_details.
- I will run the code and provide the outputs to you later; therefore, do NOT fabricate tool outputs.

Output requirement:
Output the answer in STRICT JSON format.
The ROOT must be a JSON ARRAY, where each element is one reasoning step.
Each step MUST include the following fields:
- tool_used: boolean (true/false)
- tool_type: string ("python_code"/"scientific_api"/"none")
- tool_details: string (runnable code ; or empty string if tool_used is false)
- reasoning: string (explain what this step is verifying)

Additional constraints:
- Prefer python_code steps that verify structural compatibility (substructure matching, key functional-group motifs, molecular-weight sanity checks).
- If you can reasonably decide without tools, you may set tool_used=false, but you should still provide evidence-based chemical logic.
- Do NOT output the final choice separately; the chain should make the decision clear in the reasoning steps.

Return STRICT JSON only (no extra text).
""".strip()


# -------------------------
# LLM call (chat.completions)
# -------------------------
def get_reasoning_chain(client: OpenAI, prompt: str, model: str):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1
    )
    text = resp.choices[0].message.content
    time.sleep(0.2)
    return text


# -------------------------
# Resume helper
# -------------------------
def load_done_keys(output_jsonl_path: str):
    done = set()
    if not os.path.exists(output_jsonl_path):
        return done
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = (obj.get("_source_file", ""), obj.get("_item_index", -1))
                done.add(k)
            except Exception:
                continue
    return done


# -------------------------
# Per-item processing
# -------------------------
def process_one_item(client, model, item, source_file, item_index, max_retries):
    prompt = USER_PROMPT_TEMPLATE.format(
        question=item.get("question", ""),
        A=item.get("A", ""),
        B=item.get("B", ""),
        C=item.get("C", ""),
        D=item.get("D", "")
    )

    raw = try_request_with_retries(
        get_reasoning_chain,
        max_retries=max_retries,
        delay=2,
        client=client,
        prompt=prompt,
        model=model
    )

    out = dict(item)
    out["_source_file"] = os.path.basename(source_file)
    out["_item_index"] = item_index
    out["reasoning_chain_raw"] = raw if raw is not None else ""
    out["reasoning_chain_status"] = "success" if raw is not None else "request_failed"
    out["reasoning_chain_parsed"] = []

    if raw is None:
        return out

    ok, parsed = extract_json_from_text(raw)
    if ok and isinstance(parsed, list):
        out["reasoning_chain_parsed"] = parsed
    else:
        out["reasoning_chain_status"] = "parse_error"
        out["reasoning_chain_parsed"] = []
        out["reasoning_chain_parse_error"] = True

    return out


# -------------------------
# Per-file processing
# -------------------------
def process_one_file(client, model, input_path, output_path, max_workers, limit, resume, max_retries):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input file must be a JSON array: {input_path}")

    if limit and limit > 0:
        data = data[:limit]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    done = load_done_keys(output_path) if resume else set()

    with open(output_path, "a", encoding="utf-8") as f_out:
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for idx, item in enumerate(data):
                k = (os.path.basename(input_path), idx)
                if resume and k in done:
                    continue
                futures.append(ex.submit(
                    process_one_item, client, model, item, input_path, idx, max_retries
                ))

            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"Generating reasoning chains: {os.path.basename(input_path)}"):
                res = fut.result()
                f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                f_out.flush()


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=str,
                        default="./dataset/ChemBench4K_raw/dev")
    parser.add_argument("--save-dir", type=str,
                        default="./dataset/ChemBench4K_reasoning/dev")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Per file limit; 0 means all")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--max-retries", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    input_files = sorted(glob(os.path.join(args.input_dir, "*.json")))
    if not input_files:
        raise FileNotFoundError(f"No .json files found under: {args.input-dir}")

    for in_path in input_files:
        out_name = os.path.basename(in_path).replace(".json", f".reasoning.{args.model}.jsonl")
        out_path = os.path.join(args.save_dir, out_name)

        process_one_file(
            client=client,
            model=args.model,
            input_path=in_path,
            output_path=out_path,
            max_workers=args.max_workers,
            limit=args.limit,
            resume=args.resume,
            max_retries=args.max_retries
        )


if __name__ == "__main__":
    main()
