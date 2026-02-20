# -*- coding: utf-8 -*-
# chembench4k_two_stage.py
#
# Stage 1: call LLM only, save reasoning jsonl
# Stage 2: execute python_code steps from stage1 jsonl, save executed jsonl
#
# Usage examples:
#   # stage 1 only
#   python chembench4k_two_stage.py --stage llm \
#     --input-dir .../ChemBench4K_raw/dev \
#     --save-dir  .../ChemBench4K_reasoning/dev \
#     --model Qwen2.5-72B-Instruct --max-workers 8 --resume
#
#   # stage 2 only
#   python chembench4k_two_stage.py --stage exec \
#     --reasoning-jsonl .../ChemBench4K_reasoning/dev/xxx.reasoning.Qwen2.5-72B-Instruct.jsonl \
#     --executed-jsonl  .../ChemBench4K_reasoning_executed/dev/xxx.executed.Qwen2.5-72B-Instruct.jsonl \
#     --max-workers 8 --resume
#
# Notes:
# - Executes python_code steps only, using exec() in-process; run in a controlled environment.

import os
import json
import re
import time
import ujson
import io
import contextlib
import traceback
from glob import glob
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI


# -------------------------
# Client (your setting)
# -------------------------
def build_client():
    return OpenAI(
        base_url="http://127.0.0.1:3888/v1",
        api_key="sk-********************"
    )


# -------------------------
# Retry helper
# -------------------------
def try_request_with_retries(request_function, max_retries=8, delay=2, *args, **kwargs):
    for i in range(max_retries):
        try:
            return request_function(*args, **kwargs), None
        except Exception as e:
            if i == max_retries - 1:
                return None, f"{type(e).__name__}: {e}"
            time.sleep(delay * (i + 1))
    return None, "unknown_error"


# -------------------------
# JSON extract & cleanup
# -------------------------
def remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)


def extract_json_array_from_text(text: str):
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if m:
        js_str = m.group(1)
    else:
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
# Prompt (ENGLISH, your schema)
# -------------------------
SYSTEM_PROMPT = (
    "You are a rigorous chemistry/cheminformatics researcher. "
    "Given a multiple-choice question asking which compounds are commonly used to synthesize a target SMILES, "
    "produce an evidence-supported reasoning chain. "
    "You may propose tool steps (especially runnable Python/RDKit code), but you cannot execute tools yourself. "
    "If you need computation, consolidate everything into less Python code block. "
    "All other steps must have tool_type 'none'."
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

TOOL RULES (STRICT):
- If a tool is needed, put ONLY directly runnable code (or an exact query string) in tool_details.
- Do NOT include any explanation, comments, markdown, or extra text inside tool_details.
- I will run the tool and provide outputs later; therefore:
  - Do NOT fabricate tool outputs.
  - Do NOT provide a final conclusion about which option is correct until tool outputs are provided (if tool use is required).

PYTHON_CODE RULES:
- If tool_type is "python_code", tool_details MUST be a single runnable Python snippet.
- The code MUST end with at least one print(...) that prints the final computed evidence (e.g., a dict/list/table as text).
- Do not rely on implicit/interactive display of variables.

OUTPUT FORMAT (STRICT):
Return STRICT JSON only (no extra text).
The root MUST be a JSON ARRAY; each element is one reasoning step.
Each step MUST include:
- step_id: number
- tool_used: boolean
- tool_type: string ("python_code" / "scientific_api" / "none")
- tool_details: string (runnable code / exact query; empty string if tool_used is false)
- reasoning: string (what the step checks and how it supports discriminating among options; do not invent tool results)
""".strip()


# -------------------------
# LLM call (chat.completions)
# -------------------------
def call_llm_reasoning_chain(client: OpenAI, model: str, prompt: str):
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
# Execute python tool_details (capture stdout)
# -------------------------
def exec_python_capture_stdout(code: str) -> str:
    stdout = io.StringIO()
    glb = {}
    loc = {}
    with contextlib.redirect_stdout(stdout):
        exec(code, glb, loc)
    return stdout.getvalue().strip()


def execute_steps(steps):
    executed = []
    for st in steps:
        st2 = dict(st)

        if "tool_used" not in st2:
            st2["tool_used"] = False
        if "tool_type" not in st2:
            st2["tool_type"] = "none"
        if "tool_details" not in st2:
            st2["tool_details"] = ""
        if "reasoning" not in st2:
            st2["reasoning"] = ""

        st2["tool_output"] = ""

        if st2.get("tool_used") and st2.get("tool_type") == "python_code":
            code = st2.get("tool_details", "") or ""
            try:
                out = exec_python_capture_stdout(code)
                st2["tool_output"] = out
                st2["python_exec_status"] = "success"
            except Exception as e:
                st2["python_exec_status"] = "error"
                st2["python_exec_error"] = f"{type(e).__name__}: {e}"
                st2["python_exec_traceback"] = traceback.format_exc()

        executed.append(st2)
    return executed


# -------------------------
# Resume helpers
# -------------------------
def load_done_keys(jsonl_path: str):
    done = set()
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add((obj.get("_source_file", ""), obj.get("_item_index", -1)))
            except Exception:
                continue
    return done


# -------------------------
# Stage 1: LLM only
# -------------------------
def process_one_item_llm(client, model, item, source_file, item_index, max_retries):
    out = dict(item)
    out["_source_file"] = os.path.basename(source_file)
    out["_item_index"] = item_index

    prompt = USER_PROMPT_TEMPLATE.format(
        question=item.get("question", ""),
        A=item.get("A", ""),
        B=item.get("B", ""),
        C=item.get("C", ""),
        D=item.get("D", "")
    )

    raw, err = try_request_with_retries(
        call_llm_reasoning_chain,
        max_retries=max_retries,
        delay=2,
        client=client,
        model=model,
        prompt=prompt
    )

    out["reasoning_chain_raw"] = raw if raw is not None else ""
    out["reasoning_chain_status"] = "success" if raw is not None else f"request_failed: {err}"
    out["reasoning_chain_parsed"] = []

    if raw is None:
        return out

    ok, steps = extract_json_array_from_text(raw)
    if not ok:
        out["reasoning_chain_status"] = "parse_error"
        out["reasoning_chain_parse_error_raw"] = raw
        return out

    if not isinstance(steps, list):
        out["reasoning_chain_status"] = "parse_error_not_list"
        return out

    out["reasoning_chain_parsed"] = steps
    return out


def stage_llm_on_file(client, model, input_path, output_jsonl_path,
                      max_workers, limit, resume, max_retries):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input file must be a JSON array: {input_path}")

    if limit and limit > 0:
        data = data[:limit]

    os.makedirs(os.path.dirname(output_jsonl_path) or ".", exist_ok=True)
    done = load_done_keys(output_jsonl_path) if resume else set()

    with open(output_jsonl_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for idx, item in enumerate(data):
                k = (os.path.basename(input_path), idx)
                if resume and k in done:
                    continue
                futures.append(ex.submit(
                    process_one_item_llm, client, model, item, input_path, idx, max_retries
                ))

            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"LLM: {os.path.basename(input_path)}"):
                res = fut.result()
                f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                f_out.flush()


def stage_llm_on_dir(client, model, input_dir, save_dir,
                     max_workers, limit, resume, max_retries, file_topk=3):
    os.makedirs(save_dir, exist_ok=True)
    input_files = sorted(glob(os.path.join(input_dir, "*.json")))[:file_topk]
    if not input_files:
        raise FileNotFoundError(f"No .json files found under: {input_dir}")

    for in_path in input_files:
        out_name = os.path.basename(in_path).replace(".json", f".reasoning.{model}.jsonl")
        out_path = os.path.join(save_dir, out_name)
        stage_llm_on_file(
            client=client,
            model=model,
            input_path=in_path,
            output_jsonl_path=out_path,
            max_workers=max_workers,
            limit=limit,
            resume=resume,
            max_retries=max_retries
        )


# -------------------------
# Stage 2: Exec only (read stage1 jsonl)
# -------------------------
def process_one_item_exec(obj):
    out = dict(obj)
    steps = out.get("reasoning_chain_parsed", [])
    if not isinstance(steps, list):
        out["reasoning_chain_executed"] = []
        out["exec_status"] = "no_steps"
        return out

    out["reasoning_chain_executed"] = execute_steps(steps)
    out["exec_status"] = "success"
    return out


def stage_exec_jsonl(reasoning_jsonl_path, executed_jsonl_path, max_workers, resume):
    os.makedirs(os.path.dirname(executed_jsonl_path) or ".", exist_ok=True)
    done = load_done_keys(executed_jsonl_path) if resume else set()

    # read all inputs
    items = []
    with open(reasoning_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            k = (obj.get("_source_file", ""), obj.get("_item_index", -1))
            if resume and k in done:
                continue
            items.append(obj)

    with open(executed_jsonl_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_one_item_exec, obj) for obj in items]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"EXEC: {os.path.basename(reasoning_jsonl_path)}"):
                res = fut.result()
                f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                f_out.flush()


def main():
    parser = ArgumentParser()
    parser.add_argument("--stage", type=str, choices=["llm", "exec"], required=True)

    # stage llm args
    parser.add_argument("--input-dir", type=str,
                        default="./dataset/ChemBench4K_raw/test")
    parser.add_argument("--save-dir", type=str,
                        default="./dataset/ChemBench4K_reasoning/test")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--limit", type=int, default=200, help="Per file limit; 0 means all")
    parser.add_argument("--file-topk", type=int, default=9, help="How many input json files to process")

    # stage exec args
    parser.add_argument("--reasoning-jsonl", type=str, default="./dataset/ChemBench4K_reasoning/dev/Mol2caption_benchmark.reasoning.gemini-3-flash-preview.jsonl")
    parser.add_argument("--executed-jsonl", type=str, default="./dataset/ChemBench4K_reasoning/dev/Mol2caption_benchmark.excute.gpt-5.2.jsonl")

    # common
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", default=False)

    args = parser.parse_args()

    if args.stage == "llm":
        client = build_client()
        stage_llm_on_dir(
            client=client,
            model=args.model,
            input_dir=args.input_dir,
            save_dir=args.save_dir,
            max_workers=args.max_workers,
            limit=args.limit,
            resume=args.resume,
            max_retries=args.max_retries,
            file_topk=args.file_topk
        )

    elif args.stage == "exec":
        if not args.reasoning_jsonl or not args.executed_jsonl:
            raise ValueError("--reasoning-jsonl and --executed-jsonl are required for stage=exec")
        stage_exec_jsonl(
            reasoning_jsonl_path=args.reasoning_jsonl,
            executed_jsonl_path=args.executed_jsonl,
            max_workers=args.max_workers,
            resume=args.resume
        )


if __name__ == "__main__":
    main()
