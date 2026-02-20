# -*- coding: utf-8 -*-
# physics_two_stage.py
#
# Stage 1: call LLM only, save reasoning jsonl (Filters for language='en')
# Stage 2: execute python_code steps from stage1 jsonl, save executed jsonl

import os
import json
import re
import time
import ujson
import io
import contextlib
import tempfile
import traceback
import subprocess
import sys
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
# Prompt (Modified for Physics / General QA)
# -------------------------
SYSTEM_PROMPT = (
    "You are a rigorous physics and scientific researcher. "
    "Given a physics problem, produce an evidence-supported reasoning chain to solve it. "
    "You may propose tool steps (especially runnable Python code for calculation), but you cannot execute tools yourself. "
    "If you need computation, consolidate everything into fewer Python code blocks. "
    "All other steps must have tool_type 'none'."
)

# Removed A/B/C/D options since the physics dataset is open-ended/numerical
USER_PROMPT_TEMPLATE = """
Question:
{question}

I need an evidence-supported reasoning chain to solve this problem. You may use tools such as Python packages to support the calculation and you need to give me the final results.

TOOL RULES (STRICT):
- If a tool is needed, put ONLY directly runnable code (or an exact query string) in tool_details.
- Do NOT include any explanation, comments, markdown, or extra text inside tool_details.

PYTHON_CODE RULES:
- If tool_type is "python_code", tool_details MUST be a single runnable Python snippet.
- The code MUST end with at least one print(...) that prints the final computed evidence (e.g., a number or dict).
- Do not rely on implicit/interactive display of variables.

OUTPUT FORMAT (STRICT):
Return STRICT JSON only (no extra text).
The root MUST be a JSON ARRAY; each element is one reasoning step.
Each step MUST include:
- step_id: number
- tool_used: boolean
- tool_type: string ("python_code" / "scientific_api" / "none")
- tool_details: string (runnable code / exact query; empty string if tool_used is false)
- reasoning: string (what the step checks/calculates and how it contributes to the solution)
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
# Execute python tool_details (Subprocess for safety)
# -------------------------
def exec_python_capture_stdout(code: str) -> str:
    """
    Executes Python code in a separate subprocess to ensure thread safety 
    and capture stdout correctly.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    
    try:
        # timeout=300 prevents infinite loops
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\n[Stderr]:\n{result.stderr}"
            
    except subprocess.TimeoutExpired:
        output = "Error: Execution timed out (exceeded 300s)."
    except Exception as e:
        output = f"Execution Error: {str(e)}"
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
    
    return output.strip()


def execute_steps(steps):
    # --- FIX START: Defensive processing for 'steps' container ---
    if steps is None:
        steps = []
    elif isinstance(steps, dict):
        steps = [steps]
    elif not isinstance(steps, list):
        # If it's neither list nor dict (e.g. string), return empty
        return []
    # --- FIX END ---

    executed = []
    for st in steps:
        # --- FIX START: Defensive processing for individual step items ---
        # The error happened here because 'st' was a string, not a dict.
        if not isinstance(st, dict):
            # If the item is not a dict (e.g. a string from a malformed LLM response),
            # wrap it into a safe dictionary structure so the script doesn't crash.
            st = {
                "step_id": -1,
                "tool_used": False,
                "tool_type": "none",
                "tool_details": "",
                "reasoning": str(st) # Save the content as reasoning
            }
            print(1)
        # --- FIX END ---

        st2 = dict(st) # Now this is safe

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
                # Use 'id' from the physics dataset as unique identifier if available, else index
                # Combining filename + id/index
                item_id = obj.get("id", obj.get("_item_index"))
                done.add((obj.get("_source_file", ""), item_id))
            except Exception:
                continue
    return done


# -------------------------
# Stage 1: LLM only
# -------------------------
def process_one_item_llm(client, model, item, source_file, item_index, max_retries):
    out = dict(item)
    out["_source_file"] = os.path.basename(source_file)
    # Use the dataset 'id' if available, otherwise use enumeration index
    out["_item_index"] = item.get("id", item_index)

    # Modified: Just pass the question, no Options A/B/C/D
    prompt = USER_PROMPT_TEMPLATE.format(
        question=item.get("question", "")
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


def stage_llm_main(client, model, input_file, output_jsonl_path,
                   max_workers, limit, resume, max_retries):
    """
    Reads a single JSONL file, filters for language='en', and processes.
    """
    data = []
    print(f"Loading data from {input_file}...")
    
    # Read JSONL line by line
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # --- FILTERING LOGIC HERE ---
                if obj.get("language") == "en":
                    data.append(obj)
            except json.JSONDecodeError:
                continue

    print(f"Total English items found: {len(data)}")

    if limit and limit > 0:
        data = data[:limit]
        print(f"Limiting to first {limit} items.")

    os.makedirs(os.path.dirname(output_jsonl_path) or ".", exist_ok=True)
    done = load_done_keys(output_jsonl_path) if resume else set()

    with open(output_jsonl_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for idx, item in enumerate(data):
                # Use item['id'] for resume logic if it exists
                item_id = item.get("id", idx)
                k = (os.path.basename(input_file), item_id)
                
                if resume and k in done:
                    continue
                
                futures.append(ex.submit(
                    process_one_item_llm, client, model, item, input_file, idx, max_retries
                ))

            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"LLM Processing"):
                res = fut.result()
                f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                f_out.flush()


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
            
            # Use same ID logic for resume
            item_id = obj.get("id", obj.get("_item_index"))
            k = (obj.get("_source_file", ""), item_id)
            
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

    # Physics dataset paths (Defaults updated)
    parser.add_argument("--input-file", type=str,
                        default="./dataset/Physics/PHYSCICS.jsonl",
                        help="Path to the source .jsonl file")
    
    parser.add_argument("--save-dir", type=str,
                        default="./dataset/Physics/Reasoning",
                        help="Directory to save the intermediate LLM reasoning results")
    
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Per file limit; 0 means all")

    # stage exec args
    # Note: You usually run stage 1 first, see the output filename, then pass it here for stage 2
    parser.add_argument("--reasoning-jsonl", type=str, default="", help="Input for stage 2")
    parser.add_argument("--executed-jsonl", type=str, default="", help="Output for stage 2")

    # common
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", default=False)

    args = parser.parse_args()

    if args.stage == "llm":
        client = build_client()
        
        # Construct output filename based on input filename + model
        base_name = os.path.basename(args.input_file)
        if base_name.endswith(".jsonl"):
            base_name = base_name[:-6]
        elif base_name.endswith(".json"):
            base_name = base_name[:-5]
            
        out_name = f"{base_name}.reasoning.{args.model}.jsonl"
        output_path = os.path.join(args.save_dir, out_name)

        print(f"Stage 1: Processing {args.input_file}")
        print(f"Output will be saved to: {output_path}")

        stage_llm_main(
            client=client,
            model=args.model,
            input_file=args.input_file,
            output_jsonl_path=output_path,
            max_workers=args.max_workers,
            limit=args.limit,
            resume=args.resume,
            max_retries=args.max_retries
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
