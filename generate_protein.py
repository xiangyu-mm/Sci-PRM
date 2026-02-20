# -*- coding: utf-8 -*-
"""
Run Stage 1 (LLM Generation) - WITH REFERENCE CODE:
python generate_protein_answers.py \
  --stage llm \
  --prompt-type with_ref \
  --input-file /path/to/catalytic_activity.json \
  --output-file /path/to/output_with_ref/catalytic_activity_reasoning.jsonl \
  --ark-api-key YOUR_KEY \
  --max-workers 8

Run Stage 1 (LLM Generation) - NO REFERENCE CODE:
python generate_protein_answers.py \
  --stage llm \
  --prompt-type no_ref \
  --input-file /path/to/catalytic_activity.json \
  --output-file /path/to/output_no_ref/catalytic_activity_reasoning.jsonl \
  --ark-api-key YOUR_KEY \
  --max-workers 8

Run Stage 2 (Code Execution) - Works for either output:
python generate_protein_answers.py \
  --stage exec \
  --input-file /path/to/output_with_ref/catalytic_activity_reasoning.jsonl \
  --output-file /path/to/output_with_ref/catalytic_activity_executed.jsonl \
  --max-workers 8
"""

import os
import json
import time
import re
import io
import contextlib
import traceback
from tqdm import tqdm
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import sys
import subprocess
import tempfile
# Try to import ujson for speed, fallback to json
try:
    import ujson
except ImportError:
    import json as ujson

# ---------------- Configuration ----------------

SYSTEM_PROMPT = """
You are a helpful assistant and a professional bioinformatics researcher.
You are in an English-speaking environment. Please answer strictly in English.
"""

# Template: Reference Code (Used only for "with_ref" mode)
REF_CODE_TEMPLATE = """
import time
from Bio.Blast import NCBIWWW
from Bio import SearchIO

# The input protein sequence
sequence = \"\"\"<<SEQUENCE_PLACEHOLDER>>\"\"\"

def identify_protein(seq):
    print("Querying NCBI BLAST database... (This may take a minute)")
    # Perform a blastp search against the non-redundant (nr) database
    try:
        result_handle = NCBIWWW.qblast("blastp", "nr", seq.strip())
        # Parse the result
        blast_record = SearchIO.read(result_handle, "blast-xml")
        
        # Get the top hit
        if len(blast_record) > 0:
            top_hit = blast_record[0]
            print(f"Top Hit ID: {top_hit.id}")
            print(f"Description: {top_hit.description}")
        else:
            print("No matches found.")
    except Exception as e:
        print(f"BLAST Error: {e}")

if __name__ == "__main__":
    identify_protein(sequence)
"""

# Strict format instruction
OUTPUT_FORMAT_STR = """
OUTPUT FORMAT (STRICT):
Return STRICT JSON only (no extra text).
The root MUST be a JSON ARRAY; each element is one reasoning step.
Each step MUST include:
- step_id: number
- tool_used: boolean
- tool_type: string ("python_code" / "scientific_api" / "none")
- tool_details: string (runnable code / exact query; empty string if tool_used is false)
- reasoning: string (what the step checks and how it supports discriminating among options; do not invent tool results)
"""

# Prompt 1: WITH Reference Code
USER_PROMPT_WITH_REF = """
Question: {question}

Input Protein Sequence:
{sequence}

Answer this question. You can use tools; if you use tools, you must provide specific code. I will run the code and give you the output for your next step of reasoning. You can refer to the following code to retrieve the protein ID and related description:

{ref_code}

""" + OUTPUT_FORMAT_STR

# Prompt 2: NO Reference Code (New)
USER_PROMPT_NO_REF = """
Question: {question}

Input Protein Sequence:
{sequence}

Answer this question. You can use tools; if you use tools, you must provide specific code. 
I will run the code and give you the output for your next step of reasoning. 
The code needs to be directly runnable and concise. Do not require manual API key insertion if possible.

""" + OUTPUT_FORMAT_STR

# ---------------- Helpers ----------------

def build_client(api_key: str, base_url: str):
    return OpenAI(base_url=base_url, api_key=api_key)

def try_request_with_retries(fn, max_retries=6, delay=2, **kwargs):
    """Retry wrapper with exponential backoff."""
    for i in range(max_retries):
        try:
            return fn(**kwargs), None
        except Exception as e:
            if i == max_retries - 1:
                return None, str(e)
            time.sleep(delay * (i + 1))
    return None, "unknown_error"

def remove_trailing_commas(s: str) -> str:
    """Removes trailing commas from JSON strings."""
    return re.sub(r",\s*([}\]])", r"\1", s)

def extract_json_array_from_text(text: str):
    """Robustly extracts a JSON array from text."""
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

# def exec_python_capture_stdout(code: str) -> str:
#     """Executes Python code string and captures stdout."""
#     stdout = io.StringIO()
#     # 在全局变量字典中，伪造 __name__
#     env = {'__name__': '__main__'} 
#     # loc = {}
#     try:
#         with contextlib.redirect_stdout(stdout):
#             exec(code, env, env)
#     except Exception:
#         traceback.print_exc(file=stdout)
#     return stdout.getvalue().strip()

def exec_python_capture_stdout(code: str) -> str:
    """
    Executes Python code in a separate subprocess to ensure thread safety 
    and capture stdout correctly.
    """
    # 1. 创建一个临时文件来保存代码
    # delete=False 是为了在关闭文件后让 subprocess 还能读取它，我们在 finally 中手动删除
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    
    try:
        # 2. 使用当前环境的 python 解释器执行这个临时文件
        # capture_output=True 会同时捕获 stdout 和 stderr
        # timeout=120 防止死循环代码卡死程序
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # 3. 获取输出
        output = result.stdout
        
        # 如果有报错信息，也拼接到输出中，方便调试
        if result.stderr:
            output += f"\n[Stderr]:\n{result.stderr}"
            
    except subprocess.TimeoutExpired:
        output = "Error: Execution timed out (exceeded 120s)."
    except Exception as e:
        output = f"Execution Error: {str(e)}"
    finally:
        # 4. 清理临时文件
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
    
    return output.strip()

def load_done_indices(output_file: str):
    """Loads processed indices to support resume."""
    done = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if "_item_index" in obj:
                    done.add(obj["_item_index"])
            except:
                pass
    return done

# ---------------- Stage 1: LLM Generation ----------------

def call_generator_model(client: OpenAI, model: str, system_prompt: str, user_prompt: str):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    if resp.choices:
        return resp.choices[0].message.content
    return ""

def process_one_llm(item: dict, index: int, client: OpenAI, model: str, prompt_type: str):
    out = dict(item)
    out["_item_index"] = index
    out["prompt_type"] = prompt_type  # Record which prompt was used
    
    instruction = item.get("instruction", "").strip()
    sequence = item.get("input", "").strip()
    
    if not instruction or not sequence:
        out["status"] = "skipped_empty_input"
        return out

    # Prepare Prompt based on prompt_type
    if prompt_type == "with_ref":
        specific_ref_code = REF_CODE_TEMPLATE.replace("<<SEQUENCE_PLACEHOLDER>>", sequence)
        prompt = USER_PROMPT_WITH_REF.format(
            question=instruction,
            sequence=sequence,
            ref_code=specific_ref_code
        )
    elif prompt_type == "no_ref":
        # New prompt without reference code
        prompt = USER_PROMPT_NO_REF.format(
            question=instruction,
            sequence=sequence
        )
    else:
        out["status"] = "error_invalid_prompt_type"
        return out

    # Call LLM
    raw_output, err = try_request_with_retries(
        call_generator_model,
        max_retries=5,
        delay=2,
        client=client,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt
    )

    out["llm_raw_response"] = raw_output if raw_output else ""
    out["llm_error"] = err
    
    # Parse JSON
    if raw_output:
        ok, steps = extract_json_array_from_text(raw_output)
        out["is_valid_json"] = ok
        out["parsed_steps"] = steps if ok else []
        out["status"] = "llm_success" if ok else "json_parse_error"
    else:
        out["status"] = "llm_failed"
        out["parsed_steps"] = []

    return out

def run_stage_llm(args):
    """
    Input: Raw JSON List
    Output: JSONL with LLM reasoning (parsed)
    """
    if not args.ark_api_key:
        raise ValueError("Missing API key for Stage 1.")
    
    client = build_client(api_key=args.ark_api_key, base_url=args.ark_base_url)

    # Load Input
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[:args.limit]

    # Resume Logic
    done_indices = load_done_indices(args.output_file) if args.resume else set()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    
    print(f"Running LLM Stage with prompt type: [{args.prompt_type}]")

    with open(args.output_file, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = []
            for idx, item in enumerate(data):
                if idx in done_indices:
                    continue
                # Pass prompt_type to the worker function
                futures.append(ex.submit(process_one_llm, item, idx, client, args.model, args.prompt_type))
            
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Stage 1: LLM"):
                try:
                    res = fut.result()
                    f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f_out.flush()
                except Exception as e:
                    print(f"Error in LLM stage: {e}")

# ---------------- Stage 2: Execution ----------------

def process_one_exec(item: dict):
    """
    Takes an item from Stage 1 (which contains 'parsed_steps'),
    executes the python code, and appends results.
    """
    out = dict(item)
    steps = out.get("parsed_steps", [])
    
    if not steps:
        out["exec_status"] = "no_steps_to_exec"
        out["executed_steps"] = []
        return out

    executed_steps = []
    any_execution = False

    for st in steps:
        st2 = dict(st)
        
        # Normalize fields
        if "tool_used" not in st2: st2["tool_used"] = False
        if "tool_type" not in st2: st2["tool_type"] = "none"
        if "tool_details" not in st2: st2["tool_details"] = ""
        st2["tool_output"] = ""
        st2["exec_status"] = "skipped"

        # Execute
        if st2.get("tool_used"):
            code = st2.get("tool_details", "") or ""
            # print(code)
            if code.strip():
                # --- EXECUTION HAPPENS HERE ---
                result = exec_python_capture_stdout(code)
                print(result)
                st2["tool_output"] = result
                st2["exec_status"] = "success" if "Traceback" not in result else "error"
                any_execution = True
            else:
                st2["exec_status"] = "empty_code"
        
        executed_steps.append(st2)

    out["executed_steps"] = executed_steps
    out["exec_status"] = "finished" if any_execution else "skipped_no_code"
    return out

def run_stage_exec(args):
    """
    Input: JSONL from Stage 1
    Output: JSONL with added execution results
    """
    # Load Input (JSONL)
    items = []
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Stage 2 input not found: {args.input_file}")

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except:
                    continue

    if args.limit > 0:
        items = items[:args.limit]

    # Resume Logic
    done_indices = load_done_indices(args.output_file) if args.resume else set()

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    with open(args.output_file, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = []
            for item in items:
                idx = item.get("_item_index", -1)
                if args.resume and idx != -1 and idx in done_indices:
                    continue
                
                futures.append(ex.submit(process_one_exec, item))
            
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Stage 2: Exec"):
                try:
                    res = fut.result()
                    f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f_out.flush()
                except Exception as e:
                    print(f"Error in Exec stage: {e}")

# ---------------- Main ----------------

def main():
    parser = ArgumentParser()
    
    # Mode selection
    parser.add_argument("--stage", type=str, choices=["llm", "exec"], required=True,
                        help="Select pipeline stage: 'llm' for generation, 'exec' for code execution.")
    
    # Prompt Type Selection (New)
    parser.add_argument("--prompt-type", type=str, choices=["with_ref", "no_ref"], default="with_ref",
                        help="Select prompt template: 'with_ref' includes BLAST reference code, 'no_ref' is a concise prompt without reference code.")

    # I/O
    parser.add_argument("--input-file", type=str,
                        default="./dataset/mol_instruction/Protein-oriented_Instructions/catalytic_activity.json")
    parser.add_argument("--output-file", type=str,
                        default="./dataset/mol_instruction_step/catalytic_activity.jsonl")
    
    # LLM Config (Only needed for Stage 1)
    parser.add_argument("--ark-base-url", type=str,
                        default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument("--ark-api-key", type=str,
                        default=os.getenv("ARK_API_KEY", "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"))
    parser.add_argument("--model", type=str, default="doubao-seed-1-6-251015")
    
    # Runtime Config
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=1000, help="Limit number of items to process")
    parser.add_argument("--resume", action="store_true", help="Skip already processed items in output file")

    args = parser.parse_args()

    if args.stage == "llm":
        print(f"=== Starting Stage 1: LLM Generation ===")
        print(f"Prompt Type: {args.prompt_type}")
        print(f"Input: {args.input_file}")
        print(f"Output: {args.output_file}")
        run_stage_llm(args)
    
    elif args.stage == "exec":
        print(f"=== Starting Stage 2: Code Execution ===")
        print(f"Input: {args.input_file}")
        print(f"Output: {args.output_file}")
        run_stage_exec(args)

    print("Done.")

if __name__ == "__main__":
    main()
