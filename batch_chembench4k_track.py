# -*- coding: utf-8 -*-
# chembench4k_two_stage.py
#
# Stage 1: call LLM only, save reasoning jsonl
# Stage 2: execute python_code steps from stage1 jsonl, save executed jsonl
#
# Notes:
# - Executes python_code steps using subprocess for isolation and thread safety.

import os
import json
import re
import time
import ujson
import io
import contextlib
import traceback
import subprocess
import sys
import tempfile
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
                print(out)
                st2["tool_output"] = out
                if "Execution Error:" in out or "[Stderr]:" in out:
                     st2["python_exec_status"] = "error_captured"
                else:
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
    print(f"Reading input: {reasoning_jsonl_path}")
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

    if not items:
        print(f"No items to process (or all done) for {reasoning_jsonl_path}")
        return

    with open(executed_jsonl_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_one_item_exec, obj) for obj in items]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"EXEC: {os.path.basename(reasoning_jsonl_path)}"):
                res = fut.result()
                f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                f_out.flush()


def stage_exec_on_dir(input_dir, save_dir, max_workers, resume):
    """
    Batch process all .jsonl files in input_dir for execution stage.
    """
    os.makedirs(save_dir, exist_ok=True)
    # 假设输入是 .jsonl 文件 (stage 1 的输出)
    input_files = sorted(glob(os.path.join(input_dir, "*.jsonl")))
    
    if not input_files:
        raise FileNotFoundError(f"No .jsonl files found under: {input_dir}")
    
    print(f"Found {len(input_files)} files to execute in {input_dir}")

    for in_path in input_files:
        filename = os.path.basename(in_path)
        
        # 自动生成输出文件名
        # 规则: xxx.reasoning.model.jsonl -> xxx.executed.model.jsonl
        if ".reasoning." in filename:
            out_name = filename.replace(".reasoning.", ".executed.")
        else:
            # 如果没有 reasoning 关键字，直接在 .jsonl 前加 .executed
            out_name = filename.replace(".jsonl", ".executed.jsonl")
            
        out_path = os.path.join(save_dir, out_name)
        
        print(f"\n>>> Processing: {filename} -> {out_name}")
        stage_exec_jsonl(
            reasoning_jsonl_path=in_path,
            executed_jsonl_path=out_path,
            max_workers=max_workers,
            resume=resume
        )


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

    # stage exec args (Single file)
    parser.add_argument("--reasoning-jsonl", type=str, default=None)
    parser.add_argument("--executed-jsonl", type=str, default=None)

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
        # 逻辑判断：如果是批量模式（提供了 input-dir 且该路径是目录），则批量处理
        # 否则回退到单文件模式
        
        is_batch_mode = False
        if args.input_dir and os.path.isdir(args.input_dir):
            # 这里的 input-dir 在 exec 阶段指的是包含 reasoning .jsonl 的目录
            # 为了避免混淆，用户可以显式指定 --input-dir 指向 reasoning 目录
            # 或者我们检查 args.input_dir 是否包含 jsonl
            jsonl_files = glob(os.path.join(args.input_dir, "*.jsonl"))
            if jsonl_files:
                is_batch_mode = True
        
        if is_batch_mode:
            # 使用 input-dir 作为输入源，save-dir 作为输出目录
            # 注意：这里的 save-dir 默认值可能是 reasoning 目录，你可能需要手动指定一个新的 executed 目录
            print(f"Batch execution mode on directory: {args.input_dir}")
            stage_exec_on_dir(
                input_dir=args.input_dir,
                save_dir=args.save_dir,
                max_workers=args.max_workers,
                resume=args.resume
            )
        else:
            # 单文件模式
            if not args.reasoning_jsonl or not args.executed_jsonl:
                raise ValueError("For single file mode, --reasoning-jsonl and --executed-jsonl are required. "
                                 "For batch mode, provide --input-dir containing .jsonl files.")
            stage_exec_jsonl(
                reasoning_jsonl_path=args.reasoning_jsonl,
                executed_jsonl_path=args.executed_jsonl,
                max_workers=args.max_workers,
                resume=args.resume
            )


if __name__ == "__main__":
    main()
