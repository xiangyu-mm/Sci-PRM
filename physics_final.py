# -*- coding: utf-8 -*-
"""
Run Stage 3 (Process Supervision) for Physics/Math Generation Tasks.
Features:
- Handles open-ended numerical/formula answers.
- Extracts Ground Truth from 'solution' and nested 'answer' fields.
- Real-time writing to disk (streaming).
- Resume capability.
"""

import os
import json
import time
import re
import glob
import threading
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# ---------------- Configuration ----------------

SYSTEM_PROMPT = """
You are an expert Physics/Math Evaluator and Code Interpreter.
Your goal is to assess the quality of a reasoning chain used to solve a scientific problem.
You must distinguish between valid deduction, lucky guesses, and honest failures.
"""

USER_PROMPT_TEMPLATE = """
### Instruction
{instruction}

### Input Data / Context
{input_data}

### Model's Execution History
The following is the step-by-step reasoning and code execution performed by an AI model so far:
{history_str}

---

### Evaluation Task

**Step 1: Derive Conclusion**
Based on the "Model's Execution History" above, determine what the model believes the answer is.

**Step 2: Compare with Ground Truth**
The **Correct Answer (Ground Truth)** and **Reference Solution** are provided below:
{ground_truth}

**Step 3: Evaluate & Label**
Compare the model's steps against the Reference Solution.

**CRITICAL LABELING RULES:**
1. **Standard Reasoning**: If a step follows valid physics/math logic and moves towards the solution (even if different from the reference method), `is_correct` is `true`.
2. **Code Errors**: If a step tries to run code but fails (e.g., NameError, SyntaxError) AND the model does not recover/correct it in the immediate next step, `is_correct` is **false**.
   - *Note on your specific data*: In the history provided, variables might not persist between steps. If the code fails because a variable is undefined (NameError), that step is technically **incorrect** regarding code execution.
3. **Calculation Errors**: If the reasoning is correct but the calculation (mental or code) yields a wrong number, `is_correct` is **false**.
4. **Final Result**: If the final conclusion matches the Ground Truth value (within reasonable tolerance for physics), the chain is generally successful, unless it was a lucky guess from wrong steps.

### OUTPUT FORMAT (STRICT JSON)
Please output a JSON object with the following structure:
{{
    "predicted_conclusion": "The value or expression the model derived (e.g., '11.6 N').",
    "prediction_matches_ground_truth": true/false, 
    "final_analysis": "Brief explanation of whether the reasoning was sound.",
    "step_evaluations": [
        {{ 
            "step_id": 1, 
            "is_correct": true, 
            "reason": "Correctly identified the formula for momentum." 
        }},
        {{ 
            "step_id": 2, 
            "is_correct": false, 
            "reason": "The code failed with a NameError because 'p' was not defined in this cell's context." 
        }}
    ]
}}
"""

# ---------------- Helpers ----------------

def build_client(api_key: str, base_url: str):
    return OpenAI(base_url=base_url, api_key=api_key)

def try_request_with_retries(fn, max_retries=5, delay=2, **kwargs):
    for i in range(max_retries):
        try:
            return fn(**kwargs), None
        except Exception as e:
            if i == max_retries - 1:
                return None, str(e)
            time.sleep(delay * (i + 1))
    return None, "unknown_error"

def extract_json_from_text(text: str):
    try:
        m = re.search(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        js_str = m.group(1) if m else text
        m2 = re.search(r"\{[\s\S]*\}", js_str)
        if m2:
            js_str = m2.group(0)
        js_str = re.sub(r",\s*([}\]])", r"\1", js_str) 
        return json.loads(js_str)
    except:
        return None

def format_history(executed_steps):
    if not executed_steps:
        return "No reasoning steps provided."
    out_lines = []
    for step in executed_steps:
        sid = step.get("step_id", "?")
        reasoning = step.get("reasoning", "")
        tool_code = step.get("tool_details", "")
        tool_output = step.get("tool_output", "")
        exec_status = step.get("python_exec_status", "")
        
        block = f"Step {sid}:\n"
        if reasoning:
            block += f"Thought: {reasoning}\n"
        if step.get("tool_used"):
            block += f"Code:\n{tool_code}\n"
            if exec_status:
                block += f"Execution Status: {exec_status}\n"
            s_output = str(tool_output)
            trunc_output = (s_output[:1500] + '...[truncated]') if len(s_output) > 1500 else s_output
            block += f"Execution Output:\n{trunc_output}\n"
        out_lines.append(block)
    return "\n".join(out_lines)

def format_options(item):
    # 处理没有选项的情况
    opts = []
    for key in ["A", "B", "C", "D", "E"]:
        if key in item and item[key]:
            opts.append(f"{key}: {item[key]}")
    if opts:
        return "\n".join(opts)
    return "No options provided (Open-ended calculation)."

# ---------------- Core Logic ----------------

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

def process_one_item(item: dict, client: OpenAI, model: str):
    out = dict(item)
    steps = item.get("reasoning_chain_executed", [])
    
    if not steps:
        out["labeling_status"] = "skipped_no_steps"
        return out

    # --- 核心修改：解析 Solution 和 Answer ---
    solution_text = item.get("solution", "")
    raw_answer = item.get("answer", "")
    
    # 解析嵌套列表 [["11.6"]] 或 ["11.6"] 或 "11.6"
    target_value = str(raw_answer)
    if isinstance(raw_answer, list) and len(raw_answer) > 0:
        first_el = raw_answer[0]
        if isinstance(first_el, list) and len(first_el) > 0:
            target_value = str(first_el[0]) # Handle [["11.6"]]
        else:
            target_value = str(first_el)   # Handle ["11.6"]
            
    ground_truth_display = f"Target Value: {target_value}\n\nReference Solution:\n{solution_text}"
    # ----------------------------------------

    prompt = USER_PROMPT_TEMPLATE.format(
        instruction=item.get("question", ""),
        input_data=format_options(item),
        ground_truth=ground_truth_display,
        history_str=format_history(steps)
    )

    raw_output, err = try_request_with_retries(
        call_generator_model,
        client=client,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt
    )
    
    out["stage3_judge_response"] = raw_output
    parsed = extract_json_from_text(raw_output) if raw_output else None
    
    labeled_steps = []
    
    if parsed:
        evaluations = parsed.get("step_evaluations", [])
        eval_map = {int(e.get("step_id", -1)): e for e in evaluations}
        
        last_step_id = 0
        for step in steps:
            new_step = dict(step)
            sid = int(step.get("step_id", -999))
            if sid > last_step_id: last_step_id = sid
            
            eval_info = eval_map.get(sid, {})
            new_step["is_correct"] = eval_info.get("is_correct", False)
            new_step["label_reason"] = eval_info.get("reason", "")
            labeled_steps.append(new_step)

        # Final Analysis Step
        final_analysis = parsed.get("final_analysis", "")
        predicted_conclusion = parsed.get("predicted_conclusion", "")
        matches_gt = parsed.get("prediction_matches_ground_truth", False)
        
        final_step = {
            "step_id": last_step_id + 1,
            "tool_used": False,
            "tool_type": "final_verdict",
            "tool_details": "",
            "python_exec_status": "",
            "reasoning": final_analysis,
            "tool_output": predicted_conclusion,
            "is_correct": matches_gt,
            "label_reason": final_analysis
        }
        
        labeled_steps.append(final_step)
        out["labeling_status"] = "success"
        
    else:
        out["labeling_status"] = "failed_parse"
        labeled_steps = steps

    out["reasoning_chain_labeled"] = labeled_steps
    return out

# ---------------- File Processing ----------------

def process_file(input_path, output_path, args, client):
    processed_indices = set()
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Checking existing output file: {output_path}")
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        # 优先使用 _item_index，没有则使用 id
                        idx = obj.get("_item_index", obj.get("id"))
                        if idx is not None:
                            processed_indices.add(idx)
                    except: pass
        except Exception as e:
            print(f"Warning: Could not read existing file: {e}")
        print(f"Found {len(processed_indices)} already processed items.")

    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: 
                    item = json.loads(line)
                    idx = item.get("_item_index", item.get("id"))
                    if idx not in processed_indices:
                        data.append(item)
                except: pass
    
    if args.limit > 0: 
        data = data[:args.limit]
    
    if not data:
        print(f"No new items to process for {os.path.basename(input_path)}")
        return

    print(f"Processing {os.path.basename(input_path)}: {len(data)} new items")
    
    write_lock = threading.Lock()
    file_mode = 'a' if (os.path.exists(output_path) and not args.overwrite) else 'w'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, file_mode, encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(process_one_item, item, client, args.model): item for item in data}
            
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Labeling", leave=False)
            
            for fut in pbar:
                try:
                    result = fut.result()
                    with write_lock:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()
                except Exception as e:
                    print(f"\nError processing an item: {e}")

    print(f"Finished processing. Saved to {output_path}")

def main():
    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Specific input JSONL file path")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save labeled JSONL files")
    parser.add_argument("--ark-api-key", type=str, default="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
    parser.add_argument("--ark-base-url", type=str, default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument("--model", type=str, default="doubao-seed-1-8-251228")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    client = build_client(api_key=args.ark_api_key, base_url=args.ark_base_url)
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return

    out_file = os.path.join(args.output_dir, os.path.basename(args.input_file))
    process_file(args.input_file, out_file, args, client)

if __name__ == "__main__":
    main()
