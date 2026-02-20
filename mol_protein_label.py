# -*- coding: utf-8 -*-
"""
Run Stage 3 (Process Supervision) for Open-Ended Protein QA
File: generate_protein_answers_stage3_open_ended_v2_optimized.py
"""

import os
import json
import time
import re
import glob
import hashlib
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# ---------------- Configuration ----------------

SYSTEM_PROMPT = """
You are an expert Enzymologist and Logic Evaluator.
Your goal is to assess the quality of a reasoning chain used to determine a protein's function.
"""

USER_PROMPT_TEMPLATE = """
### Instruction
{instruction}

### Input Data (Protein Sequence / Context)
{input_data}

### Model's Execution History
The following is the step-by-step reasoning and code execution performed by an AI model so far:
{history_str}

---

### Evaluation Task

**Step 1: Derive Conclusion**
Based on the "Model's Execution History" above, what is the final answer regarding the protein's catalytic activity or function? 

**Step 2: Compare with Ground Truth**
The **Correct Answer (Ground Truth)** is:
"{ground_truth}"

**Step 3: Evaluate & Label**
Compare your derived conclusion and the individual history steps against the Ground Truth.

**CRITICAL LABELING RULES:**
1. **Standard Reasoning**: If the steps lead to the Ground Truth via valid logic, `is_correct` is `true`.
2. **Hallucination Check**: If the history is empty, irrelevant, or failed (errors), BUT the conclusion somehow matches the Ground Truth (lucky guess), `is_correct` is **false**. The reasoning is invalid.
3. **Honest Failure (The case you must handle carefully)**: 
   - If the history shows tool failures or lack of information, AND the conclusion **correctly states** that the function cannot be determined (e.g., "unknown", "undetermined", "failed"), `is_correct` should be **true**.
   - **Reasoning**: It is logically correct to admit ignorance when data is missing. Do not penalize the reasoning step for the failure of the tool.

### OUTPUT FORMAT (STRICT JSON)
Please output a JSON object with the following structure:
{{
    "predicted_conclusion": "The text summary of the answer based on the history provided.",
    "prediction_is_correct": true, // true if the conclusion accurately reflects the history
    "prediction_reason": "This is a logically sound assessment of the situation.",
    "step_evaluations": [
        {{ 
            "step_id": 1, 
            "is_correct": true, 
            "reason": "Code execution was successful." 
        }},
        {{ 
            "step_id": 2, 
            "is_correct": true, 
            "reason": "Correctly identifies that the previous BLAST step failed and thus no conclusion can be drawn." 
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
        
        block = f"Step {sid}:\n"
        block += f"Thought: {reasoning}\n"
        if step.get("tool_used"):
            block += f"Code:\n{tool_code}\n"
            s_out = str(tool_output)
            trunc_output = (s_out[:1000] + '...[truncated]') if len(s_out) > 1000 else s_out
            block += f"Execution Output:\n{trunc_output}\n"
        out_lines.append(block)
    return "\n".join(out_lines)

def get_item_hash(item):
    """生成数据的唯一指纹，用于断点续传去重"""
    # 组合 instruction 和 input 作为一个唯一标识
    content = str(item.get("instruction", "")) + str(item.get("input", ""))
    return hashlib.md5(content.encode('utf-8')).hexdigest()

# ---------------- Core Logic ----------------

def call_generator_model(client: OpenAI, model: str, system_prompt: str, user_prompt: str):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    if resp.choices:
        return resp.choices[0].message.content
    return ""

def process_one_item(item: dict, client: OpenAI, model: str):
    """处理单条数据，返回处理后的字典"""
    out = dict(item)
    steps = item.get("executed_steps", [])
    if not steps:
        steps = item.get("reasoning_chain_executed", [])
    
    if not steps:
        out["labeling_status"] = "skipped_no_steps"
        return out

    ground_truth = item.get("output", "").strip()
    input_data = item.get("input", "").strip()
    instruction = item.get("instruction", "").strip()

    if not ground_truth:
        out["labeling_status"] = "skipped_no_ground_truth"
        return out

    prompt = USER_PROMPT_TEMPLATE.format(
        instruction=instruction,
        input_data=input_data,
        ground_truth=ground_truth,
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
        eval_map = {int(e.get("step_id", -999)): e for e in evaluations}
        
        last_step_id = 0
        for step in steps:
            new_step = dict(step)
            sid = int(step.get("step_id", -999))
            if sid > last_step_id: last_step_id = sid
            
            eval_info = eval_map.get(sid, {})
            new_step["step_label"] = eval_info.get("is_correct", False)
            new_step["label_reason"] = eval_info.get("reason", "No evaluation provided.")
            labeled_steps.append(new_step)

        predicted_conclusion = parsed.get("predicted_conclusion", "")
        prediction_is_correct = parsed.get("prediction_is_correct", False)
        prediction_reason = parsed.get("prediction_reason", "")
        
        if predicted_conclusion:
            final_step = {
                "step_id": last_step_id + 1,
                "tool_used": False,
                "tool_type": "conclusion_generation",
                "tool_details": "",
                "reasoning": f"Based on the history, the conclusion is: {predicted_conclusion}",
                "tool_output": predicted_conclusion,
                "step_label": prediction_is_correct,
                "label_reason": prediction_reason
            }
            labeled_steps.append(final_step)
            
        out["labeling_status"] = "success"
    else:
        out["labeling_status"] = "failed_parse"
        labeled_steps = steps 

    out["reasoning_chain_labeled"] = labeled_steps
    return out

def process_file(input_path, output_path, args, client):
    # 1. 读取原始数据
    raw_data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: raw_data.append(json.loads(line))
                except: pass
    
    if args.limit > 0: 
        raw_data = raw_data[:args.limit]

    # 2. 检查已处理的数据（断点续传）
    processed_hashes = set()
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Checking existing progress in {output_path}...")
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    processed_hashes.add(get_item_hash(item))
                except:
                    pass
        print(f"Found {len(processed_hashes)} items already processed.")

    # 3. 过滤待处理数据
    tasks = []
    for item in raw_data:
        if args.overwrite or get_item_hash(item) not in processed_hashes:
            tasks.append(item)
        else:
            # 如果不需要覆盖且已存在，则跳过
            pass
    
    print(f"Processing {os.path.basename(input_path)}: {len(tasks)} items remaining (Total: {len(raw_data)})")
    
    if not tasks:
        return

    # 4. 实时写入 (Streaming Write)
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 使用追加模式打开文件
    with open(output_path, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            # 提交任务
            future_to_item = {ex.submit(process_one_item, item, client, args.model): item for item in tasks}
            
            # 使用 as_completed 实时获取结果
            for future in tqdm(as_completed(future_to_item), total=len(tasks), desc="Labeling", leave=False):
                try:
                    result = future.result()
                    # 立即写入文件
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    # 强制刷新缓冲区，确保写入硬盘
                    f_out.flush()
                except Exception as e:
                    print(f"Error processing item: {e}")
                    # 可选：将错误记录到日志文件

def main():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="./dataset/mol_instruction_step/output_with_multi_tool/")
    parser.add_argument("--output-dir", type=str, default="./dataset/mol_instruction_step/output_labeled_prm/")
    
    parser.add_argument("--ark-api-key", type=str, default=os.getenv("ARK_API_KEY", "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"))
    parser.add_argument("--ark-base-url", type=str, default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument("--model", type=str, default="doubao-seed-1-8-251228") 
    
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=-1) 
    parser.add_argument("--overwrite", action="store_true", help="If set, will ignore existing output file and re-process everything.")
    
    args = parser.parse_args()

    client = build_client(api_key=args.ark_api_key, base_url=args.ark_base_url)
    
    target_file = "catalytic_activity_reasoning.jsonl"
    input_files = glob.glob(os.path.join(args.input_dir, target_file))
    
    if not input_files:
        print(f"No files found matching {target_file} in {args.input_dir}")
        input_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))

    for in_file in input_files:
        process_file(in_file, os.path.join(args.output_dir, os.path.basename(in_file)), args, client)

if __name__ == "__main__":
    main()
