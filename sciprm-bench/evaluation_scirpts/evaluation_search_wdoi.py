# -*- coding: utf-8 -*-
# eval_verifier_paper_check.py

import os
import json
import time
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# -------------------------
# 1. 配置与初始化
# -------------------------
def build_client():
    return OpenAI(
        base_url="http://127.0.0.1:3888/v1",
        api_key="sk-********************"
    )

# -------------------------
# 2. Prompt 构造 (针对文献真实性验证修改)
# -------------------------
def construct_verification_prompt(item, use_gt=False):
    """
    构造 Prompt。
    :param item: 数据项
    :param use_gt: Boolean, 是否允许使用 Ground Truth 信息 (仅限前100条)
    """
    meta = item.get("meta_data", {})
    inp = item.get("input", {})
    label = item.get("label", {}) 
    current_step = inp.get("current_step", {})
    history = inp.get("step_history", [])
    
    # 提取上下文信息
    question = meta.get('query', 'N/A')
    caption = meta.get('caption', 'N/A')
    context_title = meta.get('title', 'N/A')
    
    # 提取待验证的步骤信息
    tool_type = current_step.get('tool_type', 'N/A')
    tool_input = current_step.get('tool_details', 'N/A')
    tool_output = current_step.get('tool_output', 'N/A') 
    reasoning = current_step.get('reasoning', 'N/A')
    
    # 提取 DOI Ground Truth
    doi_ground_truth = label.get('doi_ground_truth', None)
    
    # 格式化历史
    history_str = "\n".join(history) if history else "No previous steps."
    
    system_prompt = (
        "You are an expert scientific fact-checker and researcher. "
        "Your task is to verify the authenticity and relevance of a specific academic paper/citation "
        "retrieved by an AI agent during a problem-solving process.\n"
        "Strictly output your response in JSON format."
    )
    
    user_prompt = f"""
### Problem Context
**Context Paper Title:** {context_title}
**Question:** {question}
**Image Caption:** {caption}

### Reasoning History
{history_str}

### Step to Verify (Web Search / Paper Retrieval)
**Tool Used:** {tool_type}
**Tool Input (Query):** {tool_input}
**Tool Output (Retrieved Paper/Info):** 
{tool_output}
"""

    # --- 逻辑修改：只有当 use_gt 为 True 且 存在 GT 数据时才插入 ---
    if use_gt and doi_ground_truth and isinstance(doi_ground_truth, str) and doi_ground_truth.strip():
        user_prompt += f"""
### Reference DOI Verification Info (Ground Truth)
We have performed a database lookup for the DOI mentioned in the 'Tool Output'. Use this information to judge the authenticity:
{doi_ground_truth}
"""
    # -----------------------------------------------------------

    user_prompt += f"""
**Reasoning:** {reasoning}

### Your Verification Tasks:
1. **Authenticity Check (Crucial):** 
   - Does the paper mentioned in the 'Tool Output' actually exist?
   - Do the Title, Author (if any), Year, and DOI (if have) match a real publication?
   - **Important:** If a 'Reference DOI Verification Info' is provided above, compare the 'Tool Output' against it. If the DOI corresponds to a completely different paper (Title/Author mismatch), mark it as Hallucinated.

2. **Relevance Check:**
   - If the paper exists, is it helpful for answering the specific 'Question' above?

### Output Format
Please output strictly in JSON format with the following keys:
{{
    "status": "Authentic" or "Hallucinated",
    "analysis": "Step-by-step verification logic. First state if the DOI/Title exists. Then state if it is relevant."
}}

**Note:** 
- Return "Authentic" ONLY if the paper is real AND the metadata (DOI/Year if have) is consistent.
- Return "Hallucinated" if the paper does not exist, the DOI is fake, or the title/DOI mismatch.
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

# -------------------------
# 3. LLM 调用与重试
# -------------------------
def call_llm(client, model, messages):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise e

def try_request_with_retries(func, *args, **kwargs):
    max_retries = 3
    for i in range(max_retries):
        try:
            return func(*args, **kwargs), None
        except Exception as e:
            if i == max_retries - 1:
                return None, str(e)
            time.sleep(2)
    return None, "unknown_error"

# -------------------------
# 4. 解析与归一化
# -------------------------
def parse_json_response(text):
    if not text: return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return None

def normalize_status(status_str):
    if not status_str: return False
    s = status_str.strip().lower()
    if ("authentic" in s or "correct" in s) and "hallucinated" not in s and "incorrect" not in s:
        return True
    if "hallucinated" in s or "fake" in s or "fabricated" in s:
        return False
    return False

# -------------------------
# 5. 单条处理逻辑
# -------------------------
def process_one_item(client, model, item, use_gt=False):
    """
    处理单条数据
    :param use_gt: 是否允许在 Prompt 中加入 Ground Truth
    """
    item_id = item.get("unique_id")
    
    gt_raw = item.get("label", {}).get("status", "Unknown")
    ground_truth = (gt_raw == "Authentic")
    
    # 传递 use_gt 给 prompt 构造函数
    messages = construct_verification_prompt(item, use_gt=use_gt)
    
    raw_res, err = try_request_with_retries(call_llm, client, model, messages)
    
    if err:
        return {
            "id": item_id,
            "status": "api_error",
            "error_msg": err,
            "ground_truth": ground_truth
        }
    
    parsed = parse_json_response(raw_res)
    if parsed and "status" in parsed:
        pred_raw = parsed["status"]
        prediction = normalize_status(pred_raw)
        
        return {
            "id": item_id,
            "status": "success",
            "ground_truth": ground_truth,
            "prediction": prediction,
            "gt_raw": gt_raw,
            "pred_raw": pred_raw,
            "used_doi_gt_in_prompt": use_gt, # 记录是否使用了GT提示
            "analysis": parsed.get("analysis", "")
        }
    else:
        return {
            "id": item_id,
            "status": "parse_error",
            "raw_response": raw_res,
            "ground_truth": ground_truth
        }

# -------------------------
# 6. 指标计算
# -------------------------
def calculate_metrics(results):
    valid_results = [r for r in results if r["status"] == "success"]
    
    tp = 0; tn = 0; fp = 0; fn = 0
    
    for r in valid_results:
        g = r["ground_truth"]
        p = r["prediction"]
        
        if g and p: tp += 1
        elif not g and not p: tn += 1
        elif not g and p: fp += 1
        elif g and not p: fn += 1
        
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    detection_rate = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "count": total,
        "parse_errors": len(results) - len(valid_results),
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "hallucination_detection_rate": round(detection_rate, 4)
        }
    }

# -------------------------
# 7. 主函数
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="e.g. gpt-4o")
    parser.add_argument("--input-file", type=str, required=True, help="Path to jsonl file")
    parser.add_argument("--save-dir", type=str, default="./toolsciverifier/sciprm-bench/eval_res_search")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    
    client = build_client()
    
    # 读取数据
    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
            
    print(f"Loaded {len(data)} items.")
    os.makedirs(args.save_dir, exist_ok=True)
    
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {}
        # 使用 enumerate 来判断索引
        for idx, item in enumerate(data):
            # 只有前 100 条数据 (index 0 到 99) 允许使用 GT
            use_gt_flag = (idx < 100)
            
            # 提交任务时传递 use_gt_flag
            future = ex.submit(process_one_item, client, args.model, item, use_gt_flag)
            futures[future] = item
            
        for fut in tqdm(as_completed(futures), total=len(data)):
            results.append(fut.result())
            
    # 计算指标
    metrics = calculate_metrics(results)
    print(json.dumps(metrics, indent=2))
    
    # 保存结果
    ts = int(time.time())
    save_path = os.path.join(args.save_dir, f"{args.model.replace('/', '-')}_{ts}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    main()
