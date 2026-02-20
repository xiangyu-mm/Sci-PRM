# -*- coding: utf-8 -*-
# eval_grpo_verifier.py

import os
import json
import time
import re
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# -------------------------
# Client Setup
# -------------------------
def build_client():
    return OpenAI(
        base_url="http://127.0.0.1:3888/v1",
        api_key="sk-********************"
    )

# -------------------------
# Retry Helper
# -------------------------
def try_request_with_retries(request_function, max_retries=5, delay=2, *args, **kwargs):
    for i in range(max_retries):
        try:
            return request_function(*args, **kwargs), None
        except Exception as e:
            if i == max_retries - 1:
                return None, f"{type(e).__name__}: {e}"
            time.sleep(delay * (i + 1))
    return None, "unknown_error"

# -------------------------
# LLM Call
# -------------------------
def call_llm(client, model, messages):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0
    )
    return resp.choices[0].message.content

# -------------------------
# Output Parsing
# -------------------------
def parse_response(response_text):
    """
    Parses 'Valid' or 'Invalid' from the response.
    Returns: True (Valid), False (Invalid), or None (Parse Error)
    """
    if not response_text:
        return None
    
    # Normalize
    text = response_text.strip().lower()
    
    # Check specifically for invalid first (as 'invalid' contains 'valid')
    # Using regex to ensure word boundaries to avoid partial matches if verbose
    if re.search(r'\binvalid\b', text):
        return False
    elif re.search(r'\bvalid\b', text):
        return True
    
    # Fallback: strict string matching if regex fails (though regex covers this)
    if "invalid" in text:
        return False
    if "valid" in text:
        return True
        
    return None

# -------------------------
# Processing Logic
# -------------------------
def process_one_item(client, model, item):
    """
    Process a single benchmark item.
    """
    item_id = item.get("id")
    messages = item.get("messages", [])
    ground_truth_str = item.get("solution", "").strip()
    
    # Convert Ground Truth to Boolean
    # Valid -> True, Invalid -> False
    if ground_truth_str == "Valid":
        ground_truth = True
    elif ground_truth_str == "Invalid":
        ground_truth = False
    else:
        # If dataset has other labels, handle or skip. Assuming binary here.
        return {
            "id": item_id,
            "status": "data_error_no_gt",
            "ground_truth": None,
            "prediction": None
        }

    # Call LLM
    raw_response, err = try_request_with_retries(
        call_llm,
        max_retries=5,
        delay=2,
        client=client,
        model=model,
        messages=messages
    )

    if err:
        return {
            "id": item_id,
            "status": f"request_failed: {err}",
            "llm_response_raw": None,
            "ground_truth": ground_truth,
            "prediction": None,
            "correct": False
        }
    # Parse Prediction
    prediction = parse_response(raw_response)
    status = "success"
    if prediction is None:
        status = "parse_error"
        # Treat parse error as incorrect or handle separately in metrics
        is_correct = False 
    else:
        is_correct = (prediction == ground_truth)

    return {
        "id": item_id,
        "source": item.get("source"),
        "status": status,
        "llm_response_raw": raw_response,
        "ground_truth": ground_truth, # True=Valid, False=Invalid
        "prediction": prediction,
        "correct": is_correct
    }

# -------------------------
# Metrics Calculation
# -------------------------
def calculate_metrics(results):
    tp = 0 # GT: Valid, Pred: Valid
    fp = 0 # GT: Invalid, Pred: Valid
    tn = 0 # GT: Invalid, Pred: Invalid
    fn = 0 # GT: Valid, Pred: Invalid
    
    parse_errors = 0
    total_valid_evals = 0
    
    for item in results:
        if item["status"] == "data_error_no_gt":
            continue
            
        if item["status"] != "success":
            parse_errors += 1
            # Decide whether to penalize parse errors. 
            # Here we skip them for Precision/Recall but they lower coverage/accuracy.
            # Usually for strict eval, parse error = False (Invalid) or just Wrong.
            # Let's count them as WRONG (mismatch) to ensure robustness.
            # But we don't know if it should have been P or N.
            # For simplicity, we track them separately and exclude from P/R/F1 calculation logic
            # unless we treat them as a specific class. 
            continue

        total_valid_evals += 1
        gt = item["ground_truth"]
        pred = item["prediction"]
        
        if gt is True and pred is True:
            tp += 1
        elif gt is True and pred is False:
            fn += 1
        elif gt is False and pred is True:
            fp += 1
        elif gt is False and pred is False:
            tn += 1

    # Safe division
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total_valid_evals if total_valid_evals > 0 else 0.0

    return {
        "total_items": len(results),
        "successful_parses": total_valid_evals,
        "parse_errors": parse_errors,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision_valid": round(precision, 4),
            "recall_valid": round(recall, 4),
            "f1_valid": round(f1, 4)
        }
    }

# -------------------------
# Main Execution
# -------------------------
def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="LLM model name to evaluate")
    parser.add_argument("--input-file", type=str, 
                        default="./toolsciverifier/sciprm-bench/test_conv_meta_grpo.jsonl")
    parser.add_argument("--save-dir", type=str, 
                        default="./toolsciverifier/sciprm-bench/eval_results")
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    client = build_client()
    
    # Load Data
    print(f"Loading data from {args.input_file}...")
    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if args.limit > 0:
        data = data[:args.limit]
        print(f"Limiting to first {args.limit} items.")

    # Prepare Output Path
    os.makedirs(args.save_dir, exist_ok=True)
    out_filename = f"eval_{args.model.replace('/', '_')}_{int(time.time())}.jsonl"
    out_path = os.path.join(args.save_dir, out_filename)
    
    results = []
    
    print(f"Starting evaluation with model: {args.model}")
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        # Using a higher max_retries inside process_one_item, so thread pool just submits
        futures = {ex.submit(process_one_item, client, args.model, item): item for item in data}
        
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            res = fut.result()
            results.append(res)

    # Save Detailed Results
    print(f"Saving detailed results to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    # Calculate and Print Metrics
    metrics = calculate_metrics(results)
    
    print("\n" + "="*40)
    print(f"EVALUATION REPORT: {args.model}")
    print("="*40)
    print(json.dumps(metrics, indent=2))
    print("="*40)

    # Save Metrics Summary
    summary_path = out_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
