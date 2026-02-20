import json
from collections import defaultdict

# 文件路径
meta_file = './toolsciverifier/sciprm-bench/test_conv_meta_with_judge_grpo.jsonl'
eval_file = './toolsciverifier/sciprm-bench/eval_results/eval_glm-4.6v_1770389297.jsonl'

def calculate_random_baseline(name, y_true):
    """
    根据 Ground Truth 分布，计算 50/50 随机猜测的理论指标。
    """
    if not y_true:
        print(f"--- {name} (无数据) ---")
        return

    # 1. 统计真实分布
    total = len(y_true)
    actual_positives = sum(1 for y in y_true if y) # 真实为 True 的数量
    actual_negatives = total - actual_positives    # 真实为 False 的数量

    if total == 0:
        return

    # 2. 计算随机猜测 (50% True / 50% False) 的期望混淆矩阵
    # 期望 TP = 真实正样本 * 0.5
    exp_tp = actual_positives * 0.5
    # 期望 FN = 真实正样本 * 0.5
    exp_fn = actual_positives * 0.5
    
    # 期望 FP = 真实负样本 * 0.5
    exp_fp = actual_negatives * 0.5
    # 期望 TN = 真实负样本 * 0.5
    exp_tn = actual_negatives * 0.5

    # 3. 计算指标
    # Accuracy: (TP + TN) / Total => (0.5P + 0.5N) / (P + N) = 0.5
    accuracy = 0.5
    
    # Precision: TP / (TP + FP) => 0.5P / (0.5P + 0.5N) = P / Total (即数据中正样本的比例)
    denom_prec = exp_tp + exp_fp
    precision = exp_tp / denom_prec if denom_prec > 0 else 0
    
    # Recall: TP / (TP + FN) => 0.5P / P = 0.5
    recall = 0.5
    
    # F1 Score
    denom_f1 = precision + recall
    f1 = 2 * (precision * recall) / denom_f1 if denom_f1 > 0 else 0

    print(f"--- {name} [Random Baseline] ---")
    print(f"样本总数 : {total}")
    print(f"正样本数 : {actual_positives} (占比 {actual_positives/total:.2%})")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("-" * 40)

def main():
    # 1) 建立 ID -> (code_judge, source) 映射
    id_to_meta = {}
    print(f"正在读取元数据文件: {meta_file} ...")
    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                sid = data["id"]
                id_to_meta[sid] = {
                    "code_judge": bool(data.get("code_judge", False)),
                    "source": data.get("source", "UNKNOWN")
                }
    except FileNotFoundError:
        print("错误：找不到元数据文件。")
        return

    # 2) 读取评估结果 (只关心 Ground Truth)
    # - 全局列表
    code_true = []
    reasoning_true = []

    # - 按 source 分组
    group_true = defaultdict(lambda: {True: [], False: []})

    missing_ids = 0
    skipped_none_gt = 0

    print(f"正在读取评估文件: {eval_file} ...")
    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)

                sample_id = data.get("id")
                if sample_id not in id_to_meta:
                    missing_ids += 1
                    continue

                meta = id_to_meta[sample_id]
                is_code_judge = meta["code_judge"]
                source = meta["source"]

                gt = data.get("ground_truth")

                if gt is None:
                    skipped_none_gt += 1
                    continue

                gt = bool(gt)

                # 全局收集
                if is_code_judge:
                    code_true.append(gt)
                else:
                    reasoning_true.append(gt)

                # 按 Source 收集
                group_true[source][is_code_judge].append(gt)

    except FileNotFoundError:
        print("错误：找不到评估结果文件。")
        return

    print("\n" + "=" * 50)
    print("RANDOM BASELINE REPORT (50/50 Guess)")
    print("=" * 50 + "\n")

    # --- Global ---
    calculate_random_baseline("Global | Code Verification", code_true)
    calculate_random_baseline("Global | Reasoning Logic", reasoning_true)
    
    all_true = code_true + reasoning_true
    calculate_random_baseline("Global | Overall (Total)", all_true)

    # --- By Source ---
    print("\n" + "=" * 50)
    print("BY SOURCE BREAKDOWN")
    print("=" * 50 + "\n")

    for source in sorted(group_true.keys()):
        print(f">>>> SOURCE: {source} <<<<")
        
        y_true_c = group_true[source][True]
        y_true_r = group_true[source][False]
        
        if y_true_c:
            calculate_random_baseline(f"{source} | Code", y_true_c)
        
        if y_true_r:
            calculate_random_baseline(f"{source} | Reasoning", y_true_r)
            
        y_true_all = y_true_c + y_true_r
        if y_true_all:
            calculate_random_baseline(f"{source} | Overall", y_true_all)
        print("")

if __name__ == "__main__":
    main()
