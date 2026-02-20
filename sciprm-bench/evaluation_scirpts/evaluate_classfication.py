import json
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 文件路径
meta_file = './toolsciverifier/sciprm-bench/test_conv_meta_with_judge_grpo.jsonl'
eval_file = './toolsciverifier/sciprm-bench/eval_results/eval_glm-4.6v_1770389297.jsonl'


def calculate_and_print_metrics(name, y_true, y_pred):
    """计算并打印各项指标"""
    if not y_true:
        print(f"--- {name} (无有效数据) ---")
        return

    y_true = [bool(y) for y in y_true]
    y_pred = [bool(y) for y in y_pred]

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    print(f"--- {name} ---")
    print(f"样本数量: {len(y_true)}")
    print(f"Accuracy (准确率): {acc:.4f}")
    print(f"Precision (精确率): {precision:.4f}")
    print(f"Recall    (召回率): {recall:.4f}")
    print(f"F1 Score  (F1分数): {f1:.4f}")
    print(f"混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("-" * 30)


def main():
    # 1) 建立 ID -> (code_judge, source) 映射
    id_to_meta = {}
    print(f"正在读取元数据文件: {meta_file} ...")
    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                sid = data["id"]
                id_to_meta[sid] = {
                    "code_judge": bool(data.get("code_judge", False)),
                    "source": data.get("source", "UNKNOWN")
                }
    except FileNotFoundError:
        print("错误：找不到元数据文件。")
        return

    # 2) 读取评估结果，分别：
    # - 全局按 code_judge 汇总
    code_true, code_pred = [], []
    reasoning_true, reasoning_pred = [], []

    # - 按 source 再按 code_judge 汇总
    # group[source][code_judge]['y_true'/'y_pred']
    group = defaultdict(lambda: {
        True:  {"y_true": [], "y_pred": []},
        False: {"y_true": [], "y_pred": []},
    })

    missing_ids = 0
    skipped_none_gt = 0
    fixed_none_pred = 0

    print(f"正在读取评估文件: {eval_file} ...")
    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)

                sample_id = data.get("id")
                if sample_id not in id_to_meta:
                    missing_ids += 1
                    continue

                meta = id_to_meta[sample_id]
                is_code_judge = meta["code_judge"]
                source = meta["source"]

                gt = data.get("ground_truth")
                pred = data.get("prediction")

                # 清洗
                if gt is None:
                    skipped_none_gt += 1
                    continue
                if pred is None:
                    fixed_none_pred += 1
                    pred = False

                gt = bool(gt)
                pred = bool(pred)

                # 全局按 code_judge
                if is_code_judge:
                    code_true.append(gt)
                    code_pred.append(pred)
                else:
                    reasoning_true.append(gt)
                    reasoning_pred.append(pred)

                # 按 source + code_judge
                group[source][is_code_judge]["y_true"].append(gt)
                group[source][is_code_judge]["y_pred"].append(pred)

    except FileNotFoundError:
        print("错误：找不到评估结果文件。")
        return

    if missing_ids > 0:
        print(f"警告: 有 {missing_ids} 个评估结果的 ID 在元数据文件中未找到。")
    if skipped_none_gt > 0:
        print(f"警告: 跳过了 {skipped_none_gt} 个 Ground Truth 为 None 的样本。")
    if fixed_none_pred > 0:
        print(f"提示: 修正了 {fixed_none_pred} 个 Prediction 为 None 的样本 (默认为 False)。")

    print("\n" + "=" * 40)
    print("EVALUATION REPORT (GLOBAL)")
    print("=" * 40 + "\n")

    calculate_and_print_metrics("Code Verification (Code Judge = True)", code_true, code_pred)
    calculate_and_print_metrics("Reasoning Logic (Code Judge = False)", reasoning_true, reasoning_pred)

    all_true = code_true + reasoning_true
    all_pred = code_pred + reasoning_pred
    calculate_and_print_metrics("Overall (Total)", all_true, all_pred)

    # 3) 按 source 输出
    print("\n" + "=" * 40)
    print("EVALUATION REPORT (BY SOURCE)")
    print("=" * 40 + "\n")

    for source in sorted(group.keys()):
        print(f"\n######## SOURCE: {source} ########")

        y_true_c = group[source][True]["y_true"]
        y_pred_c = group[source][True]["y_pred"]
        y_true_r = group[source][False]["y_true"]
        y_pred_r = group[source][False]["y_pred"]

        calculate_and_print_metrics(f"{source} | Code Verification (Code Judge = True)", y_true_c, y_pred_c)
        calculate_and_print_metrics(f"{source} | Reasoning Logic (Code Judge = False)", y_true_r, y_pred_r)

        # source overall（可选，但通常很有用）
        y_true_all = y_true_c + y_true_r
        y_pred_all = y_pred_c + y_pred_r
        calculate_and_print_metrics(f"{source} | Overall", y_true_all, y_pred_all)


if __name__ == "__main__":
    main()
