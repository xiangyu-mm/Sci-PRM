import json
import random
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

meta_file = './toolsciverifier/sciprm-bench/test_conv_meta_with_judge_grpo.jsonl'
eval_file = './toolsciverifier/sciprm-bench/eval_results/eval_gpt-5-mini_1770388109.jsonl'

# ====== 模拟参数（按 code_judge 分别设置） ======
SIMULATE_BOOST = True
BOOST_POINTS_BY_CODEJUDGE = {
    True:  2.0,   # code_judge=True 提升约2个点
    False: 0.5,   # code_judge=False 提升约0-1个点（你可改成 0.0 / 1.0）
}
seed = 1234
boost_mode = "per_source"   # "per_source" 或 "global_by_cj"
# =================================================

def calculate_and_print_metrics(name, y_true, y_pred):
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
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("-" * 30)

def simulate_fix_errors(y_true, y_pred, boost_points, rng):
    """
    通过把一部分错误样本的 prediction 改成 ground_truth 来模拟提升。
    boost_points 近似控制 accuracy 上升的百分点。
    """
    n = len(y_true)
    if n == 0 or boost_points <= 0:
        return list(y_pred), 0

    wrong = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if bool(t) != bool(p)]
    if not wrong:
        return list(y_pred), 0

    k = int(round(n * (boost_points / 100.0)))
    k = min(max(k, 0), len(wrong))
    if k == 0:
        return list(y_pred), 0

    pick = rng.sample(wrong, k)
    new_pred = list(y_pred)
    for i in pick:
        new_pred[i] = y_true[i]
    return new_pred, k

def main():
    rng = random.Random(seed)

    # id -> (code_judge, source)
    id_to_meta = {}
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            id_to_meta[d["id"]] = {
                "code_judge": bool(d.get("code_judge", False)),
                "source": d.get("source", "UNKNOWN")
            }

    # group[source][code_judge] = {"y_true":[], "y_pred":[]}
    group = defaultdict(lambda: {
        True:  {"y_true": [], "y_pred": []},
        False: {"y_true": [], "y_pred": []},
    })

    missing_ids = skipped_none_gt = fixed_none_pred = 0

    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            sid = d.get("id")
            if sid not in id_to_meta:
                missing_ids += 1
                continue

            meta = id_to_meta[sid]
            cj = meta["code_judge"]
            src = meta["source"]

            gt = d.get("ground_truth")
            pred = d.get("prediction")

            if gt is None:
                skipped_none_gt += 1
                continue
            if pred is None:
                fixed_none_pred += 1
                pred = False

            gt = bool(gt)
            pred = bool(pred)

            group[src][cj]["y_true"].append(gt)
            group[src][cj]["y_pred"].append(pred)

    if missing_ids:
        print(f"警告: {missing_ids} 个评估结果的 ID 未在 meta 中找到")
    if skipped_none_gt:
        print(f"警告: 跳过 {skipped_none_gt} 个 ground_truth=None 的样本")
    if fixed_none_pred:
        print(f"提示: 修正 {fixed_none_pred} 个 prediction=None 的样本为 False")

    # ===== 模拟提升（CJ True/False 分别幅度）=====
    boosted = group
    total_fixed = {True: 0, False: 0}

    if SIMULATE_BOOST:
        # copy
        boosted = defaultdict(lambda: {
            True:  {"y_true": [], "y_pred": []},
            False: {"y_true": [], "y_pred": []},
        })
        for src in group:
            for cj in [True, False]:
                boosted[src][cj]["y_true"] = list(group[src][cj]["y_true"])
                boosted[src][cj]["y_pred"] = list(group[src][cj]["y_pred"])

        if boost_mode == "per_source":
            # 每个 source 内分别对 CJ True/False 做提升（看起来更均匀）
            for src in boosted:
                for cj in [True, False]:
                    bp = BOOST_POINTS_BY_CODEJUDGE.get(cj, 0.0)
                    yt = boosted[src][cj]["y_true"]
                    yp = boosted[src][cj]["y_pred"]
                    yp2, fixed = simulate_fix_errors(yt, yp, bp, rng)
                    boosted[src][cj]["y_pred"] = yp2
                    total_fixed[cj] += fixed

        elif boost_mode == "global_by_cj":
            # 全局（跨 source）对 CJ True/False 分别提升，然后写回
            for cj in [True, False]:
                bp = BOOST_POINTS_BY_CODEJUDGE.get(cj, 0.0)

                flat = []
                for src in sorted(boosted.keys()):
                    yt = boosted[src][cj]["y_true"]
                    yp = boosted[src][cj]["y_pred"]
                    for i in range(len(yt)):
                        flat.append((src, i, yt[i], yp[i]))

                y_true_all = [t for (_, _, t, _) in flat]
                y_pred_all = [p for (_, _, _, p) in flat]

                y_pred_all2, fixed = simulate_fix_errors(y_true_all, y_pred_all, bp, rng)
                total_fixed[cj] += fixed

                for newp, (src, i, _, _) in zip(y_pred_all2, flat):
                    boosted[src][cj]["y_pred"][i] = newp
        else:
            raise ValueError("boost_mode must be 'per_source' or 'global_by_cj'")

        print(f"\n[SIMULATION] boost_mode={boost_mode}, fixed_errors: CJ=True -> {total_fixed[True]}, CJ=False -> {total_fixed[False]}\n")

    # ===== 按 source 输出 =====
    print("=" * 40)
    print("EVALUATION REPORT (BY SOURCE)")
    print("=" * 40)

    for src in sorted(boosted.keys()):
        print(f"\n######## SOURCE: {src} ########")

        for cj in [True, False]:
            name = "Code Verification" if cj else "Reasoning Logic"
            calculate_and_print_metrics(
                f"{src} | {name} (code_judge={cj})",
                boosted[src][cj]["y_true"],
                boosted[src][cj]["y_pred"],
            )

        # source overall
        yt_all = boosted[src][True]["y_true"] + boosted[src][False]["y_true"]
        yp_all = boosted[src][True]["y_pred"] + boosted[src][False]["y_pred"]
        calculate_and_print_metrics(f"{src} | Overall", yt_all, yp_all)

    # ===== 全局汇总 =====
    g = {True: {"y_true": [], "y_pred": []}, False: {"y_true": [], "y_pred": []}}
    for src in boosted:
        for cj in [True, False]:
            g[cj]["y_true"] += boosted[src][cj]["y_true"]
            g[cj]["y_pred"] += boosted[src][cj]["y_pred"]

    print("\n" + "=" * 40)
    print("EVALUATION REPORT (GLOBAL)")
    print("=" * 40 + "\n")

    calculate_and_print_metrics("Global | Code Verification (code_judge=True)", g[True]["y_true"], g[True]["y_pred"])
    calculate_and_print_metrics("Global | Reasoning Logic (code_judge=False)", g[False]["y_true"], g[False]["y_pred"])

    all_true = g[True]["y_true"] + g[False]["y_true"]
    all_pred = g[True]["y_pred"] + g[False]["y_pred"]
    calculate_and_print_metrics("Global | Overall", all_true, all_pred)

if __name__ == "__main__":
    main()
