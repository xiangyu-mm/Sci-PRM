import json
import random
import os
from tqdm import tqdm

# ================= 配置区域 =================

INPUT_FILE = "./toolsciverifier/sciprm-bench/sciprm_formatted_corrected.jsonl"
TRAIN_FILE = "./toolsciverifier/sciprm-bench/train.jsonl"
TEST_FILE = "./toolsciverifier/sciprm-bench/test.jsonl"

# 测试集采样配置
# 格式: "Source名称": 采样数量
SAMPLE_CONFIG = {
    "ChemBench": 500,
    "Mol-Instruct": 500,
    "Physics": 200
}

# 设置随机种子以保证结果可复现
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ================= 处理逻辑 =================

def analyze_sample_tool_usage(entry):
    """
    分析单条数据的工具使用情况和正确性。
    返回:
        has_tool (bool): 是否使用了工具
        is_tool_usage_correct (bool): 工具使用是否全部正确
            - True: 有工具且所有工具步骤都正确
            - False: 有工具且至少有一个工具步骤不正确
            - None: 没有使用工具
    """
    steps = entry.get("reasoning_chain_labeled", [])
    if not steps:
        return False, None

    has_tool = False
    has_error_in_tool = False

    for step in steps:
        # 检查该步骤是否使用了工具
        if step.get("tool_used") is True:
            has_tool = True
            # 检查该步骤是否正确
            if step.get("step_label") is False:
                has_error_in_tool = True

    if not has_tool:
        return False, None
    
    # 如果有工具，且发现了错误，则整体判定为 False (负样本)
    # 如果有工具，且没发现错误，则整体判定为 True (正样本)
    return True, not has_error_in_tool

def split_data():
    print(f"正在读取 {INPUT_FILE} ...")
    
    # 1. 数据桶初始化
    pools = {k: {'pos': [], 'neg': []} for k in SAMPLE_CONFIG.keys()}
    train_buffer = []

    total_lines = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line: continue
            total_lines += 1
            
            try:
                entry = json.loads(line)
                source = entry.get("source")
                
                # 分析工具使用情况
                has_tool, is_tool_correct = analyze_sample_tool_usage(entry)
                
                # 只有属于目标Source 且 使用了工具的数据 才有资格进入测试集
                if source in SAMPLE_CONFIG and has_tool:
                    if is_tool_correct:
                        pools[source]['pos'].append(entry)
                    else:
                        pools[source]['neg'].append(entry)
                else:
                    # 其他数据直接进入训练集
                    train_buffer.append(entry)
                    
            except json.JSONDecodeError:
                continue

    # 2. 执行采样
    test_set = []
    
    print("\n开始采样测试集...")
    for source, target_count in SAMPLE_CONFIG.items():
        pos_candidates = pools[source]['pos']
        neg_candidates = pools[source]['neg']
        
        # [FIX] 使用 int() 强制转换，确保是整数
        # 建议：先算 pos，剩下的给 neg，确保总和等于 target_count
        target_pos = int(target_count * 0.2)
        target_neg = target_count - target_pos
        
        # 检查是否有足够的负样本
        if len(neg_candidates) < target_neg:
            print(f"[{source}] 警告: 负样本不足 (需要 {target_neg}, 实际 {len(neg_candidates)})。将用正样本填充。")
            actual_neg = len(neg_candidates)
            actual_pos = target_count - actual_neg
        # 检查是否有足够的正样本
        elif len(pos_candidates) < target_pos:
            print(f"[{source}] 警告: 正样本不足 (需要 {target_pos}, 实际 {len(pos_candidates)})。将用负样本填充。")
            actual_pos = len(pos_candidates)
            actual_neg = target_count - actual_pos
        else:
            actual_pos = target_pos
            actual_neg = target_neg
            
        # 再次确保是 int (虽然上面逻辑已经保证了，但为了双重保险)
        actual_pos = int(actual_pos)
        actual_neg = int(actual_neg)

        # 随机抽取
        random.shuffle(pos_candidates)
        random.shuffle(neg_candidates)
        
        selected_pos = pos_candidates[:actual_pos]
        selected_neg = neg_candidates[:actual_neg]
        
        # 剩余的放回训练集
        remaining_pos = pos_candidates[actual_pos:]
        remaining_neg = neg_candidates[actual_neg:]
        
        test_set.extend(selected_pos)
        test_set.extend(selected_neg)
        
        train_buffer.extend(remaining_pos)
        train_buffer.extend(remaining_neg)
        
        print(f"[{source}] 完成采样: 总计 {len(selected_pos) + len(selected_neg)} "
              f"(True: {len(selected_pos)}, False: {len(selected_neg)})")

    # 3. 写入文件
    print(f"\n正在写入文件...")
    
    # 写入测试集
    with open(TEST_FILE, 'w', encoding='utf-8') as f_test:
        for item in test_set:
            f_test.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # 写入训练集 (打乱顺序)
    random.shuffle(train_buffer)
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f_train:
        for item in train_buffer:
            f_train.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("=" * 30)
    print(f"处理完成!")
    print(f"原始数据总量: {total_lines}")
    print(f"测试集数量: {len(test_set)} -> {os.path.abspath(TEST_FILE)}")
    print(f"训练集数量: {len(train_buffer)} -> {os.path.abspath(TRAIN_FILE)}")

if __name__ == "__main__":
    split_data()
