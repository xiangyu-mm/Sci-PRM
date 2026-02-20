import json
import glob
import os
from tqdm import tqdm

# ================= 配置区域 =================

OUTPUT_FILE = "merged_prm_unified.jsonl"

# 定义数据源配置
# type: 用于区分处理逻辑
# name: 写入最终 json 的 source 字段值
# paths: 文件或文件夹列表
DATA_SOURCES = [
    {
        "type": "physics",
        "name": "Physics",
        "paths": [
            "./dataset/Physics/Reasoning/PHYSCICS.reasoning.Qwen/output_physics_labeled/PHYSCICS.reasoning.gemini-3-flash-preview_exc.jsonl",
            "./dataset/Physics/Reasoning/PHYSCICS.reasoning.Qwen/output_physics_labeled/Qwen2.5-14B-Instruct_exc.jsonl"
        ]
    },
    {
        "type": "chembench",
        "name": "ChemBench",
        "paths": [
            "./dataset/ChemBench4K_labeled/test"
        ]
    },
    {
        "type": "mol_instruct",
        "name": "Mol-Instruct",
        "paths": [
            "./dataset/mol_instruction_step/output_labeled_prm"
        ]
    }
]

# ================= 处理函数 =================

def get_all_files(paths):
    """获取路径列表下的所有 jsonl 文件"""
    all_files = []
    for p in paths:
        if os.path.isfile(p):
            all_files.append(p)
        elif os.path.isdir(p):
            files_in_dir = glob.glob(os.path.join(p, "*.jsonl"))
            all_files.extend(files_in_dir)
        else:
            print(f"警告: 路径不存在 -> {p}")
    return all_files

def format_chembench_question(data):
    """处理 ChemBench：将选项拼接到 Question 中"""
    q_text = data.get("question", "").strip()
    options_text = []
    
    # 遍历可能的选项键值
    for opt_key in ["A", "B", "C", "D", "E"]:
        if opt_key in data and data[opt_key]:
            options_text.append(f"{opt_key}: {data[opt_key]}")
    
    if options_text:
        return q_text + "\nOptions:\n" + "\n".join(options_text)
    return q_text

def process_line(data, source_config):
    """根据不同的 source 类型转换数据格式"""
    source_type = source_config["type"]
    source_name = source_config["name"]
    
    # 1. 检查是否有核心标签字段，没有则丢弃
    if "reasoning_chain_labeled" not in data or not data["reasoning_chain_labeled"]:
        return None

    new_entry = {
        "source": source_name,
        "id": data.get("id", data.get("_item_index", None)), # 保留一个ID方便追踪
        "reasoning_chain_labeled": data["reasoning_chain_labeled"]
    }

    # 2. 根据类型标准化 Question 和 Answer
    if source_type == "physics":
        # Physics: question 保持不变
        # solution 是详细解答，answer 是简略结果。通常用 solution 作为 answer 训练
        new_entry["question"] = data.get("question", "")
        new_entry["answer"] = data.get("answer", "")
        new_entry["solution"] = data.get("solution","")
        # 保留元数据
        if "domain" in data: new_entry["domain"] = data["domain"]
        if "difficulty" in data: new_entry["difficulty"] = data["difficulty"]

    elif source_type == "chembench":
        # ChemBench: 合并选项到 question
        new_entry["question"] = format_chembench_question(data)
        new_entry["answer"] = data.get("answer", "")

    elif source_type == "mol_instruct":
        # Mol-Instruct: 合并 instruction 和 input 为 question
        instr = data.get("instruction", "").strip()
        inp = data.get("input", "").strip()
        
        # 拼接逻辑：如果有 input，拼接到 instruction 后面
        if inp:
            new_entry["question"] = f"{instr}\nInput:\n{inp}"
        else:
            new_entry["question"] = instr
            
        # output 重命名为 answer
        new_entry["answer"] = data.get("output", "")

    # 3. 最终检查：确保 question 和 answer 不为空
    if not new_entry.get("question") or not new_entry.get("answer"):
        return None

    return new_entry

# ================= 主执行逻辑 =================

def main():
    total_written = 0
    total_skipped = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for config in DATA_SOURCES:
            files = get_all_files(config["paths"])
            print(f"正在处理来源: {config['name']} (共 {len(files)} 个文件)...")
            
            for file_path in tqdm(files, desc=f"Processing {config['name']}"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f_in:
                        for line in f_in:
                            line = line.strip()
                            if not line: continue
                            
                            try:
                                raw_data = json.loads(line)
                                processed_data = process_line(raw_data, config)
                                
                                if processed_data:
                                    f_out.write(json.dumps(processed_data, ensure_ascii=False) + "\n")
                                    total_written += 1
                                else:
                                    total_skipped += 1
                                    
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    print(f"读取文件出错 {file_path}: {e}")

    print("=" * 30)
    print("处理完成！")
    print(f"输出文件: {os.path.abspath(OUTPUT_FILE)}")
    print(f"成功合并数据条数: {total_written}")
    print(f"跳过无效/无标签数据: {total_skipped}")

if __name__ == "__main__":
    main()
