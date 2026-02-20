import json
import random
import os

# 文件路径配置
input_file_path = './toolsciverifier/sciprm-bench/train_conv_grpo.jsonl'
# 新文件保存路径，我加了一个后缀 _sampled
output_file_path = './toolsciverifier/sciprm-bench/train_conv_grpo_sampled.jsonl'

def sample_dataset():
    valid_lines = []
    invalid_lines = []
    
    print(f"正在读取文件: {input_file_path} ...")
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    solution = data.get("solution")
                    
                    if solution == "Valid":
                        valid_lines.append(data)
                    elif solution == "Invalid":
                        invalid_lines.append(data)
                except json.JSONDecodeError:
                    print("跳过格式错误的行")
                    continue
                    
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file_path}")
        return

    print(f"读取完成。统计如下：")
    print(f" - Valid 总数: {len(valid_lines)}")
    print(f" - Invalid 总数: {len(invalid_lines)}")

    # 设置采样数量
    target_valid_count = 6000
    
    # 进行采样
    if len(valid_lines) > target_valid_count:
        print(f"正在从 Valid 数据中随机采样 {target_valid_count} 条...")
        sampled_valid = random.sample(valid_lines, target_valid_count)
    else:
        print(f"警告：Valid 数据不足 {target_valid_count} 条，将保留全部 Valid 数据。")
        sampled_valid = valid_lines

    # 合并数据 (所有的 Invalid + 采样的 Valid)
    final_data = invalid_lines + sampled_valid
    
    # 打乱数据顺序 (Shuffle)，这对训练很重要
    random.shuffle(final_data)
    
    print(f"最终数据集数量: {len(final_data)} (Invalid: {len(invalid_lines)} + Valid: {len(sampled_valid)})")
    
    # 写入新文件
    print(f"正在写入新文件: {output_file_path} ...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print("处理完成！")

if __name__ == "__main__":
    sample_dataset()
