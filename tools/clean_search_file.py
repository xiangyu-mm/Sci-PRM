import json
import os

# 定义文件路径
input_file_path = './toolsciverifier/results_stage3/verified_universal_msearth_open_search.jsonl'
# 定义输出文件路径 (在原文件名后加上 _filtered)
output_file_path = input_file_path.replace('.jsonl', '_filtered.jsonl')

def process_file():
    total_count = 0
    filtered_count = 0
    kept_count = 0

    print(f"开始处理文件: {input_file_path}")
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in, \
             open(output_file_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                total_count += 1
                try:
                    data = json.loads(line)
                    should_keep = True
                    
                    # 检查 generated_answer_parsed 字段
                    parsed_steps = data.get("generated_answer_parsed", [])
                    
                    # 遍历每一个步骤
                    if isinstance(parsed_steps, list):
                        for step in parsed_steps:
                            # 检查是否存在 verification_result 且 status 为 Unverifiable
                            ver_result = step.get("verification_result", {})
                            if ver_result.get("status") == "Unverifiable":
                                should_keep = False
                                break
                    
                    # 写入符合条件的数据
                    if should_keep:
                        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                        kept_count += 1
                    else:
                        filtered_count += 1
                        
                except json.JSONDecodeError:
                    print(f"警告: 第 {total_count} 行 JSON 解析失败，已跳过。")
                    continue

        print("-" * 30)
        print("处理完成！")
        print(f"原始数据总量: {total_count}")
        print(f"被筛掉的数据: {filtered_count}")
        print(f"保留的数据量: {kept_count}")
        print(f"结果已保存至: {output_file_path}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file_path}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    process_file()
