import json
import os

# 文件路径
input_file_path = "./toolsciverifier/merged_prm_unified.jsonl"
output_file_path = input_file_path.replace(".jsonl", "_formatted.jsonl")

def process_file():
    print(f"正在处理文件: {input_file_path} ...")
    
    processed_count = 0
    modified_count = 0
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in, \
             open(output_file_path, 'w', encoding='utf-8') as f_out:
            print("starting......")
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    line_modified = False
                    
                    # 检查 reasoning_chain_labeled 字段
                    if "reasoning_chain_labeled" in data and isinstance(data["reasoning_chain_labeled"], list):
                        for step in data["reasoning_chain_labeled"]:
                            # 获取 exec_status，如果不存在则为 None
                            exec_status = step.get("exec_status")
                            
                            # 逻辑：只有当 exec_status 存在（即这是一个执行了工具的步骤）
                            # 并且 exec_status 不等于 "success" 时，将 step_label 设为 false
                            print(exec_status)
                            if "error" in str(exec_status):
                                if step.get("step_label") != False:
                                    step["step_label"] = False
                                    line_modified = True
                    
                    # 写入处理后的数据（ensure_ascii=False 保证中文不被转义）
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    
                    processed_count += 1
                    if line_modified:
                        modified_count += 1
                        
                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误，跳过该行: {e}")
                    continue

        print("-" * 30)
        print(f"处理完成！")
        print(f"总行数: {processed_count}")
        print(f"修改了数据的行数: {modified_count}")
        print(f"新文件已保存至: {output_file_path}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    process_file()
