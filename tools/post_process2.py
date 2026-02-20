import json
import os

# 输入文件路径
input_path = "./toolsciverifier/merged_prm_unified_formatted.jsonl"
# 输出文件路径
output_path = "./toolsciverifier/sciprm-bench/sciprm_formatted.jsonl"

# 想要统一成的 tool_type 名称
TARGET_TOOL_TYPE_NAME = "final_verdict"

def process_file():
    print(f"开始处理文件: {input_path}")
    count = 0
    skipped_count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # 获取 reasoning_chain_labeled 列表
                    chain = data.get("reasoning_chain_labeled", [])
                    
                    # -------------------------------------------------
                    # 新增逻辑：检查是否所有 step 都有 step_label
                    # -------------------------------------------------
                    has_missing_label = False
                    if chain:
                        for step in chain:
                            if "step_label" not in step:
                                has_missing_label = True
                                break
                    
                    if has_missing_label:
                        # 如果发现缺少 step_label，直接跳过，不写入文件
                        # print(f"跳过数据 ID {data.get('id', 'unknown')}: 缺少 step_label")
                        skipped_count += 1
                        continue
                    # -------------------------------------------------

                    if chain and len(chain) > 0:
                        # 获取最后一个 step
                        last_step = chain[-1]
                        
                        # Debug: 打印不符合预期的 tool_type (仅为了观察，不影响逻辑)
                        if last_step.get("tool_type") not in ["generate_conclusion", "conclusion_generation", "final_verdict"]:
                            # 为了防止刷屏，只打印类型
                            print(f"发现非标准 tool_type: {last_step.get('tool_type')}")
                            pass

                        # 1. 统一 tool_type 名称
                        last_step["tool_type"] = TARGET_TOOL_TYPE_NAME
                        
                        # 2. 将 reasoning 替换为 tool_output 的内容
                        if "tool_output" in last_step:
                            last_step["reasoning"] = last_step["tool_output"]
                        else:
                            # 如果没有 tool_output，保留原样或打印警告
                            print(f"Warning: ID {data.get('id')} 缺少 tool_output")
                        
                        # 更新回去
                        chain[-1] = last_step
                        data["reasoning_chain_labeled"] = chain
                    
                    # 写入新文件
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    count += 1
                    
                except json.JSONDecodeError:
                    print(f"跳过无效的 JSON 行: {line[:50]}...")
                    continue

        print(f"处理完成！")
        print(f"成功写入: {count} 条")
        print(f"过滤删除: {skipped_count} 条 (因缺少 step_label)")
        print(f"新文件已保存至: {output_path}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    process_file()
