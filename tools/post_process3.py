import json

# 文件路径
input_file_path = './toolsciverifier/sciprm-bench/sciprm_formatted.jsonl'
output_file_path = './toolsciverifier/sciprm-bench/sciprm_formatted_corrected.jsonl'

# 特例字符串：超时错误
TIMEOUT_ERROR_STR = "Error: Execution timed out (exceeded 120s)."
# 限制超时数据的最大数量
MAX_TIMEOUT_COUNT = 150

def process_file():
    current_timeout_count = 0
    kept_lines_count = 0
    skipped_lines_count = 0
    
    print(f"开始处理文件: {input_file_path}")
    print(f"超时错误限制数量: {MAX_TIMEOUT_COUNT}")
    
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # --- 第一步：检查本条数据是否包含超时错误 ---
                has_timeout = False
                if "reasoning_chain_labeled" in data:
                    for step in data["reasoning_chain_labeled"]:
                        tool_output = step.get("tool_output", "")
                        if tool_output and TIMEOUT_ERROR_STR in tool_output:
                            has_timeout = True
                            break
                
                # --- 第二步：根据配额决定是否保留该行 ---
                if has_timeout:
                    if current_timeout_count >= MAX_TIMEOUT_COUNT:
                        # 超过了150条，跳过这一行（丢弃）
                        skipped_lines_count += 1
                        continue
                    else:
                        # 没超过，计数加1，并继续后续的处理逻辑
                        current_timeout_count += 1

                # --- 第三步：处理标签（之前的逻辑） ---
                # 如果代码走到这里，说明这行数据是被保留的。
                # 现在检查其他的错误（空输出、SyntaxError等），将 label 标为 False
                
                if "reasoning_chain_labeled" in data:
                    for step in data["reasoning_chain_labeled"]:
                        original_label = step.get("step_label")
                        tool_used = step.get("tool_used", False)
                        
                        tool_output = step.get("tool_output", "")
                        if tool_output is None:
                            tool_output = ""
                        
                        output_lower = tool_output.lower()
                        should_be_false = False
                        
                        if tool_used:
                            # 1. 如果是超时错误，我们已经决定保留这行数据，且不将其视为逻辑错误（保持原标签）
                            if TIMEOUT_ERROR_STR in tool_output:
                                pass 
                            
                            # 2. 检查是否为空
                            elif not tool_output.strip():
                                should_be_false = True
                                
                            # 3. 检查其他 Error 或 Stderr (排除掉超时的情况)
                            elif "error" in output_lower or "[stderr]" in output_lower:
                                should_be_false = True
                        
                        # 修改标签
                        if should_be_false and original_label is not False:
                            print(step)
                            step["step_label"] = False

                # 写入文件
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                kept_lines_count += 1
                    
            except json.JSONDecodeError:
                print(f"警告: 第 {line_num + 1} 行 JSON 解析失败，已跳过。")
                continue

    print("-" * 30)
    print(f"处理完成。")
    print(f"保留的超时错误数据: {current_timeout_count} 条")
    print(f"丢弃的过量超时数据: {skipped_lines_count} 条")
    print(f"最终文件总行数: {kept_lines_count}")
    print(f"结果已保存至: {output_file_path}")

if __name__ == "__main__":
    process_file()
