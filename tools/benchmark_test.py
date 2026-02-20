import json

# 输入和输出文件路径
input_file = './toolsciverifier/sciprm-bench/test.jsonl'
output_file = './toolsciverifier/sciprm-bench/test_conv_meta_with_judge.jsonl'

# System Prompt
SYSTEM_PROMPT = (
    "You are an expert process verifier. "
    "Your task is to evaluate the proposed reasoning step and tool usage code BEFORE execution. "
    "Check if the logic is sound and the code is correct for the given problem. "
    "The execution result of the previous step (if any) is provided as context. "
    "Response format: 'Valid' or 'Invalid' only."
)

def format_tool_code(step):
    """只格式化代码部分"""
    if not step.get("tool_used", False):
        return "" 
    
    return (
        f"Tool Type: {step.get('tool_type', 'N/A')}\n"
        f"Proposed Code:\n```python\n{step.get('tool_details', '')}\n```"
    )

def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # 1. 提取元数据
            source = data.get('source', '')
            sample_id = data.get('id', '')
            
            question = data.get('question', '')
            reasoning_chain = data.get('reasoning_chain_labeled', [])
            
            # System message 初始化，code_judge 默认为 False
            messages = [
                {
                    "role": "system", 
                    "content": SYSTEM_PROMPT,
                    "code_judge": False 
                }
            ]
            
            # 状态变量初始化
            previous_tool_output = None
            previous_tool_used = False
            
            for i, step in enumerate(reasoning_chain):
                step_id = step.get('step_id', i + 1)
                
                # 判断当前步是否涉及代码评判
                # 如果 tool_used 为 True，说明这一步包含代码，需要评判代码
                is_code_verification = step.get('tool_used', False)
                
                # --- 构建 User Content ---
                user_parts = []
                
                # A. 第一步放入 Question
                if i == 0:
                    user_parts.append(f"Question:\n{question}\n")
                
                # B. 只有当上一步确实使用了工具时，才展示 Observation
                if i > 0 and previous_tool_used:
                    obs_content = previous_tool_output if previous_tool_output is not None else "No output."
                    user_parts.append(f"Observation (Output from previous step):\n```\n{obs_content}\n```\n")
                
                # C. 放入当前步骤的 Reasoning 和 Code
                step_content = (
                    f"Current Step {step_id}:\n"
                    f"Reasoning: {step.get('reasoning', '')}\n"
                    f"{format_tool_code(step)}\n\n"
                    "Is this step valid and correct?"
                )
                user_parts.append(step_content)
                
                user_message = "\n".join(user_parts)
                
                # 添加 User 消息，带上 code_judge
                messages.append({
                    "role": "user", 
                    "content": user_message,
                    "code_judge": is_code_verification
                })
                
                # --- 构建 Assistant Content ---
                is_valid = step.get('step_label', False)
                verdict = "Valid" if is_valid else "Invalid"
                
                assistant_content = f"{verdict}"
                
                # 添加 Assistant 消息，带上 code_judge (与 User 对应)
                messages.append({
                    "role": "assistant", 
                    "content": assistant_content,
                    "code_judge": is_code_verification
                })
                
                # --- 更新状态供下一轮使用 ---
                previous_tool_output = step.get('tool_output', '')
                previous_tool_used = step.get('tool_used', False)

            # 2. 构建包含 source, id 和 messages 的最终字典
            output_record = {
                "source": source,
                "id": sample_id,
                "messages": messages
            }

            # 写入文件
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')

    print(f"Conversion complete. Saved to {output_path}")

if __name__ == "__main__":
    process_file(input_file, output_file)
