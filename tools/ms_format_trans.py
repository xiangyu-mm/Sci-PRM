import json

# 输入和输出文件路径
input_file = './toolsciverifier/sciprm-bench/train.jsonl'
output_file = './toolsciverifier/sciprm-bench/train_conv.jsonl'

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
            question = data.get('question', '')
            reasoning_chain = data.get('reasoning_chain_labeled', [])
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            
            # 状态变量初始化
            previous_tool_output = None
            previous_tool_used = False
            
            for i, step in enumerate(reasoning_chain):
                step_id = step.get('step_id', i + 1)
                
                # --- 构建 User Content ---
                user_parts = []
                
                # 1. 第一步放入 Question
                if i == 0:
                    user_parts.append(f"Question:\n{question}\n")
                
                # 2. 只有当上一步确实使用了工具时，才展示 Observation
                if i > 0 and previous_tool_used:
                    # 处理 output 为 None 的情况，防止报错
                    obs_content = previous_tool_output if previous_tool_output is not None else "No output."
                    user_parts.append(f"Observation (Output from previous step):\n```\n{obs_content}\n```\n")
                
                # 3. 放入当前步骤的 Reasoning 和 Code
                step_content = (
                    f"Current Step {step_id}:\n"
                    f"Reasoning: {step.get('reasoning', '')}\n"
                    f"{format_tool_code(step)}\n\n"
                    "Is this step valid and correct?"
                )
                user_parts.append(step_content)
                
                user_message = "\n".join(user_parts)
                messages.append({"role": "user", "content": user_message})
                
                # --- 构建 Assistant Content ---
                is_valid = step.get('step_label', False)
                verdict = "Valid" if is_valid else "Invalid"
                
                assistant_content = f"{verdict}"
                messages.append({"role": "assistant", "content": assistant_content})
                
                # --- 更新状态供下一轮使用 ---
                previous_tool_output = step.get('tool_output', '')
                previous_tool_used = step.get('tool_used', False)

            # 写入文件
            fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')

    print(f"Conversion complete. Saved to {output_path}")

if __name__ == "__main__":
    process_file(input_file, output_file)
