import json
import os

# 输入文件路径
input_file = './toolsciverifier/results/msearth_open.answers.filtered.jsonl'
# 输出文件路径 (为了安全起见，保存为一个新文件)
output_file = './toolsciverifier/results/msearth_open.answers.filtered.processed.jsonl'

def process_line(data):
    # 1. 提取并重构 steps
    steps = []
    
    # 检查是否存在源字段 grounded_generated_answer_parsed
    if 'grounded_generated_answer_parsed' in data:
        source_data = data['grounded_generated_answer_parsed']
        
        # 输入格式中，steps 在该字段内部
        if isinstance(source_data, dict) and 'steps' in source_data:
            steps = source_data['steps']
        
        # 处理完后删除旧字段 (根据您的输出示例，旧key被替换了)
        del data['grounded_generated_answer_parsed']

    # 2. 处理每个 step
    processed_steps = []
    for idx, step in enumerate(steps):
        # 添加 step_id (从1开始)
        step['step_id'] = idx + 1
        
        # 检查是否为 web_search 工具
        if step.get('tool_used') is True and step.get('tool_type') == 'web_search':
            # 添加您要求的 verification_result
            step['verification_result'] = {
                "status": "Correct", 
                "issue_type": "No_Content", 
                "analysis": "None."
            }
        
        processed_steps.append(step)

    # 3. 将处理后的列表赋值给新字段名 generated_answer_parsed
    data['generated_answer_parsed'] = processed_steps

    # 4. 添加 verify_status
    data['verify_status'] = "processed"

    # 5. 重命名状态字段 (grounded_generation_status -> generation_status)
    if 'grounded_generation_status' in data:
        data['generation_status'] = data['grounded_generation_status']
        del data['grounded_generation_status']

    return data

print(f"正在处理文件: {input_file} ...")

try:
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        count = 0
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                json_obj = json.loads(line)
                processed_obj = process_line(json_obj)
                f_out.write(json.dumps(processed_obj, ensure_ascii=False) + '\n')
                count += 1
            except json.JSONDecodeError as e:
                print(f"解析错误跳过一行: {e}")
                
    print(f"处理完成！")
    print(f"共处理 {count} 行数据。")
    print(f"结果已保存至: {output_file}")

except FileNotFoundError:
    print(f"错误: 找不到输入文件 {input_file}")
except Exception as e:
    print(f"发生错误: {str(e)}")
