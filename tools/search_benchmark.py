import json
import os

# 配置路径
input_file = './toolsciverifier/results_stage3/search_data.jsonl'
output_file = './toolsciverifier/results_stage3/verifier_benchmark_dataset.jsonl'

processed_count = 0
extracted_samples = 0

print(f"正在从 {input_file} 提取测试样本...")

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
            
            # 1. 基础筛选：只处理处理成功的条目
            if data.get("generation_status") != "success" or data.get("verify_status") != "processed":
                continue
            
            processed_count += 1
            
            # 获取上下文信息
            context_info = {
                "question_id": data.get("question_id"),
                "query": data.get("query"),
                "caption": data.get("caption", ""),
                "title": data.get("title", ""),
                "subject": data.get("subject", "")
            }
            
            # 获取解析后的步骤
            steps = data.get("generated_answer_parsed", [])
            history_text = [] # 用于记录之前的步骤
            
            # 2. 遍历步骤，提取有 verification_result 的步骤
            for step in steps:
                step_id = step.get("step_id")
                reasoning = step.get("reasoning_process", "")
                
                # 检查该步骤是否有验证结果（作为 Ground Truth）
                if "verification_result" in step:
                    
                    # 构建测试样本
                    test_sample = {
                        "unique_id": f"{context_info['question_id']}_step_{step_id}",
                        "meta_data": context_info,
                        
                        # 输入部分：模型验证所需的全部信息
                        "input": {
                            "step_history": history_text.copy(), # 之前的步骤历史
                            "current_step": {
                                "step_id": step_id,
                                "tool_type": step.get("tool_type"),
                                "tool_details": step.get("tool_details"), # 工具输入/Query
                                "tool_output": step.get("tool_output"),   # 工具输出/检索结果
                                "reasoning": reasoning
                            }
                        },
                        
                        # 输出部分：作为标签 (Label)
                        "label": step["verification_result"]
                    }
                    
                    # 写入文件
                    fout.write(json.dumps(test_sample, ensure_ascii=False) + '\n')
                    extracted_samples += 1
                
                # 将当前步骤加入历史，供后续步骤参考
                # 格式化为文本，模拟模型生成的上下文
                history_str = f"Step {step_id}: {reasoning}"
                if step.get("tool_used"):
                    history_str += f" | Tool: {step.get('tool_type')} | Input: {step.get('tool_details')} | Output: {step.get('tool_output')}"
                history_text.append(history_str)

        except json.JSONDecodeError:
            continue

print(f"处理完成！")
print(f"扫描原始对话数: {processed_count}")
print(f"生成测试样本数: {extracted_samples}")
print(f"保存路径: {output_file}")
