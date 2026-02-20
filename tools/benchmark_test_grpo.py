import json
import os
from copy import deepcopy

# 输入文件路径 (建议使用上一生成的文件名 test_conv_meta_with_judge.jsonl)
input_path = './toolsciverifier/sciprm-bench/test_conv_meta_with_judge.jsonl'
# 输出文件路径
output_path = input_path.replace('.jsonl', '_grpo.jsonl')

def process_dataset(input_file, output_file):
    print(f"开始处理数据: {input_file}")
    
    total_conversations = 0
    total_grpo_samples = 0
    
    # 初始化全局 ID 计数器
    global_id_counter = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                messages = data.get('messages', [])
                
                # 保留 source
                source = data.get('source', 'Unknown')
                
                total_conversations += 1
                
                # 历史对话缓存
                history = []
                
                # 遍历消息
                for i, msg in enumerate(messages):
                    role = msg['role']
                    
                    if role == 'system':
                        history.append(msg)
                    
                    elif role == 'user':
                        history.append(msg)
                        
                    elif role == 'assistant':
                        # === 修改点: 提取 code_judge 状态 ===
                        # 从 assistant 的消息中获取该步骤是否为代码评判
                        is_code_judge = msg.get('code_judge', False)
                        
                        grpo_entry = {
                            "source": source,
                            "id": global_id_counter,  # 全局唯一 ID
                            "messages": deepcopy(history), # 输入给模型的历史 (System + User)
                            "solution": msg['content'],    # 模型的预期输出 (Valid/Invalid)
                            "code_judge": is_code_judge    # 新增字段：标记此样本是否为代码评判
                        }
                        
                        # 写入一行 JSONL
                        fout.write(json.dumps(grpo_entry, ensure_ascii=False) + '\n')
                        
                        # 更新计数器
                        total_grpo_samples += 1
                        global_id_counter += 1
                        
                        # 将 assistant 的回复加入历史，作为下一个样本的上下文
                        history.append(msg)
                        
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line")
                continue

    print(f"处理完成！")
    print(f"原始对话数量: {total_conversations}")
    print(f"生成的 GRPO 样本数量: {total_grpo_samples}")
    print(f"ID 范围: 0 到 {global_id_counter - 1}")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    # 简单的检查输入文件是否存在
    if os.path.exists(input_path):
        process_dataset(input_path, output_path)
    else:
        print(f"错误: 找不到输入文件 {input_path}")
