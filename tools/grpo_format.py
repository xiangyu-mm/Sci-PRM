import json
import os
from copy import deepcopy

# 输入文件路径
input_path = './toolsciverifier/sciprm-bench/train_conv.jsonl'
# 输出文件路径 (在同目录下生成 _grpo 后缀的文件)
output_path = input_path.replace('.jsonl', '_grpo.jsonl')

def process_dataset(input_file, output_file):
    print(f"开始处理数据: {input_file}")
    
    total_conversations = 0
    total_grpo_samples = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                messages = data.get('messages', [])
                total_conversations += 1
                
                # 历史对话缓存，初始包含 system prompt (如果有)
                history = []
                
                # 遍历消息
                for i, msg in enumerate(messages):
                    role = msg['role']
                    
                    if role == 'system':
                        history.append(msg)
                    
                    elif role == 'user':
                        history.append(msg)
                        
                    elif role == 'assistant':
                        # 当遇到 assistant 的回复时，这构成了一个完整的 GRPO 训练样本
                        # Input (messages): 当前 history (包含刚才的 user 输入)
                        # Extra (solution): assistant 的真实回复 (Valid/Invalid)
                        
                        grpo_entry = {
                            "messages": deepcopy(history),
                            # 这里将真实标签放入 solution 字段，ORM 可以直接读取 kwargs['solution']
                            "solution": msg['content'] 
                        }
                        
                        # 写入一行 JSONL
                        fout.write(json.dumps(grpo_entry, ensure_ascii=False) + '\n')
                        total_grpo_samples += 1
                        
                        # 将 assistant 的回复加入历史，作为下一轮对话的上下文
                        history.append(msg)
                        
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line")
                continue

    print(f"处理完成！")
    print(f"原始对话数量: {total_conversations}")
    print(f"生成的 GRPO 样本数量: {total_grpo_samples}")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    process_dataset(input_path, output_path)
