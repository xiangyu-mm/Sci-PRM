import json
import os
import re

def parse_chain_general(data_dict):
    """
    通用解析函数，用于处理 grounded 或 baseline 中的 steps。
    支持两种格式：
    1. 新格式：{"steps": [{"step_id": 1...}, {...}]} (List)
    2. 旧格式：{"step1": {...}, "step2": {...}} (Dict keys)
    """
    chain = []
    
    # --- 情况 A: 新格式 (List inside "steps") ---
    if "steps" in data_dict and isinstance(data_dict["steps"], list):
        for idx, step_info in enumerate(data_dict["steps"], 1):
            if not isinstance(step_info, dict):
                continue
                
            # 格式化 tool_details
            tool_details = step_info.get("tool_details", "")
            if isinstance(tool_details, (dict, list)):
                tool_details = json.dumps(tool_details)
            
            chain_item = {
                "step_id": idx, # 列表顺序即为步骤ID
                "tool_used": step_info.get("tool_used", False),
                "tool_type": step_info.get("tool_type", "unknown"),
                "tool_details": tool_details,
                "reasoning": step_info.get("reasoning_process", ""),
                "tool_output": step_info.get("tool_output", ""),
                "exec_status": "success" if step_info.get("tool_output") else "unknown",
                "step_label": None,
                "label_reason": None
            }
            chain.append(chain_item)
        return chain

    # --- 情况 B: 旧格式 (Flattened keys "step1", "step2"...) ---
    step_keys = [k for k in data_dict.keys() if k.startswith("step")]
    
    # 按数字排序
    step_keys.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 999)

    step_counter = 1
    for key in step_keys:
        step_info = data_dict[key]
        if not isinstance(step_info, dict):
            continue
            
        tool_details = step_info.get("tool_details", "")
        if isinstance(tool_details, (dict, list)):
            tool_details = json.dumps(tool_details)
        
        chain_item = {
            "step_id": step_counter,
            "tool_used": step_info.get("tool_used", False),
            "tool_type": step_info.get("tool_type", "unknown"),
            "tool_details": tool_details,
            "reasoning": step_info.get("reasoning_process", ""),
            "tool_output": step_info.get("tool_output", ""),
            "exec_status": "success" if step_info.get("tool_output") else "unknown",
            "step_label": None,
            "label_reason": None
        }
        chain.append(chain_item)
        step_counter += 1

    return chain

def parse_grounded_chain(grounded_data):
    """
    解析 grounded_generated_answer_parsed
    """
    data_to_process = grounded_data
    
    # 处理 raw_text 嵌套的情况
    if isinstance(grounded_data, dict) and "raw_text" in grounded_data:
        try:
            raw_text = grounded_data["raw_text"]
            data_to_process = json.loads(raw_text)
        except Exception:
            return []

    if not isinstance(data_to_process, dict):
        return []

    # 调用通用解析逻辑
    return parse_chain_general(data_to_process)

def parse_baseline_chain(baseline_data):
    """
    解析 baseline_generated_answer_parsed
    """
    if not baseline_data or not isinstance(baseline_data, dict):
        return []

    # 1. 尝试使用通用逻辑解析 (应对 baseline 也变成了 steps 列表的情况)
    chain = parse_chain_general(baseline_data)
    if chain:
        return chain

    # 2. 如果通用逻辑没解析出东西，回退到旧的 Baseline 格式 (直接是 parameters 字段)
    if "parameters" in baseline_data or "name" in baseline_data:
        params = baseline_data.get("parameters", {})
        tool_details = json.dumps(params)
        
        chain_item = {
            "step_id": 1,
            "tool_used": True,
            "tool_type": baseline_data.get("name", "unknown"),
            "tool_details": tool_details,
            "reasoning": "Baseline generation typically performs a direct tool call.",
            "tool_output": "N/A", 
            "exec_status": "success",
            "step_label": None,
            "label_reason": None
        }
        return [chain_item]
    
    return []

def process_file(input_path, output_path):
    print(f"Processing input: {input_path}")
    
    total_read = 0
    grounded_kept = 0
    baseline_kept = 0
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
            
            total_read += 1
            try:
                row = json.loads(line)
                
                common_question = row.get("query", "")
                common_answer = row.get("response", "")
                base_id = row.get("question_id", "unknown")
                
                # --- Grounded ---
                grounded_parsed = row.get("grounded_generated_answer_parsed")
                if grounded_parsed:
                    grounded_chain = parse_grounded_chain(grounded_parsed)
                    if grounded_chain:
                        grounded_record = {
                            "source": "ToolSciVerifier",
                            "id": f"{base_id}_grounded",
                            "reasoning_chain_labeled": grounded_chain,
                            "question": common_question,
                            "answer": common_answer
                        }
                        f_out.write(json.dumps(grounded_record, ensure_ascii=False) + "\n")
                        grounded_kept += 1

                # --- Baseline ---
                baseline_parsed = row.get("baseline_generated_answer_parsed")
                if baseline_parsed:
                    baseline_chain = parse_baseline_chain(baseline_parsed)
                    if baseline_chain:
                        baseline_record = {
                            "source": "ToolSciVerifier",
                            "id": f"{base_id}_baseline",
                            "reasoning_chain_labeled": baseline_chain,
                            "question": common_question,
                            "answer": common_answer
                        }
                        f_out.write(json.dumps(baseline_record, ensure_ascii=False) + "\n")
                        baseline_kept += 1
                
                if total_read % 1000 == 0:
                    print(f"Read {total_read} lines... (Grounded kept: {grounded_kept}, Baseline kept: {baseline_kept})")
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing line {total_read}: {e}")

    print("-" * 30)
    print(f"Processing complete.")
    print(f"Total lines read: {total_read}")
    print(f"Grounded entries saved: {grounded_kept}")
    print(f"Baseline entries saved: {baseline_kept}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    input_file = "./toolsciverifier/results/msearth_open.answers.jsonl"
    output_file = "./toolsciverifier/results/msearth_open_processed_mol_instruct.jsonl"
    
    if os.path.exists(input_file):
        process_file(input_file, output_file)
    else:
        print(f"Error: Input file does not exist: {input_file}")
