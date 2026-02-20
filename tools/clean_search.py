import json
import os
import re

# 输入和输出文件路径
input_path = "./toolsciverifier/results/msearth_open.answers.jsonl"
output_path = "./toolsciverifier/results/msearth_open.answers.filtered.jsonl"

# 编译正则表达式：用于匹配并删除 DOI 相关信息
# 逻辑：
# (?:,\s*)?       -> 可选的逗号和空格 (例如处理 ", DOI:...")
# (?:with\s+)?    -> 可选的 "with " (例如处理 "with DOI...")
# \bdoi\b         -> 单词 boundary 后的 "doi" (忽略大小写)
# \s*:?\s*        -> 可选的冒号和空格
# 10\.\d{4,}/     -> DOI 的标准开头 (10.xxxx/)
# [^,\s"'\)]+     -> 匹配后续字符直到遇到逗号、空格、引号或右括号
doi_pattern = re.compile(r'(?:,\s*)?(?:with\s+)?\bdoi\b\s*:?\s*10\.\d{4,}/[^,\s"\'\)]+', re.IGNORECASE)

def remove_doi_info(text):
    """如果输入是字符串，则去除其中的 DOI 信息；否则原样返回"""
    if isinstance(text, str):
        # 替换匹配到的 DOI 模式为空字符串，并去除首尾可能残留的多余空格
        cleaned = doi_pattern.sub('', text).strip()
        # 处理可能残留的句尾标点（如 "2020 ." -> "2020."）
        cleaned = cleaned.replace(' .', '.')
        return cleaned
    return text

def process_data():
    count_processed = 0
    count_saved = 0

    print(f"开始处理文件: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                count_processed += 1
            except json.JSONDecodeError:
                print(f"警告: 第 {count_processed + 1} 行 JSON 解析失败，已跳过。")
                continue

            # --- 条件 1: 筛选 grounded_generated_answer_parsed 存在的数据 ---
            grounded_parsed = data.get("grounded_generated_answer_parsed")
            if not grounded_parsed:
                continue

            # --- 安全获取 title (修复 AttributeError) ---
            title_raw = data.get("title")
            if not title_raw or not isinstance(title_raw, str):
                continue
            title = title_raw.strip()

            # --- 条件 2: Title 必须在 web_search 的 tool_output 中被查询到 ---
            steps = grounded_parsed.get("steps", [])
            is_title_found = False

            # 用于判断是否保留该条数据
            for step in steps:
                if step.get("tool_used") is True and step.get("tool_type") == "web_search":
                    tool_output = step.get("tool_output")
                    if tool_output and isinstance(tool_output, str):
                        if title.lower() in tool_output.lower():
                            is_title_found = True
                            # 既然已经找到了，不需要break，因为后续还要继续遍历steps做清洗操作
                            # 但为了性能和逻辑分离，我们先标记，下面统一清洗

            if is_title_found:
                # --- 新增操作: 清洗 DOI 信息 ---
                # 遍历步骤，清洗 tool_output 和 reasoning_process
                for step in steps:
                    if "tool_output" in step:
                        step["tool_output"] = remove_doi_info(step["tool_output"])
                    if "reasoning_process" in step:
                        step["reasoning_process"] = remove_doi_info(step["reasoning_process"])
                
                # 清洗 final_result
                if "final_result" in grounded_parsed:
                    grounded_parsed["final_result"] = remove_doi_info(grounded_parsed["final_result"])
                
                # 更新回 data 对象 (其实 steps 是引用，已经更新了，但为了保险)
                data["grounded_generated_answer_parsed"] = grounded_parsed

                # --- 条件 3: 去掉无用的 raw 和 baseline 信息 ---
                keys_to_remove = [
                    "grounded_generated_answer_raw", 
                    "baseline_generated_answer_raw", 
                    "baseline_generated_answer_parsed"
                ]
                
                for key in keys_to_remove:
                    data.pop(key, None)

                # 写入新文件
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                count_saved += 1

    print(f"处理完成。")
    print(f"共扫描行数: {count_processed}")
    print(f"符合条件并保存的行数: {count_saved}")
    print(f"输出文件路径: {output_path}")

if __name__ == "__main__":
    process_data()
