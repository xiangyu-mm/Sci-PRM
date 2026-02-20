#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import ast
import re
from collections import Counter
from pathlib import Path

INPUT = Path("./toolsciverifier/results/python_tool2.jsonl")

def iter_strings(obj):
    """递归遍历任意 JSON 对象，产出其中所有字符串字段。"""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_strings(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_strings(it)

def extract_imports_from_code(code: str):
    """
    从一段 Python 代码中提取 import 的顶层包名。
    返回 set[str]
    """
    pkgs = set()

    # 去掉常见 markdown 代码围栏，提升 parse 成功率
    code = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", code.strip())
    code = re.sub(r"\s*```$", "", code)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return pkgs

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0]
                if name:
                    pkgs.add(name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                name = node.module.split(".")[0]
                if name:
                    pkgs.add(name)
    return pkgs

def looks_like_python(code: str):
    """粗略判断一段字符串是否像 Python 代码，避免全量字符串都 parse。"""
    s = code
    return ("import " in s) or ("from " in s and " import " in s) or ("def " in s) or ("torch." in s)

def main():
    pkg_counter = Counter()
    total_lines = 0
    parsed_lines = 0

    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 在整条 JSON 里找“像 python 的字符串”，并抽取 import
            found_any = False
            for s in iter_strings(obj):
                if looks_like_python(s):
                    pkgs = extract_imports_from_code(s)
                    if pkgs:
                        found_any = True
                        for p in pkgs:
                            pkg_counter[p] += 1
            if found_any:
                parsed_lines += 1

    print(f"Input file: {INPUT}")
    print(f"Total jsonl lines: {total_lines}")
    print(f"Lines containing parsed imports: {parsed_lines}")
    print(f"Unique python packages: {len(pkg_counter)}")
    print("\nTop packages by occurrence (per code-block occurrence):")
    for pkg, cnt in pkg_counter.most_common(50):
        print(f"{pkg}\t{cnt}")

if __name__ == "__main__":
    main()
