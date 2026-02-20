# -*- coding: utf-8 -*-
import json
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_FILE = "./dataset/mol_instruction_step/output_with_ref/catalytic_activity_reasoning.jsonl"
OUTPUT_FILE = "./dataset/mol_instruction_step/output_with_ref/catalytic_activity_executed_mock.jsonl"
MAX_WORKERS = 8  # 线程数
# ===========================================

def get_ncbi_details(uniprot_id):
    """
    策略 A: 查询 NCBI IPG 数据库 (优先)
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    try:
        # 1. Search IPG
        search_url = f"{base_url}/esearch.fcgi"
        search_params = {"db": "ipg", "term": uniprot_id, "retmode": "json"}
        
        try:
            resp = requests.get(search_url, params=search_params, timeout=5)
        except requests.RequestException:
            return None, None 

        if resp.status_code != 200: 
            return None, None
        
        data = resp.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        
        if not id_list: 
            return None, None 

        # 2. Get Summary
        summary_url = f"{base_url}/esummary.fcgi"
        summary_params = {"db": "ipg", "id": ",".join(id_list), "retmode": "json"}
        
        try:
            resp = requests.get(summary_url, params=summary_params, timeout=5)
        except requests.RequestException:
            return None, None

        if resp.status_code != 200: 
            return None, None
        
        summary_data = resp.json()
        result_dict = summary_data.get("result", {})
        
        for uid in id_list:
            if uid not in result_dict: continue
            record = result_dict[uid]
            
            # 提取 Accession
            acc = record.get("accession", "")
            if "." in acc: acc = acc.split(".")[0] 
            
            # 提取 Description
            title = record.get("title", "")
            if not title:
                p = record.get("product", "Unknown protein")
                o = record.get("organism", "Unknown organism")
                title = f"{p} [{o}]"
            
            return acc, title 

    except Exception:
        return None, None

    return None, None

def get_uniprot_details(uniprot_id):
    """
    策略 B: 查询 UniProt API (兜底)
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    try:
        try:
            resp = requests.get(url, timeout=10)
        except requests.RequestException:
            return None, None

        if resp.status_code != 200: 
            return None, None
        
        data = resp.json()
        
        # 1. 提取描述
        try:
            rec_name = data['proteinDescription']['recommendedName']['fullName']['value']
        except KeyError:
            try:
                rec_name = data['proteinDescription']['submissionNames'][0]['fullName']['value']
            except:
                rec_name = "Protein"
        
        try:
            organism = data['organism']['scientificName']
        except KeyError:
            organism = "Organism"
            
        description = f"{rec_name} [{organism}]"
        
        # 2. 提取 Accession
        refseq_acc = None
        for xref in data.get('uniProtKBCrossReferences', []):
            if xref['database'] == 'RefSeq':
                refseq_acc = xref['id']
                if refseq_acc.startswith("WP_"): break 
        
        if not refseq_acc:
            refseq_acc = uniprot_id 
            
        if "." in refseq_acc: refseq_acc = refseq_acc.split(".")[0]
        
        return refseq_acc, description

    except Exception:
        return None, None

def fetch_protein_info(uniprot_id):
    """
    组合策略逻辑
    """
    # 1. 尝试 NCBI
    acc, desc = get_ncbi_details(uniprot_id)
    # print("DEBUG: NCBI Result", desc) # 如果觉得刷屏可以注释掉
    if acc and desc:
        return acc, desc, "NCBI_IPG"
    
    # 2. 如果失败，尝试 UniProt
    acc, desc = get_uniprot_details(uniprot_id)
    if acc and desc:
        return acc, desc, "UniProt_Fallback"
    
    # 3. 都失败
    return None, None, "Failed"

def process_item(line):
    try:
        item = json.loads(line.strip())
    except:
        return None

    out = dict(item)
    steps = out.get("parsed_steps", [])
    
    # 如果没有 steps，直接跳过
    if not steps:
        out["exec_status"] = "no_steps_to_exec"
        out["executed_steps"] = []
        return json.dumps(out, ensure_ascii=False)

    executed_steps = []
    any_execution = False
    
    # 获取 ID
    protein_accession = item.get("metadata", {}).get("protein_accession", "")
    
    # 预取信息
    final_acc = None
    final_desc = None
    source = "None"
    
    if protein_accession:
        final_acc, final_desc, source = fetch_protein_info(protein_accession)

    for st in steps:
        st2 = dict(st)
        
        # 补全默认字段
        if "tool_used" not in st2: st2["tool_used"] = False
        if "tool_type" not in st2: st2["tool_type"] = "none"
        if "tool_details" not in st2: st2["tool_details"] = ""
        st2["tool_output"] = ""
        st2["exec_status"] = "skipped"

        # 拦截 python_code
        if st2.get("tool_used") and st2.get("tool_type") == "python_code":
            code = st2.get("tool_details", "")
            if code.strip():
                if final_acc and final_desc:
                    # 构造伪造输出
                    mock_output = (
                        f"Querying NCBI BLASTp database...\n"
                        f"Top Hit Accession: {final_acc}\n"
                        f"Top Hit Description: {final_desc}\n"
                        f"E-value: 0.0"
                    )
                    st2["tool_output"] = mock_output
                    st2["exec_status"] = "success"
                    any_execution = True
                else:
                    st2["tool_output"] = f"No homologous proteins found for accession: {protein_accession}."
                    st2["exec_status"] = "error"
            else:
                st2["exec_status"] = "empty_code"
        
        executed_steps.append(st2)

    out["executed_steps"] = executed_steps
    out["exec_status"] = "finished" if any_execution else "skipped_no_code"
    
    return json.dumps(out, ensure_ascii=False)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Not found {INPUT_FILE}")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Strategy: NCBI IPG -> (fallback) -> UniProt API")

    # 1. 读取所有行（用于进度条总数）
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 2. 打开输出文件（"w"模式会清空旧文件，如需追加请改用"a"）
    # 将文件对象保持打开状态，直到所有任务完成
    print("Starting processing and real-time writing...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交任务
            futures = {executor.submit(process_item, line): line for line in lines}
            
            # 使用 as_completed 实时获取完成的任务
            for future in tqdm(as_completed(futures), total=len(lines), desc="Mocking BLAST"):
                try:
                    res = future.result()
                    if res:
                        # 3. 实时写入并刷新缓冲区
                        f_out.write(res + "\n")
                        f_out.flush() 
                except Exception as e:
                    print(f"Error processing item: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
