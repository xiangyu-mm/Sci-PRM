# -*- coding: utf-8 -*-
"""
Read a JSON array file.
For each item, generate TWO answers:
  1) baseline: only uses question (may retrieve wrong paper)
  2) grounded: uses question + paper + reference text (+ keywords, output_requirements)
Both runs MUST use web_search first (enforced by prompt).
Output JSONL: keep ALL original fields + new baseline/grounded generated answer fields.

Run:
python generate_answers.py \
  --input-file /path/input.json \
  --output-file /path/out.jsonl \
  --ark-base-url https://ark.cn-beijing.volces.com/api/v3 \
  --ark-api-key YOUR_KEY \
  --model doubao-seed-1-6-251015 \
  --max-workers 4 \
  --limit 0
"""
import os
import json
import time
import re
import ujson
from tqdm import tqdm
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


# -------------------------
# Client (Ark style)
# -------------------------
def build_client(api_key: str, base_url: str):
    return OpenAI(base_url=base_url, api_key=api_key)


# -------------------------
# Tool config
# -------------------------
TOOLS = [{
    "type": "web_search",
    "max_keyword": 20,
    "limit": 30,
}]


# -------------------------
# Prompts
# -------------------------
SYSTEM_PROMPT = """
You are a professional scientific researcher. Your task is to answer scientific questions by first retrieving relevant academic papers and then using appropriate tools for step-by-step reasoning.
"""

USER_PROMPT_TEMPLATE = """
Question: {question}

This question is from a scientific paper. You need to first retrieve the relevant paper before answering. Necessary information for answering the question may be missing and needs to be found from the paper; the retrieved information must be cited with paper title, year and DOI/URL.

Answer the question step by step. You can use various tools when answering, such as web search, python code, scientific APIs, etc.; when using tools, you must provide specific usage methods:

For web search: provide the exact query string you used.
For python code: provide the complete, runnable code snippet.
For no tool: set tool_type to "none".
Output the answer in STRICT JSON format. Each step must include the following fields:

tool_used: boolean (true/false)
tool_type: string ("web_search"/"python_code"/"scientific_api"/"none")
tool_details: string (exact search query / full code / "none")
tool_output: string (retrieved paper info / code execution result / "none")
reasoning_process: string (detailed logical deduction for the step, include paper citations)
Additionally, add a "final_result" field at the root level to present the final answer to the question.

Return STRICT JSON only (no extra text).
"""

USER_PROMPT_GROUNDED_TEMPLATE = """
Question: {question}

Target Paper (field 'paper'): {paper}
The Evidence text in the Target Paper (field 'reference text'): {ref_text}

You MUST:
1.  Propose 2–5 candidate web search query strings. Queries must be derived **exclusively from the content in the Question**; the Paper title hint and Reference text hint MUST NOT be included in any query.
2.  Execute web search using one or more of the proposed queries, and iterate until you retrieve a result whose title **exactly matches the Paper title hint**.
3.  Cite the matched paper with three core elements: full title, publication year, and a valid DOI or official URL.
4.  Answer the Question in a step-by-step manner, and explicitly include details of all tool usage in each step.
5.  The Target Paper (corresponding to the Paper title hint) is **for verification purposes only**—to confirm whether the retrieved paper is correct. It must NOT be used as a query for paper retrieval. Additionally, the Paper title hint must be excluded from the reasoning process entirely.
6.  The Evidence text in the Target Paper refers to the content within the paper corresponding to the Paper title hint. This evidence text can be used as a reference for reasoning **only after the web search is completed and the correct paper is confirmed**.

Output the answer in STRICT JSON format. Each step must include the following fields:

tool_used: boolean (true/false)
tool_type: string ("web_search"/"python_code"/"scientific_api"/"none")
tool_details: string (exact search query / full code / "none")
tool_output: string (retrieved paper info / code execution result / "none")
reasoning_process: string (detailed logical deduction for the step, include paper citations)
Additionally, add a "final_result" field at the root level to present the final answer to the question.

Return STRICT JSON only (no extra text).
"""


# -------------------------
# JSON extract & cleanup
# -------------------------
def remove_trailing_commas(s: str) -> str:
    """Remove trailing commas before } or ] in JSON-like strings."""
    return re.sub(r",\s*([}\]])", r"\1", s)


def extract_json_from_text(text: str):
    """
    Extract JSON object from model output.
    Strategy:
      1) If fenced ```json ... ``` exists, parse inside.
      2) Else fallback to first {...} block.
    """
    # 1) fenced code block
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if m:
        js_str = m.group(1)
    else:
        # 2) first JSON object
        m2 = re.search(r"{[\s\S]*}", text)
        if not m2:
            return False, text
        js_str = m2.group(0)

    try:
        js_str = remove_trailing_commas(js_str)
        return True, ujson.loads(js_str)
    except Exception:
        return False, text


# -------------------------
# Retry helper
# -------------------------
def try_request_with_retries(fn, max_retries=6, delay=2, **kwargs):
    """Retry wrapper with exponential backoff."""
    for i in range(max_retries):
        try:
            return fn(**kwargs), None
        except Exception as e:
            if i == max_retries - 1:
                return None, str(e)
            time.sleep(delay * (i + 1))
    return None, "unknown_error"


# -------------------------
# Model call
# -------------------------
def call_generator_model(client: OpenAI, model: str, system_prompt: str, user_prompt: str):
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}
        ],
        tools=TOOLS,
        temperature=0
    )

    if getattr(resp, "output_text", None):
        return resp.output_text

    # fallback: concatenate outputs
    parts = []
    try:
        for o in getattr(resp, "output", []) or []:
            content = getattr(o, "content", None)
            if not content:
                continue
            for c in content:
                if getattr(c, "type", "") in ("output_text", "text"):
                    parts.append(getattr(c, "text", ""))
        return "\n".join(parts) if parts else json.dumps({"error": "no output"})
    except Exception:
        return json.dumps({"error": "failed to extract output"})


# -------------------------
# Processor
# -------------------------
class AnswerGenerator:
    def __init__(self, client: OpenAI, input_file: str, output_file: str, model: str,
                 max_workers: int = 4, limit: int = 0):
        self.client = client
        self.input_file = input_file
        self.output_file = output_file
        self.model = model
        self.max_workers = max_workers
        self.limit = limit

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Input file must be a JSON array.")
        if limit and limit > 0:
            data = data[:limit]
        self.data_list = data

    def process_one(self, item: dict):
        """
        Keep ALL original fields, add:
          baseline_generated_answer_raw/parsed/status
          grounded_generated_answer_raw/parsed/status
        """
        out = dict(item)  # keep original fields

        out.update({
            "baseline_generated_answer_raw": "",
            "baseline_generated_answer_parsed": {},
            "baseline_generation_status": "success",   # success/error/parse_error

            "grounded_generated_answer_raw": "",
            "grounded_generated_answer_parsed": {},
            "grounded_generation_status": "success",   # success/error/parse_error
        })

        question = item.get("question", "")
        if not question:
            out["baseline_generation_status"] = "error"
            out["baseline_generated_answer_raw"] = "Empty question field"
            out["grounded_generation_status"] = "error"
            out["grounded_generated_answer_raw"] = "Empty question field"
            return out

        # ---------------- baseline ----------------
        baseline_user_prompt = USER_PROMPT_TEMPLATE.format(question=question)
        raw_output, err = try_request_with_retries(
            call_generator_model,
            max_retries=8,
            delay=2,
            client=self.client,
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=baseline_user_prompt
        )

        if raw_output is None:
            out["baseline_generation_status"] = "error"
            out["baseline_generated_answer_raw"] = f"Request failed: {err}"
        else:
            out["baseline_generated_answer_raw"] = raw_output
            ok, parsed = extract_json_from_text(raw_output)
            if ok and isinstance(parsed, dict):
                out["baseline_generated_answer_parsed"] = parsed
            else:
                out["baseline_generation_status"] = "parse_error"
                out["baseline_generated_answer_parsed"] = {
                    "parse_error": "Invalid JSON format",
                    "raw_text": raw_output
                }

        # ---------------- grounded ----------------
        keywords = item.get("keywords", "")
        if isinstance(keywords, list):
            keywords = ", ".join([str(x) for x in keywords])
        else:
            keywords = "" if keywords is None else str(keywords)

        grounded_user_prompt = USER_PROMPT_GROUNDED_TEMPLATE.format(
            question=question,
            paper=item.get("paper", ""),
            ref_text=item.get("reference text", "")
        )

        raw_output, err = try_request_with_retries(
            call_generator_model,
            max_retries=8,
            delay=2,
            client=self.client,
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=grounded_user_prompt
        )

        if raw_output is None:
            out["grounded_generation_status"] = "error"
            out["grounded_generated_answer_raw"] = f"Request failed: {err}"
        else:
            out["grounded_generated_answer_raw"] = raw_output
            ok, parsed = extract_json_from_text(raw_output)
            if ok and isinstance(parsed, dict):
                out["grounded_generated_answer_parsed"] = parsed
            else:
                out["grounded_generation_status"] = "parse_error"
                out["grounded_generated_answer_parsed"] = {
                    "parse_error": "Invalid JSON format",
                    "raw_text": raw_output
                }

        return out

    def run(self):
        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)

        with open(self.output_file, "w", encoding="utf-8") as f_out:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futures = [ex.submit(self.process_one, item) for item in self.data_list]
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=f"Generating answers: {self.model}"):
                    res = fut.result()
                    f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f_out.flush()


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str,
                        default="/mnt/shared-storage-user/sciprismax/zhaoxiangyu/deep_research/1_judge_v4_[gpt-5.1-2025-11-13].json")
    parser.add_argument("--output-file", type=str,
                        default="./toolsciverifier/results/deepresearch_preference_data.jsonl")
    parser.add_argument("--ark-base-url", type=str,
                        default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument("--ark-api-key", type=str, default=os.getenv("ARK_API_KEY", "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"))
    parser.add_argument("--model", type=str, default="doubao-seed-1-6-251015")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only process first N items.")
    args = parser.parse_args()

    if not args.ark_api_key:
        raise ValueError("Missing API key. Provide --ark-api-key or set ARK_API_KEY env var.")

    client = build_client(api_key=args.ark_api_key, base_url=args.ark_base_url)

    generator = AnswerGenerator(
        client=client,
        input_file=args.input_file,
        output_file=args.output_file,
        model=args.model,
        max_workers=args.max_workers,
        limit=args.limit
    )
    generator.run()


if __name__ == "__main__":
    main()
