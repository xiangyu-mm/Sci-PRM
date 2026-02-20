# -*- coding: utf-8 -*-
import os
import json
import time
import re
import base64
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
    "limit": 10,
}]


# -------------------------
# Prompts
# -------------------------
SYSTEM_PROMPT = """
You are a professional scientific researcher. Your task is to answer scientific questions by first retrieving relevant academic papers and then using appropriate tools for step-by-step reasoning.
"""

# Modified Prompt: Root is a JSON Array, NO final_result field.
USER_PROMPT_TEMPLATE = """
Question: {question}

You MUST:
1.  Propose 2–5 candidate web search query strings. Queries must be derived **exclusively from the content in the Question&captions**; 
2.  Execute web search using one or more of the proposed queries.
3.  Cite the matched paper with three core elements: full title, publication year, and a valid DOI or official URL.
4.  Answer the Question in a step-by-step manner, and explicitly include details of all tool usage in each step.

OUTPUT FORMAT (STRICT):
Return STRICT JSON only (no extra text).
The root MUST be a JSON ARRAY containing multiple step objects.
Each step object MUST include the following fields:
- step_id: number
- tool_used: boolean (true/false)
- tool_type: string ("web_search"/"none")
- tool_details: string (exact search query / "none")
- tool_output: string (retrieved paper info / code execution result / "none")
- reasoning_process: string (detailed logical deduction for the step, include paper citations)
"""


# -------------------------
# JSON extract & cleanup
# -------------------------
def remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)


def extract_json_from_text(text: str):
    # 1. Try extracting from Markdown code blocks first
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if m:
        js_str = m.group(1)
    else:
        # 2. Heuristic extraction: Find the first '[' and the last ']'
        # Since we expect a list, we prioritize '[' over '{'
        start_idx = -1
        end_idx = -1
        
        # Find start of list
        start_idx = text.find('[')
        
        # Find end of list (search backwards)
        end_idx = text.rfind(']')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            js_str = text[start_idx : end_idx + 1]
        else:
            # Fallback: try object just in case model messed up
            m2 = re.search(r"{[\s\S]*}", text)
            if m2:
                js_str = m2.group(0)
            else:
                return False, text

    try:
        js_str = remove_trailing_commas(js_str)
        return True, ujson.loads(js_str)
    except Exception:
        return False, text


# -------------------------
# Retry helper
# -------------------------
def try_request_with_retries(fn, max_retries=6, delay=2, **kwargs):
    for i in range(max_retries):
        try:
            return fn(**kwargs), None
        except Exception as e:
            if i == max_retries - 1:
                return None, str(e)
            time.sleep(delay * (i + 1))
    return None, "unknown_error"


# -------------------------
# Image helper (local file -> data URL)
# -------------------------
def guess_mime(path: str) -> str:
    p = path.lower()
    if p.endswith(".png"):
        return "image/png"
    if p.endswith(".jpg") or p.endswith(".jpeg"):
        return "image/jpeg"
    if p.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"


def image_file_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b = f.read()
    mime = guess_mime(path)
    enc = base64.b64encode(b).decode("utf-8")
    return f"data:{mime};base64,{enc}"


# -------------------------
# Parse question text from msearth "query"
# -------------------------
def extract_question_from_query(q: str) -> str:
    if not q:
        return ""
    # common pattern: "...Question:\n...."
    m = re.search(r"Question:\s*(.*)", q, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: strip leading <image>
    return q.replace("<image>", "").strip()


# -------------------------
# Model call (multimodal)
# -------------------------
def call_generator_model_mm(client: OpenAI, model: str, system_prompt: str, user_prompt: str,
                            image_data_urls=None):
    if image_data_urls is None:
        image_data_urls = []

    user_content = []
    # put images first
    for url in image_data_urls:
        user_content.append({"type": "input_image", "image_url": url})
    user_content.append({"type": "input_text", "text": user_prompt})

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ],
        tools=TOOLS,
        temperature=0
    )

    if getattr(resp, "output_text", None):
        return resp.output_text

    # fallback: concatenate outputs
    parts = []
    for o in getattr(resp, "output", []) or []:
        content = getattr(o, "content", None)
        if not content:
            continue
        for c in content:
            if getattr(c, "type", "") in ("output_text", "text"):
                parts.append(getattr(c, "text", ""))
    return "\n".join(parts) if parts else json.dumps({"error": "no output"})


# -------------------------
# Processor
# -------------------------
class AnswerGeneratorMMEarth:
    def __init__(self, client: OpenAI, input_file: str, output_file: str, model: str,
                 images_dir: str, max_workers: int = 4, limit: int = 0):
        self.client = client
        self.input_file = input_file
        self.output_file = output_file
        self.model = model
        self.images_dir = images_dir
        self.max_workers = max_workers
        self.limit = limit

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Input file must be a JSON array.")
        if limit and limit > 0:
            data = data[:limit]
        self.data_list = data

    def _load_images_as_data_urls(self, item: dict):
        imgs = item.get("images", []) or []
        urls = []
        for name in imgs:
            local_path = name
            if not os.path.isabs(local_path):
                local_path = os.path.join(self.images_dir, name)

            if not os.path.exists(local_path):
                continue
            urls.append(image_file_to_data_url(local_path))
        return urls

    def process_one(self, item: dict):
        out = dict(item)

        out.update({
            "generated_answer_raw": "",
            "generated_answer_parsed": [], # Default to empty list
            "generation_status": "success",
        })

        query = item.get("query", "")
        question = extract_question_from_query(query)

        if not question:
            out["generation_status"] = "error"
            out["generated_answer_raw"] = "Empty question/query field"
            return out

        image_data_urls = self._load_images_as_data_urls(item)

        # ---------------- Single Pass ----------------
        # Note: We only pass {question}. Implicitly, the model sees the images/captions as input_image.
        formatted_prompt = USER_PROMPT_TEMPLATE.format(
            question=question
        )

        raw_output, err = try_request_with_retries(
            call_generator_model_mm,
            max_retries=8,
            delay=2,
            client=self.client,
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=formatted_prompt,
            image_data_urls=image_data_urls
        )

        if raw_output is None:
            out["generation_status"] = "error"
            out["generated_answer_raw"] = f"Request failed: {err}"
        else:
            out["generated_answer_raw"] = raw_output
            ok, parsed = extract_json_from_text(raw_output)
            
            # We expect a List. 
            if ok and isinstance(parsed, list):
                out["generated_answer_parsed"] = parsed
            elif ok and isinstance(parsed, dict):
                # If model accidentally returned a dict (e.g. wrapped in {"steps": ...})
                # try to rescue it, otherwise just save the dict
                if "steps" in parsed and isinstance(parsed["steps"], list):
                     out["generated_answer_parsed"] = parsed["steps"]
                else:
                     out["generated_answer_parsed"] = parsed
            else:
                out["generation_status"] = "parse_error"
                out["generated_answer_parsed"] = {
                    "parse_error": "Invalid JSON format (Expected List)",
                    "raw_text": raw_output
                }

        return out

    def run(self):
        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)

        with open(self.output_file, "w", encoding="utf-8") as f_out:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futures = [ex.submit(self.process_one, item) for item in self.data_list]
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=f"Generating answers (MMEarth): {self.model}"):
                    res = fut.result()
                    f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f_out.flush()


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-file", type=str,
                        default="/mnt/shared-storage-user/sciprismax/zhaoxiangyu/MSEarth/msearth_open.json")
    parser.add_argument("--output-file", type=str,
                        default="./toolsciverifier/results/msearth_open_search.jsonl")
    parser.add_argument("--images-dir", type=str,
                        default="/mnt/shared-storage-user/sciprismax/zhaoxiangyu/MSEarth/mmearth_images")
    parser.add_argument("--ark-base-url", type=str,
                        default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument("--ark-api-key", type=str,
                        default=os.getenv("ARK_API_KEY", "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"))
    parser.add_argument("--model", type=str, default="doubao-seed-1-6-251015")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.ark_api_key:
        raise ValueError("Missing API key. Provide --ark-api-key or set ARK_API_KEY env var.")

    client = build_client(api_key=args.ark_api_key, base_url=args.ark_base_url)

    generator = AnswerGeneratorMMEarth(
        client=client,
        input_file=args.input_file,
        output_file=args.output_file,
        model=args.model,
        images_dir=args.images_dir,
        max_workers=args.max_workers,
        limit=args.limit
    )
    generator.run()


if __name__ == "__main__":
    main()
