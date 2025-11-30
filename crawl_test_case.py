import os
import json
import random
import re
import requests
from difflib import SequenceMatcher
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# === Load OpenAI ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# === Config ===
DATASET_DIR = r"C:\Users\FPTSHOP\2025.1\ProjectI\sample-smart-contract-dataset"
OUTPUT_PATH = r"C:\Users\FPTSHOP\2025.1\ProjectI\test_cases.jsonl"
NUM_SAMPLES = 150


# --- Helper: convert github to raw ---
def to_raw_url(url: str) -> str:
    if not isinstance(url, str):
        return ""
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url


# --- Download raw markdown ---
def fetch_raw_content(url: str) -> str:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and len(r.text) > 50:
            return r.text
        print(f"[!] Không tải được markdown: {url}, status {r.status_code}")
    except Exception as e:
        print(f"[!] Lỗi tải markdown {url}: {e}")
    return ""


# --- Match issue title from markdown sections ---
def find_matching_section(md: str, title: str):
    """
    Return the markdown section closest to the issue `title`.
    """
    lines = md.split("\n")
    sections = {}
    current = None
    buffer = []

    for line in lines:
        if re.match(r"^#{1,4}\s", line):  # detect section
            if current:
                sections[current] = "\n".join(buffer)
            current = line.strip("# ").strip()
            buffer = []
        else:
            buffer.append(line)

    if current and buffer:
        sections[current] = "\n".join(buffer)

    # fuzzy match by similarity
    best_title = None
    best_score = 0

    for sec_title in sections.keys():
        score = SequenceMatcher(None, sec_title.lower(), title.lower()).ratio()
        if score > best_score:
            best_score = score
            best_title = sec_title

    if best_score < 0.6:
        return None, None  # not similar enough

    return best_title, sections[best_title]


# --- Extract Q/A using OpenAI but on small sections ---
def extract_qa_from_section(section_md: str):
    prompt = f"""
Bạn là công cụ phân tích nội dung một issue của báo cáo audit.

Nhiệm vụ:
- Trích xuất "question" = phần mô tả lỗi (Description / Issue)
- Trích xuất "answers" = tất cả các phần Recommendation / Fix / Mitigation

Nếu có nhiều đề xuất sửa lỗi → trả về mảng answers.

Format JSON ONLY:
{{
  "question": "...",
  "answers": ["...", "..."]
}}

=== Markdown Section ===
{section_md}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    text = response.choices[0].message.content.strip()

    # cleanup JSON wrapping
    text = re.sub(r"^```json\s*|```$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```|```$", "", text).strip()

    try:
        data = json.loads(text)
        return data.get("question", ""), data.get("answers", [])
    except:
        print("[!] JSON parse failed:", text[:300])
        return "", []


# === Load dataset ===
all_docs = []
for root, _, files in os.walk(DATASET_DIR):
    for name in files:
        if name.endswith(".json"):
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    all_docs.append((path, data))
            except:
                continue

print("Tổng số documents:", len(all_docs))

# Chỉ lấy doc có link
valid_docs = [(p, d) for p, d in all_docs if d.get("source_link")]
samples = random.sample(valid_docs, min(NUM_SAMPLES, len(valid_docs)))
print("Samples:", len(samples))

# === MAIN PROCESS ===
output = []
used_files = set()

for file_path, doc in tqdm(samples, desc="🔎 Extracting issues"):
    url = doc["source_link"]
    raw_url = to_raw_url(url)

    md = fetch_raw_content(raw_url)
    if not md:
        continue

    # Step 1: find the correct markdown section by title
    found_title, section_md = find_matching_section(md, doc["title"])
    if not section_md:
        print(f"[!] Không tìm thấy section cho title: {doc['title']}")
        continue

    # Step 2: extract Q/A from that section
    question, answers = extract_qa_from_section(section_md)
    if not question:
        print("[!] Không tìm thấy question trong section")
        continue

    output.append({
        "id": doc.get("id"),
        "title": found_title,
        "original_title": doc["title"],
        "source": url,
        "raw_url": raw_url,
        "question": question,
        "answers": answers,
        "impact": doc.get("impact", ""),
        "firm": doc.get("firm_name", ""),
        "protocol": doc.get("protocol_name", ""),
    })

    used_files.add(file_path)

print("Tổng số test case:", len(output))

# Save JSONL
with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
    for item in output:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print("Đã lưu vào:", OUTPUT_PATH)
