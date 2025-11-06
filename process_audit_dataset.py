import os
import re
import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from PyPDF2 import PdfReader
from io import BytesIO

# --- Cấu hình ---
dataset_dir = r"C:\Users\FPTSHOP\2025.1\ProjectI\sample-smart-contract-dataset"
output_path = os.path.join(dataset_dir, "processed_documents.jsonl")

# --- HÀM HỖ TRỢ ---

def clean_markdown(text: str) -> str:
    """Làm sạch markdown và link."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)     # bỏ [text](url)
    text = re.sub(r'[*_`#>]+', '', text)                     # bỏ markdown
    text = re.sub(r'\s+', ' ', text).strip()                 # chuẩn hóa space
    return text


def fetch_text_from_url(url: str) -> str:
    """
    Truy cập URL và trả về text thuần.
    - Nếu là PDF: dùng PyPDF2 đọc.
    - Nếu là trang web (HTML): dùng BeautifulSoup để lấy text.
    - Nếu lỗi hoặc không thể tải, trả về chuỗi rỗng.
    """
    if not url or not isinstance(url, str):
        return ""

    # --- Chuyển đổi link GitHub blob -> raw ---
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

    try:
        headers = {"User-Agent": "Mozilla/5.0 (RAG Dataset Builder)"}
        resp = requests.get(url, headers=headers, timeout=20)

        if resp.status_code != 200:
            return ""

        content_type = resp.headers.get("Content-Type", "")

        # --- Nếu là PDF ---
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            try:
                pdf_reader = PdfReader(BytesIO(resp.content))
                text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
                return clean_markdown(text)
            except Exception as e:
                print(f"[!] Không thể đọc PDF, fallback sang HTML: {url} ({e})")

        # --- Nếu là HTML hoặc fallback ---
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        return clean_markdown(text)

    except Exception as e:
        print(f"Lỗi tải {url}: {e}")
    return ""


def extract_from_json(data: dict) -> dict:
    """Trích xuất trường chính cho RAG."""
    # chọn link tốt nhất để fetch nội dung
    link = data.get("source_link") or data.get("github_link") or data.get("pdf_link")

    external_text = fetch_text_from_url(link) if link else ""

    return {
        "id": data.get("id"),
        "title": data.get("title", ""),
        "kind": data.get("kind", ""),
        "summary": clean_markdown(data.get("summary", "")),
        "content": clean_markdown(data.get("content", "")) + "\n" + external_text,
        "impact": data.get("impact", "").upper(),
        "firm": data.get("firm_name") or (data.get("auditfirms_auditfirm") or {}).get("name", ""),
        "protocol": data.get("protocol_name") or (data.get("protocols_protocol") or {}).get("name", ""),
        "date": data.get("report_date", ""),
        "source": link or "",
        "slug": data.get("slug", "")
    }

# --- XỬ LÝ TOÀN BỘ FILE ---

processed = []
for root, dirs, files in os.walk(dataset_dir):
    for file_name in tqdm(files, desc="Processing JSON files"):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(root, file_name)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    processed.append(extract_from_json(item))
            elif isinstance(data, dict):
                processed.append(extract_from_json(data))

        except Exception as e:
            print(f"Lỗi đọc {file_name}: {e}")

# --- Lưu ra JSONL ---
with open(output_path, "w", encoding="utf-8") as out_f:
    for doc in processed:
        json.dump(doc, out_f, ensure_ascii=False)
        out_f.write("\n")

print(f"\n✅ Đã xử lý {len(processed)} tài liệu.")
print(f"📄 File đầu ra: {output_path}")
