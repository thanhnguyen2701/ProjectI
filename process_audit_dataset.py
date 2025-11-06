import os, json, re

dataset_dir = r"C:\Users\FPTSHOP\2025.1\ProjectI\sample-smart-contract-dataset"

output_path = os.path.join(dataset_dir, "processed_documents.jsonl")

def clean_markdown(text: str) -> str:
    """Làm sạch markdown và link."""
    if not isinstance(text, str):
        return ""
    # Xóa liên kết [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Xóa markdown như **bold**, _italic_, `code`
    text = re.sub(r'[*_`#>]+', '', text)
    # Chuẩn hóa space " "
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_from_json(data: dict) -> dict:
    """Trích xuất các trường quan trọng cho RAG."""
    return {
        "id": data.get("id"),
        "title": data.get("title", ""),
        "content": clean_markdown(data.get("content", "")),
        "impact": data.get("impact", ""),
        "firm": data.get("firm_name") or (data.get("auditfirms_auditfirm") or {}).get("name", ""),
        "protocol": data.get("protocol_name") or (data.get("protocols_protocol") or {}).get("name", ""),
        "date": data.get("report_date", ""),
        "source": data.get("source_link", "")
    }

processed = []

# Duyệt toàn bộ file JSON trong thư mục (kể cả thư mục con)
for root, dirs, files in os.walk(dataset_dir):
    for file_name in files:
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(root, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Một số file có thể chứa danh sách findings
            if isinstance(data, list):
                for item in data:
                    processed.append(extract_from_json(item))
            elif isinstance(data, dict):
                processed.append(extract_from_json(data))
        except Exception as e:
            print(f"Lỗi đọc {file_name}: {e}")

# Lưu thành JSONL
with open(output_path, "w", encoding="utf-8") as out_f:
    for doc in processed:
        json.dump(doc, out_f, ensure_ascii=False)
        out_f.write("\n")

print(f"Đã xử lý {len(processed)} tài liệu.")
print(f"File đầu ra: {output_path}")
