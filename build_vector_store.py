import os, json, re
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from tiktoken import get_encoding
from dotenv import load_dotenv

load_dotenv()

os.environ["CHROMA_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Đường dẫn tới file JSONL bạn đã tạo ở bước trước
jsonl_path = r"C:\Users\FPTSHOP\2025.1\ProjectI\sample-smart-contract-dataset\processed_documents.jsonl"
chroma_path = r"C:\Users\FPTSHOP\2025.1\ProjectI\chroma_db"

# ---------------------
# 1️⃣ HÀM CHIA CHUNK
# ---------------------
def chunk_text(text, max_tokens=500, overlap=100):
    enc = get_encoding("cl100k_base")  # tokenizer của OpenAI
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start += max_tokens - overlap
    return chunks

# ---------------------
# 2️⃣ KHỞI TẠO CHROMA
# ---------------------
client = chromadb.PersistentClient(path=chroma_path)

# Sử dụng embedding của OpenAI (cần API key)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small")

collection = client.get_or_create_collection(
    name="smart_contract_audits",
    embedding_function=openai_ef
)

# ---------------------
# 3️⃣ ĐỌC FILE JSONL
# ---------------------
documents = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        if doc.get("content"):
            documents.append(doc)

print(f"📄 Đọc {len(documents)} tài liệu gốc.")

# ---------------------
# 4️⃣ CHUNK + THÊM VÀO CHROMA
# ---------------------
ids, texts, metadatas = [], [], []

for i, doc in enumerate(tqdm(documents)):
    chunks = chunk_text(doc["content"], max_tokens=500, overlap=100)
    for j, chunk in enumerate(chunks):
        ids.append(f"{doc['id']}_{j}")
        texts.append(chunk)
        metadatas.append({
            "id": doc["id"],
            "title": doc["title"],
            "impact": doc["impact"],
            "firm": doc["firm"],
            "protocol": doc["protocol"],
            "date": doc["date"],
            "source": doc["source"]
        })

# ---------------------
# 5️⃣ CHIA NHỎ KHI THÊM VÀO CHROMA
# ---------------------
batch_size = 100  # Số chunk xử lý mỗi lần (bạn có thể tăng lên 200 nếu muốn nhanh hơn)
for i in range(0, len(texts), batch_size):
    batch_ids = ids[i:i+batch_size]
    batch_texts = texts[i:i+batch_size]
    batch_meta = metadatas[i:i+batch_size]
    collection.add(
        ids=batch_ids,
        documents=batch_texts,
        metadatas=batch_meta
    )
    print(f"✅ Đã thêm {i + len(batch_texts)} / {len(texts)} chunks")

print(f"✅ Đã thêm {len(texts)} chunks vào ChromaDB.")
print(f"💾 Database lưu tại: {chroma_path}")
