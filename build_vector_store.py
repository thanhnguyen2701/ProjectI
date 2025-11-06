import os
import json
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from tiktoken import get_encoding
from dotenv import load_dotenv

# ======================
# 🔧 CẤU HÌNH BAN ĐẦU
# ======================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY chưa được đặt trong file .env")

os.environ["CHROMA_OPENAI_API_KEY"] = OPENAI_API_KEY

jsonl_path = r"C:\Users\FPTSHOP\2025.1\ProjectI\sample-smart-contract-dataset\processed_documents.jsonl"
chroma_path = r"C:\Users\FPTSHOP\2025.1\ProjectI\chroma_db"
collection_name = "smart_contract_audits"

# ======================
# ⚙️ HÀM HỖ TRỢ
# ======================

def chunk_text(text: str, max_tokens: int = 800, overlap: int = 100):
    """
    Chia văn bản thành các đoạn nhỏ theo token (phù hợp với text-embedding-3-small)
    """
    enc = get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - overlap

    return chunks


# ======================
# 🧠 KHỞI TẠO CHROMADB
# ======================

client = chromadb.PersistentClient(path=chroma_path)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(model_name="text-embedding-3-small")

collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_ef
)

# ======================
# 📚 ĐỌC FILE JSONL
# ======================

if not os.path.exists(jsonl_path):
    raise FileNotFoundError(f"❌ Không tìm thấy file JSONL tại: {jsonl_path}")

documents = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        if doc.get("content"):
            documents.append(doc)

print(f"📄 Đọc {len(documents)} tài liệu gốc từ {jsonl_path}")

# ======================
# 🧩 XỬ LÝ CHUNK & ĐƯA VÀO VECTOR STORE
# ======================

ids, texts, metadatas = [], [], []

print("\n🔹 Đang chia chunk và chuẩn bị dữ liệu...")
for i, doc in enumerate(tqdm(documents, desc="Chunking")):
    chunks = chunk_text(doc["content"], max_tokens=800, overlap=100)
    for j, chunk in enumerate(chunks):
        unique_id = f"{doc['id']}_{j}"

        ids.append(unique_id)
        texts.append(chunk)
        metadatas.append({
            "id": doc.get("id"),
            "title": doc.get("title", ""),
            "impact": doc.get("impact", ""),
            "firm": doc.get("firm", ""),
            "protocol": doc.get("protocol", ""),
            "date": doc.get("date", ""),
            "source": doc.get("source", "")
        })

# ======================
# 💾 THÊM VÀO CHROMADB
# ======================

print(f"\n🧠 Bắt đầu thêm {len(texts)} chunks vào collection '{collection_name}'...")

batch_size = 100
added = 0

for i in tqdm(range(0, len(texts), batch_size), desc="Embedding & Adding"):
    batch_ids = ids[i:i + batch_size]
    batch_texts = texts[i:i + batch_size]
    batch_meta = metadatas[i:i + batch_size]

    try:
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_meta
        )
        added += len(batch_texts)
    except Exception as e:
        print(f"⚠️ Lỗi batch {i // batch_size}: {e}")

print(f"\n✅ Đã thêm {added:,} chunks vào collection '{collection_name}'.")
print(f"📂 Database lưu tại: {chroma_path}")

# ======================
# 📊 TỔNG KẾT
# ======================

summary = collection.count()
print(f"\n📈 Tổng số vector trong DB hiện tại: {summary}")
print("🏁 Hoàn tất xây dựng vector store.")
