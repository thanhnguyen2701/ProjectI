import os
import chromadb
from openai import OpenAI
from typing import Optional, Dict, List
from dotenv import load_dotenv

load_dotenv()

# Cấu hình
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["CHROMA_OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
CHROMA_DB_PATH = r"C:\Users\FPTSHOP\2025.1\ProjectI\chroma_db"
COLLECTION_NAME = "smart_contract_audits"
OPENAI_CHAT_MODEL = "gpt-4o-mini"    # Hoặc thay bằng model bạn muốn/được phép dùng
TOP_K = 6                            # số chunk lấy về

# Khởi tạo

# Chroma client & collection
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# Hàm lấy kết quả retrieval từ Chroma
def retrieve(question: str, top_k: int = TOP_K, filters: Optional[Dict] = None):
    """
    Trả về list of hits: [{id, document, metadata, distance}]
    filters là dict metadata để lọc, ví dụ {"firm": "Pashov Audit Group", "impact": "LOW"}
    """
    where_clause = None
    if filters:
        # Nếu filters chứa nhiều điều kiện, chuyển về dạng cú pháp mới
        if len(filters) > 1:
            where_clause = {"$and": [{k: v} for k, v in filters.items()]}
        else:
            # Chỉ 1 điều kiện
            key, value = next(iter(filters.items()))
            where_clause = {key: value}

    if where_clause:
        query_result = collection.query(
            query_texts=[question],
            n_results=top_k,
            where=where_clause
        )
    else:
        query_result = collection.query(
            query_texts=[question],
            n_results=top_k
        )

    # parse result
    ids = query_result["ids"][0]
    docs = query_result["documents"][0]
    metas = query_result["metadatas"][0]
    dists = query_result.get("distances", [[]])[0]
    hits = []
    for i, doc in enumerate(docs):
        hits.append({
            "id": ids[i] if i < len(ids) else None,
            "document": doc,
            "metadata": metas[i] if i < len(metas) else {},
            "distance": dists[i] if i < len(dists) else None
        })
    return hits

# Hàm dựng prompt từ hits + câu hỏi
def build_prompt(question: str, hits: List[Dict], max_context_chars: int = 2500) -> str:
    """
    Gom các đoạn retrieved thành context (cắt nếu quá dài),
    rồi trả về prompt string cho LLM.
    """
    context_parts = []
    total_chars = 0
    for h in hits:
        meta = h.get("metadata", {})
        header = f"[id:{meta.get('id')} | impact:{meta.get('impact')} | firm:{meta.get('firm')} | protocol:{meta.get('protocol')} | date:{meta.get('date')}]"
        snippet = h.get("document", "").strip()
        part = header + "\n" + snippet + "\n"
        # only add while not exceeding budget
        if total_chars + len(part) > max_context_chars:
            break
        context_parts.append(part)
        total_chars += len(part)

    context = "\n---\n".join(context_parts)
    prompt = (
        "You are an expert security auditor assistant specialized in smart-contract findings.\n"
        "Use ONLY the information provided in the CONTEXT to answer the USER question. Do not hallucinate. "
        "If the context does not contain enough info, say you don't have enough information and optionally suggest next steps.\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "USER QUESTION:\n"
        f"{question}\n\n"
        "INSTRUCTIONS:\n"
        "1) Provide a concise answer (3-6 sentences) and then a short bulleted summary of the key supporting points.\n"
        "2) Cite sources by their [id:...] tags from the context.\n"
        "3) If the user asked for examples or code, only provide examples that appear in the context.\n"
    )
    return prompt

# Hàm gọi OpenAI ChatCompletion
def ask_llm(prompt: str, model: str = OPENAI_CHAT_MODEL, temperature: float = 0.0, max_tokens: int = 512):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful, concise assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()
# Hàm tiện ích tổng hợp trả lời từ retrieval+LLM
def answer_question(question: str, top_k: int = TOP_K, filters: Optional[Dict] = None):
    hits = retrieve(question, top_k=top_k, filters=filters)
    if not hits:
        return {"answer": "No documents found for that query (check filters).", "hits": []}
    prompt = build_prompt(question, hits)
    answer = ask_llm(prompt)
    # attach simple citations list for UI
    cited_ids = []
    for h in hits:
        meta = h.get("metadata", {})
        cited_ids.append({
            "id": meta.get("id"),
            "title": meta.get("title"),
            "source": meta.get("source")
        })
    return {"answer": answer, "hits": cited_ids}

# Ví dụ sử dụng
if __name__ == "__main__":
    # # Ví dụ 1: câu hỏi chung, không lọc
    # q1 = "What are common LOW impact vulnerabilities in the dataset?"
    # out1 = answer_question(q1, top_k=6, filters=None)
    # print("ANSWER 1:\n", out1["answer"])
    # print("\nSOURCES:\n")
    # for s in out1["hits"]:
    #     print(s)

    # Ví dụ 2: filter theo firm + impact
    # q2 = "Summarize findings related to vault expiration issues."
    # filters = {"firm": "Pashov Audit Group", "impact": "LOW"}
    # out2 = answer_question(q2, top_k=6, filters=filters)
    # print("\n\nANSWER 2:\n", out2["answer"])
    # print("\nSOURCES:\n")
    # for s in out2["hits"]:
    #     print(s)

    q3 = "Which protocols were audited by Spearbit and have LOW impact issues?"
    out3 = answer_question(q3, top_k=15, filters=None)
    print("ANSWER:\n", out3["answer"])
    print("\nSOURCES:\n")
    for s in out3["hits"]:
        print(s)