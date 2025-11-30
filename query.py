import os
import json
from tqdm import tqdm
import chromadb
from openai import OpenAI
from typing import Optional, Dict, List
import numpy as np
from dotenv import load_dotenv

# =============== LOAD CONFIG ===============
load_dotenv()
os.environ["CHROMA_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

CHROMA_DB_PATH = r"C:\Users\FPTSHOP\2025.1\ProjectI\chroma_db"
TEST_FILE = r"C:\Users\FPTSHOP\2025.1\ProjectI\test_cases.jsonl"
OUTPUT_FILE = r"C:\Users\FPTSHOP\2025.1\ProjectI\evaluation_results.jsonl"

COLLECTION_NAME = "smart_contract_audits"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
TOP_K = 20
MAX_CONTEXT_CHARS = 4000

# =============== INIT CLIENTS ===============
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============== RETRIEVAL ===============
def retrieve(question: str, top_k: int = TOP_K, filters: Optional[Dict] = None):
    queries = [
        question,
        f"Analyze Solidity vulnerability: {question}",
        f"Possible exploits in code: {question}",
        f"Fix for this Solidity bug: {question}"
    ]
    hits = []

    for q in queries:
        result = collection.query(query_texts=[q], n_results=top_k, where=filters)
        for i, doc in enumerate(result["documents"][0]):
            meta = result["metadatas"][0][i]
            hits.append({
                "id": result["ids"][0][i],
                "document": doc,
                "metadata": meta,
                "distance": result["distances"][0][i]
            })

    # Gộp và sắp theo độ tương đồng
    seen = {}
    for h in hits:
        if h["id"] not in seen or h["distance"] < seen[h["id"]]["distance"]:
            seen[h["id"]] = h
    hits = sorted(seen.values(), key=lambda x: x["distance"])[:top_k]

    return hits

# =============== PROMPT BUILDER ===============
def build_prompt_with_context(question: str, hits: List[Dict], max_context_chars: int = MAX_CONTEXT_CHARS) -> (str, str):
    """
    Tạo prompt cho LLM, chỉ ẩn answer nhưng đưa đầy đủ context từ hits.
    Trả về (prompt, context_used)
    """
    context_parts = []
    total_chars = 0
    for h in hits:
        meta = h.get("metadata", {})
        header = (
            f"[id:{meta.get('id')} | impact:{meta.get('impact')} | "
            f"firm:{meta.get('firm')} | protocol:{meta.get('protocol')} | date:{meta.get('date')}]"
        )
        snippet = h.get("document", "").strip()
        part = header + "\n" + snippet + "\n"
        if total_chars + len(part) > max_context_chars:
            break
        context_parts.append(part)
        total_chars += len(part)

    context_used = "\n---\n".join(context_parts)

    prompt = f"""
You are a professional blockchain security auditor.

Analyze the following Solidity code or security finding.

CONTEXT (previous smart contract audit findings):
{context_used}

QUESTION (without answer):
{question}

Please provide a detailed vulnerability analysis, explanation, and potential fix based only on the context and your expertise.

RESPONSE FORMAT:
1. Vulnerability Type(s): <short summary>
2. Explanation: <3–6 sentences explaining the issue>
3. Fix Suggestion (if any): <code or advice>
4. References: [id:...] if relevant
"""
    return prompt.strip(), context_used

# =============== LLM CALL ===============
def ask_llm(prompt: str, model: str = OPENAI_CHAT_MODEL, temperature: float = 0.0, max_tokens: int = 600):
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert Solidity auditor."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

# =============== SIMILARITY SCORE ===============
def embed_text(text: str):
    """Tạo embedding vector bằng model text-embedding-3-small."""
    try:
        resp = embedding_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(resp.data[0].embedding)
    except Exception as e:
        print(f"[!] Embedding error: {e}")
        return np.zeros(1536)

def similarity_score(text1: str, text2: str) -> float:
    """Tính cosine similarity giữa hai embedding."""
    e1, e2 = embed_text(text1), embed_text(text2)
    if np.linalg.norm(e1) == 0 or np.linalg.norm(e2) == 0:
        return 0.0
    sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    return round(float(sim), 3)

# =============== MAIN LOOP ===============
if __name__ == "__main__":
    print("🚀 Đang chạy đánh giá mô hình RAG trên test_cases.jsonl...\n")

    # --- Load test cases ---
    test_cases = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                test_cases.append(json.loads(line))
            except:
                continue

    print(f"📦 Tổng số test: {len(test_cases)}\n")

    results = []

    # --- Chạy đánh giá ---
    for i, case in enumerate(tqdm(test_cases, desc="🔍 Evaluating test cases")):
        question = case.get("question", "")

        # ---- Load ground truth (answer or answers[]) ----
        raw_answer = case.get("answer", "") or case.get("answers", [])

        if isinstance(raw_answer, list):
            # Gộp tất cả câu trả lời lại
            ground_truth = "\n".join(raw_answer)
        elif isinstance(raw_answer, str):
            ground_truth = raw_answer
        else:
            ground_truth = ""

        if not question.strip():
            continue

        hits = retrieve(question)
        prompt, context_used = build_prompt_with_context(question, hits)
        model_answer = ask_llm(prompt)
        score = similarity_score(model_answer, ground_truth)

        results.append({
            "id": case.get("id"),
            "title": case.get("title"),
            "impact": case.get("impact"),
            "firm": case.get("firm"),
            "protocol": case.get("protocol"),
            "score": score,
            "model_answer": model_answer,
            "ground_truth": ground_truth,
            "context_used": context_used,   # <--- Lưu context
            "source": case.get("source"),
        })

        # In nhanh vài kết quả đầu
        if i < 3:
            print(f"\n==== Test #{i+1} ====")
            print(f"Title: {case.get('title')}")
            print(f"Score: {score}")
            print(f"Model Answer:\n{model_answer[:500]}...\n")
            print(f"Ground Truth:\n{ground_truth[:500]}...\n")
            print(f"Context Used:\n{context_used[:500]}...\n")

    # --- Lưu kết quả ---
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for r in results:
            json.dump(r, out, ensure_ascii=False)
            out.write("\n")

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n✅ Đánh giá hoàn tất. Trung bình similarity: {avg_score:.3f}")
    print(f"📁 Kết quả chi tiết được lưu tại: {OUTPUT_FILE}")
