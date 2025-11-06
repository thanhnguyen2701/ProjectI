import os
import chromadb
from openai import OpenAI
from typing import Optional, Dict, List
from dotenv import load_dotenv

# =============== LOAD CONFIG ===============
load_dotenv()
os.environ["CHROMA_OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

CHROMA_DB_PATH = r"C:\Users\FPTSHOP\2025.1\ProjectI\chroma_db"
COLLECTION_NAME = "smart_contract_audits"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
TOP_K = 40
MAX_CONTEXT_CHARS = 8000

# =============== INIT CLIENTS ===============
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============== RETRIEVAL ===============
def retrieve(question: str, top_k: int = TOP_K, filters: Optional[Dict] = None):
    where_clause = None
    if filters:
        if len(filters) > 1:
            where_clause = {"$and": [{k: v} for k, v in filters.items()]}
        else:
            key, value = next(iter(filters.items()))
            where_clause = {key: value}

    result = collection.query(
        query_texts=[question],
        n_results=top_k,
        where=where_clause
    )
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result.get("distances", [[]])[0]

    hits = []
    for i, doc in enumerate(docs):
        hits.append({
            "id": ids[i],
            "document": doc,
            "metadata": metas[i],
            "distance": dists[i]
        })
    return hits

# =============== PROMPT BUILDER ===============
def build_prompt(question: str, hits: List[Dict], max_context_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Gom các đoạn retrieved thành context (cắt nếu quá dài),
    rồi trả về prompt string cho LLM.
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

    context = "\n---\n".join(context_parts)

    prompt = f"""
You are a professional blockchain security auditor assistant.
You will analyze the provided Solidity code snippet for potential vulnerabilities.

Use the CONTEXT (findings, examples, and historical vulnerabilities) to identify possible issues.
Base your reasoning ONLY on the provided context and your own Solidity audit reasoning.

CONTEXT:
{context}

CODE SNIPPET TO ANALYZE:
{question}

TASK:
1. Identify whether the code contains a reentrancy vulnerability (or any other serious issue).
2. Explain the reasoning clearly and concisely (5–8 sentences).
3. Provide suggestions on how to fix or mitigate the issue.
4. Reference relevant findings from the context using [id:...] tags.
"""
    return prompt.strip()

# =============== OPENAI CALL ===============
def ask_llm(prompt: str, model: str = OPENAI_CHAT_MODEL, temperature: float = 0.0, max_tokens: int = 700):
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a smart-contract security expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

# =============== MAIN EXECUTION ===============
if __name__ == "__main__":
    # 🧩 👉 Đây là nơi bạn thay đổi code hoặc câu hỏi mỗi lần chạy
    solidity_code = """
Does this Solidity code contain a reentrancy vulnerability?

contract Vault {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint amount) public {
        require(balances[msg.sender] >= amount, "Not enough funds");
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        balances[msg.sender] -= amount;
    }
}
    """

    # 🔍 Retrieval + Prompt build
    hits = retrieve(solidity_code, top_k=TOP_K)
    prompt = build_prompt(solidity_code, hits)
    answer = ask_llm(prompt)

    print("\n===================== 🔎 ANALYSIS RESULT =====================\n")
    print(answer)
    print("\n==============================================================\n")

    print("📚 SOURCES (Top context chunks):")
    for s in hits[:5]:
        meta = s.get("metadata", {})
        print(f" - [id:{meta.get('id')}] {meta.get('title', '')} ({meta.get('source', '')})")
