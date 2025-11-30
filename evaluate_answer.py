import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EVAL_MODEL = "gpt-4o-mini"


def safe_parse_json(raw):
    """Parse JSON an toàn."""
    try:
        return json.loads(raw)
    except Exception:
        return {
            "correct": False,
            "reason": "Evaluator returned invalid JSON format."
        }


def normalize_text_block(text):
    """Chuyển mọi loại dữ liệu về text an toàn để đưa vào prompt."""
    if text is None:
        return ""
    if isinstance(text, list):
        return "\n".join(str(x) for x in text)
    return str(text)


def evaluate_llm_answer(question, model_answer, ground_truth, context_used):
    """Đánh giá model_answer dựa trên ground_truth + context."""

    # Normalize inputs
    question = normalize_text_block(question)
    model_answer = normalize_text_block(model_answer)
    gt = normalize_text_block(ground_truth)
    context_used = normalize_text_block(context_used)[:8000]  # tránh overflow

    prompt = f"""
You are a senior smart contract auditor.

Your task is to evaluate whether the model's answer is CORRECT or INCORRECT
compared to the ground truth.

You MUST consider:
- QUESTION
- RETRIEVED CONTEXT
- MODEL ANSWER
- GROUND TRUTH

Follow the criteria:
- Does the model identify correct vulnerability type(s)?
- Is reasoning technically aligned with the ground truth?
- Are mitigation steps accurate?
- Are there factual or logical errors?

Respond in JSON ONLY.

---------------------------
QUESTION:
{question}

RETRIEVED CONTEXT:
{context_used}

MODEL ANSWER (to evaluate):
{model_answer}

GROUND TRUTH ANSWER:
{gt}

---------------------------

Output JSON ONLY:
{{
  "correct": true | false,
  "reason": "2-4 sentences explaining the decision"
}}
"""

    response = client.chat.completions.create(
        model=EVAL_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an unbiased correctness checker. "
                    "You output ONLY JSON — no explanations, no comments."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


# ======================
# Run on dataset
# ======================

INPUT = r"C:\Users\FPTSHOP\2025.1\ProjectI\evaluation_results.jsonl"
OUTPUT = r"C:\Users\FPTSHOP\2025.1\ProjectI\evaluation_llm_boolean.jsonl"

results = []

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        # Use "question" if exists, fallback to "title"
        question_text = item.get("question", item.get("title", ""))

        raw_eval = evaluate_llm_answer(
            question=question_text,
            model_answer=item.get("model_answer", ""),
            ground_truth=item.get("ground_truth", ""),
            context_used=item.get("context_used", "")
        )

        item["llm_evaluation"] = safe_parse_json(raw_eval)
        results.append(item)


with open(OUTPUT, "w", encoding="utf-8") as f:
    for r in results:
        json.dump(r, f, ensure_ascii=False)
        f.write("\n")

print("🔥 Boolean LLM Evaluation DONE")
print("📁 Saved:", OUTPUT)