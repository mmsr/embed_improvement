from rewrite.prompt import build_prompt
from rewrite.validator import validate_numbers

MAX_RETRIES = 3

async def rewrite_triplet(triplet: dict, llm) -> dict:
    pos = triplet["positive"]
    neg = triplet["negative"]
    prompt = build_prompt(pos, neg)

    for _ in range(MAX_RETRIES):
        result = await llm.rewrite(prompt)

        return {
                **triplet,
                "positive_rewritten": result["positive"],
                "negative_rewritten": result["negative"]
            }
