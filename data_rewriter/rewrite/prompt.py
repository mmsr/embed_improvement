def build_prompt(positive: str, negative: str) -> str:
    return f"""
You are rewriting sentences for a contrastive learning dataset.

Rules:
- Rewrite each sentence using different wording
- Keep the meaning the same
- Keep ALL numbers exactly the same
- Do NOT add, remove, or modify any numbers
- Do NOT explain anything
- Do NOT write same words and style in positive and negative sentences. They are different.

Importanant:
- if possible replace company name with other company names
- else replace company name with similar type of companies or simple 'The <type of company> Company' or 'The <type of firm> Firm'.
  a. Microsoft revenue grow by 10%. -> The tech company reported increase in revenue by 10 %.
  b. JP Morgan share price dropped by 5%. -> The Banking company price declined by 5%.
- Only If you don't know type of comapny then just mention The company

Return output as valid JSON only:
{{
  "positive": "...",
  "negative": "..."
}}

Input:
Positive: "{positive}"
Negative: "{negative}"
"""
