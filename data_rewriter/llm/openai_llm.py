import json
import re
from openai import AsyncOpenAI
## gpt-4.1-nano
## gpt-4.1-mini
class OpenAILLM:
    def __init__(self, model="gpt-4.1-nano"):
        self.client = AsyncOpenAI()
        self.model = model

    async def rewrite(self, prompt: str) -> dict:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )

        content = resp.choices[0].message.content.strip()

        # 🔥 Strip markdown fences
        content = re.sub(r"^```(?:json)?", "", content)
        content = re.sub(r"```$", "", content)
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parse failed. Raw content:\n{content}") from e
