import json
import asyncio
#import google.generativeai as genai 
from llm.base import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self, model="gemini-1.5-pro"):
        genai.configure()
        self.model = genai.GenerativeModel(model)

    async def rewrite(self, prompt: str) -> dict:
        def _call():
            resp = self.model.generate_content(prompt)
            return json.loads(resp.text)

        return await asyncio.to_thread(_call)
