from groq import Groq
import os


class LLMClient:

    def __init__(self, model_name: str = "openai/gpt-oss-120b", logger=None):
        self.logger = logger
        self.model_name = model_name

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")

        self.client = Groq(api_key=api_key)

    async def generate(self, prompt: str) -> str:
        """
        Async wrapper around Groq LLM call.
        """

        try:
            import asyncio

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                )
            )

            return response.choices[0].message.content

        except Exception as e:
            # if self.logger:
            #     self.logger.exception(f"[GROQ LLM] ERROR: {e}")
            print(e)
            return "Error generating response."
