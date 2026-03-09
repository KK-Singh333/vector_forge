import httpx
import time

class KnowledgeClient:

    def __init__(self, base_url: str):
        self.base_url = base_url

    async def search(self, user_id: int, query: str, k: int = 5):
        time1=time.time()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/search/",
                json={
                    "user_id": user_id,
                    "query": query,
                    "k": k
                },
                timeout=5
            )
        response_time=time.time()-time1
        response.raise_for_status()
        return response.json()["results"],response_time


    async def upload_pdf(self, user_id: int, pdf_id: str, file):

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/ingest/",
                files={
                    "file": (file.filename, await file.read(), "application/pdf")
                },
                data={
                    "user_id": str(user_id),
                    "pdf_id": pdf_id
                },
                timeout=20
            )

        response.raise_for_status()
        return response.json()
