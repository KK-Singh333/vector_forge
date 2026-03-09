import httpx
class UserService:

    def __init__(self,base_url):
        self._users = {}
        self._next_id = 1
        self.base_url=base_url

    async def register(self, username: str):

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/register/",
                json={
                    "user_id": self._next_id,
                }
            )
        
        print(response)
        response = response.json()
        if response.get('results') in 'Success':
            user_id = self._next_id
            self._next_id += 1

            self._users[user_id] = {
                "user_id": user_id,
                "username": username
            }
            return self._users[user_id]
        else:
            return None


    def exists(self, user_id: int):
        return user_id in self._users
