from fastapi import APIRouter
from api.schemas import RegisterRequest

router = APIRouter(prefix="/register", tags=["Register"])

from api.container import database_manager


@router.post("/")
def search(request: RegisterRequest):

    try:
        database_manager.conn.execute('INSERT INTO users (user_id,current_chunk_count) VALUES (?,?)',(request.user_id,0))
        database_manager.conn.commit()
        print('Success')
        return {
            "results": "Success"
        }
    except Exception as e:
        print(e)
        return {
            "results": str(e)
        }

