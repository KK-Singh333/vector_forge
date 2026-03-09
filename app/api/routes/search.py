from fastapi import APIRouter
from api.schemas import SearchRequest

router = APIRouter(prefix="/search", tags=["Search"])

from api.container import cached_search_service


@router.post("/")
def search(request: SearchRequest):

    results = cached_search_service.search(
        query_text=request.query,
        user_id=request.user_id,
        k=request.k
    )

    return {
        "count": len(results),
        "results": results
    }
