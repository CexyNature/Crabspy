"""JSON health endpoint for monitoring and Docker healthchecks."""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
