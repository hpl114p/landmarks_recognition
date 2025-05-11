from fastapi import APIRouter
from .landmark_route import router as landmark_cls_route

router = APIRouter()
router.include_router(landmark_cls_route, prefix="")

