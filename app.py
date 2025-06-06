import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI
from middleware.http import LogMiddleware
from middleware.cors import setup_cors
from routes.base import router

app = FastAPI()

app.add_middleware(LogMiddleware)
setup_cors(app)
app.include_router(router)

