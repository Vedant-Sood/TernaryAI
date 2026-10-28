"""Local HTTP API (FastAPI). Binds to 127.0.0.1 by default; there is no authentication.

    GET  /health               liveness
    GET  /models               registered models, installed / loaded state
    POST /load    {"model"}    load a model
    POST /unload  {"model"}    unload a model and free memory
    POST /route   {"prompt"}   router recommendation only
    POST /generate {"prompt", "model": "auto"|name, "max_new_tokens", "temperature", "top_k", "top_p"}
    GET  /system               RAM / GPU usage
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine import Engine, system_stats
from .router import route


class GenerateRequest(BaseModel):
    prompt: str
    model: str = "auto"
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=5.0)
    top_k: int | None = 50
    top_p: float | None = 0.95


class ModelRequest(BaseModel):
    model: str


class RouteRequest(BaseModel):
    prompt: str


def create_app(engine: Engine | None = None) -> FastAPI:
    eng = engine or Engine()
    app = FastAPI(title="TriForge local API", version="0.1.0")

    @app.get("/health")
    def health():
        return {"ok": True, "device": str(eng.device)}

    @app.get("/models")
    def models():
        return eng.available()

    @app.post("/load")
    def load(req: ModelRequest):
        try:
            return eng.load(req.model)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, str(e))

    @app.post("/unload")
    def unload(req: ModelRequest):
        eng.unload(req.model)
        return {"unloaded": req.model}

    @app.post("/route")
    def do_route(req: RouteRequest):
        r = route(req.prompt)
        return {"specialist": r.specialist, "scores": r.scores, "reason": r.reason}

    @app.post("/generate")
    def generate(req: GenerateRequest):
        try:
            text, stats = eng.generate(req.prompt, model=req.model, max_new_tokens=req.max_new_tokens,
                                       temperature=req.temperature, top_k=req.top_k, top_p=req.top_p)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, str(e))
        return {"text": text, "stats": stats}

    @app.get("/system")
    def system():
        return system_stats()

    return app
