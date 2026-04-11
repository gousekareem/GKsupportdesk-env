"""
SupportDeskEnv — FastAPI server.
Endpoints: /reset /step /state /health /tasks /metrics /replay /openenv.yaml
"""
from __future__ import annotations
import os, sys, threading, time
sys.path.insert(0, os.path.dirname(__file__))

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from env import SupportDeskEnv, TASK_CONFIGS
from models import TicketAction, StepResult, ResetResult, StateResult

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SupportDeskEnv",
    description=(
        "OpenEnv-compliant customer support ticket management environment. "
        "7 tasks from easy to hardest: classify, prioritize, resolve, "
        "sentiment-detect, thread-summarise, compliance-check, deescalate."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# One SupportDeskEnv per task (manages sessions internally)
_envs: dict[str, SupportDeskEnv] = {}


def _env(task: str) -> SupportDeskEnv:
    if task not in TASK_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task}'. Valid tasks: {list(TASK_CONFIGS.keys())}",
        )
    if task not in _envs:
        _envs[task] = SupportDeskEnv(task)
    return _envs[task]


# ── Keep-alive thread (prevents HF Space from sleeping) ───────────────────────

def _keepalive():
    """Ping /health every 25 minutes to prevent HF Space sleep."""
    import urllib.request
    port = int(os.getenv("PORT", "7860"))
    while True:
        time.sleep(25 * 60)
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5)
        except Exception:
            pass

_ka_thread = threading.Thread(target=_keepalive, daemon=True)
_ka_thread.start()


# ── Startup log ───────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    print(
        f"\n{'='*55}\n"
        f"  SupportDeskEnv v3.0.0 — OpenEnv compliant\n"
        f"  7 tasks | 20 tickets | Port {os.getenv('PORT', '7860')}\n"
        f"{'='*55}\n",
        flush=True,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "SupportDeskEnv",
        "version": "3.0.0",
        "framework": "OpenEnv",
        "tasks": list(TASK_CONFIGS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/health", "/tasks",
                      "/metrics", "/replay", "/openenv.yaml"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "supportdesk-env",
        "version": "3.0.0",
        "tasks_loaded": len(TASK_CONFIGS),
    }


@app.get("/tasks")
def list_tasks():
    return {
        name: {
            "difficulty": cfg["difficulty"],
            "max_steps": cfg["max_steps"],
            "description": cfg["description"][:150] + "...",
            "ticket_count": len(cfg["ticket_pool"]),
        }
        for name, cfg in TASK_CONFIGS.items()
    }


@app.get("/openenv.yaml", response_class=PlainTextResponse)
def serve_yaml():
    """Serve openenv.yaml as HTTP endpoint for openenv validate."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "openenv.yaml")
    if not os.path.exists(yaml_path):
        raise HTTPException(status_code=404, detail="openenv.yaml not found")
    with open(yaml_path) as f:
        return f.read()


@app.post("/reset", response_model=ResetResult)
def reset(task: str = Query(default="ticket-classify")):
    """Reset environment and start a new episode. Returns session_id."""
    result, _ = _env(task).reset()
    return result


@app.post("/step", response_model=StepResult)
def step(
    action: TicketAction,
    task: str = Query(default="ticket-classify"),
    session_id: str = Query(default=""),
):
    """Execute one action. Requires session_id from /reset."""
    if not session_id:
        raise HTTPException(400, "session_id is required. Call /reset first.")
    try:
        return _env(task).step(action, session_id)
    except (RuntimeError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.get("/state", response_model=StateResult)
def state(
    task: str = Query(default="ticket-classify"),
    session_id: str = Query(default=""),
):
    """Return current episode state."""
    if not session_id:
        raise HTTPException(400, "session_id is required.")
    try:
        return _env(task).state(session_id)
    except KeyError as e:
        raise HTTPException(400, str(e))


@app.get("/metrics")
def metrics(task: str = Query(default="ticket-classify")):
    """Live metrics: episodes run, avg score, best score."""
    return _env(task).metrics()


@app.get("/replay")
def replay(
    task: str = Query(default="ticket-classify"),
    session_id: str = Query(default=""),
):
    """Full episode replay: every observation, action, reward, grader breakdown."""
    if not session_id:
        raise HTTPException(400, "session_id is required.")
    try:
        st = _env(task).state(session_id)
        return {
            "session_id": session_id,
            "task": task,
            "total_reward": st.total_reward,
            "avg_reward": st.avg_reward,
            "steps": len(st.grader_history),
            "done": st.done,
            "emotional_trajectory": st.emotional_trajectory,
            "history": st.grader_history,
        }
    except KeyError as e:
        raise HTTPException(400, str(e))


def main():
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        workers=1,
        timeout_keep_alive=60,
    )


if __name__ == "__main__":
    main()