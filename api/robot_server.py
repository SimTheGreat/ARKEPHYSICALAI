import os
import subprocess
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RobotExecuteRequest(BaseModel):
    script: str = Field(..., description="Python script to execute, e.g. move2.py")
    args: List[str] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = None


class RobotJobStatus(BaseModel):
    job_id: str
    status: str
    script: str
    args: List[str]
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    return_code: Optional[int] = None
    output: Optional[str] = None
    error: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


app = FastAPI(title="Robot Comms Server", version="1.0.0")
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.RLock()


def _split_allowlist(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _require_auth(x_robot_token: Optional[str]) -> None:
    expected = os.getenv("ROBOT_SERVER_TOKEN", "").strip()
    if expected and x_robot_token != expected:
        raise HTTPException(status_code=403, detail="Invalid robot token.")


def _allowed_scripts() -> List[str]:
    raw = os.getenv("ROBOT_ALLOWED_SCRIPTS", "move2.py")
    scripts = _split_allowlist(raw)
    return scripts or ["move2.py"]


def _robot_workdir() -> str:
    configured = os.getenv("ROBOT_WORKDIR", "").strip()
    if configured:
        return os.path.abspath(configured)
    return os.path.abspath(os.path.dirname(__file__))


def _run_job(job_id: str, command: List[str], cwd: str) -> None:
    with jobs_lock:
        jobs[job_id]["status"] = "RUNNING"
        jobs[job_id]["started_at"] = utc_now_iso()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        with jobs_lock:
            jobs[job_id]["finished_at"] = utc_now_iso()
            jobs[job_id]["return_code"] = completed.returncode
            jobs[job_id]["output"] = (completed.stdout or "")[-4000:]
            jobs[job_id]["error"] = (completed.stderr or "")[-4000:]
            jobs[job_id]["status"] = "SUCCEEDED" if completed.returncode == 0 else "FAILED"
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]["finished_at"] = utc_now_iso()
            jobs[job_id]["status"] = "FAILED"
            jobs[job_id]["error"] = str(exc)


@app.get("/health")
def health():
    return {"status": "healthy", "time": utc_now_iso()}


@app.post("/robot/execute")
def robot_execute(payload: RobotExecuteRequest, x_robot_token: Optional[str] = Header(default=None)):
    _require_auth(x_robot_token)

    script_name = os.path.basename(payload.script.strip())
    if script_name not in _allowed_scripts():
        raise HTTPException(status_code=400, detail=f"Script '{script_name}' is not in ROBOT_ALLOWED_SCRIPTS.")

    workdir = _robot_workdir()
    script_path = os.path.abspath(os.path.join(workdir, script_name))
    if not script_path.startswith(workdir):
        raise HTTPException(status_code=400, detail="Invalid script path.")
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail=f"Script not found: {script_path}")

    job_id = f"job_{uuid4().hex[:12]}"
    command = ["python3", script_path] + [str(a) for a in payload.args]
    with jobs_lock:
        jobs[job_id] = RobotJobStatus(
            job_id=job_id,
            status="QUEUED",
            script=script_name,
            args=[str(a) for a in payload.args],
            created_at=utc_now_iso(),
            context=payload.context,
        ).model_dump()

    thread = threading.Thread(target=_run_job, args=(job_id, command, workdir), daemon=True)
    thread.start()

    return {"ok": True, "job_id": job_id, "status": "QUEUED", "command": command}


@app.get("/robot/jobs/{job_id}")
def robot_job_status(job_id: str, x_robot_token: Optional[str] = Header(default=None)):
    _require_auth(x_robot_token)
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return {"ok": True, "job": job}

