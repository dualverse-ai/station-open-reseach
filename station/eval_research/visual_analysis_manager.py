"""
Lightweight visual-analysis jobs for the Research Center.

These jobs are separate from scored research evaluations. They let agents ask a
coder to generate or inspect plots and return an explanatory report.
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from station import constants
from station import file_io_utils
from station.cli_worker_backends import (
    apply_codex_proxy_overrides,
    build_cli_worker_runtime_env,
    detect_cli_worker_executable,
    get_cli_worker_backend,
)
from .runtime_paths import ResearchRuntimePaths, detect_station_python_executable, ensure_runtime_layout


VISUAL_TERMINAL_STATUSES = {"completed", "failed", "blocked"}


@dataclass
class ActiveVisualSession:
    job_id: str
    session_id: str
    process: subprocess.Popen
    transcript_handle: Any
    stderr_handle: Any
    report_path: str
    transcript_path: str
    stderr_path: str
    last_message_path: Optional[str]


class VisualAnalysisManager:
    def __init__(
        self,
        paths: Optional[ResearchRuntimePaths] = None,
        log_queue=None,
        notification_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.paths = paths or ensure_runtime_layout()
        self.log_queue = log_queue
        self.notification_callback = notification_callback
        self.active_sessions: Dict[str, ActiveVisualSession] = {}
        file_io_utils.ensure_dir_exists(self.paths.visual_jobs_dir)
        file_io_utils.ensure_dir_exists(self.paths.visual_storage_dir)

    def _push_log_event(self, event_type: str, data: Dict[str, Any]):
        if self.log_queue is None:
            return
        try:
            self.log_queue.put_nowait({"event": event_type, "data": data, "timestamp": time.time()})
        except Exception as exc:
            print(f"VisualAnalysisManager: Failed to queue log event: {exc}")

    def _job_path(self, job_id: str) -> str:
        return os.path.join(self.paths.visual_jobs_dir, f"{job_id}.yaml")

    def _job_storage_dir(self, job_id: str) -> str:
        return os.path.join(self.paths.visual_storage_dir, str(job_id))

    def _next_job_id(self) -> str:
        max_id = 0
        for filename in file_io_utils.list_files(self.paths.visual_jobs_dir, ".yaml"):
            stem = filename[:-5]
            try:
                max_id = max(max_id, int(stem))
            except ValueError:
                continue
        return str(max_id + 1)

    def create_job(
        self,
        *,
        author: str,
        lineage: str,
        title: str,
        context: str,
        script: str,
        images: List[str],
        question: str,
        tick: int,
    ) -> Dict[str, Any]:
        job_id = self._next_job_id()
        output_dir = self._job_storage_dir(job_id)
        file_io_utils.ensure_dir_exists(output_dir)
        job = {
            "id": job_id,
            "status": "queued",
            "author": author,
            "lineage": (lineage or "unknown").lower(),
            "title": title,
            "context": context,
            "script": script,
            "images": images,
            "question": question,
            "submitted_tick": tick,
            "created_timestamp": time.time(),
            "output_dir": f"storage/{constants.RESEARCH_STORAGE_VISUAL_DIR}/{job_id}",
            "report_path": f"storage/{constants.RESEARCH_STORAGE_VISUAL_DIR}/{job_id}/report.md",
        }
        file_io_utils.save_yaml(job, self._job_path(job_id))
        self._push_log_event("research_visual_job_queued", {"job_id": job_id, "author": author})
        return job

    def wake(self, reason: str = ""):
        self._push_log_event("research_visual_wake", {"reason": reason})

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        data = file_io_utils.load_yaml(self._job_path(str(job_id)))
        return data if isinstance(data, dict) else None

    def list_jobs(self) -> List[Dict[str, Any]]:
        jobs: List[Dict[str, Any]] = []
        for filename in file_io_utils.list_files(self.paths.visual_jobs_dir, ".yaml"):
            data = file_io_utils.load_yaml(os.path.join(self.paths.visual_jobs_dir, filename))
            if isinstance(data, dict):
                jobs.append(data)
        return sorted(jobs, key=lambda item: int(item.get("id", 0) or 0))

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        job = self.get_job(job_id)
        if not job:
            return None
        job.update(updates)
        job["updated_timestamp"] = time.time()
        file_io_utils.save_yaml(job, self._job_path(str(job_id)))
        return job

    def mark_completed(self, job_id: str, *, report_path: str) -> Optional[Dict[str, Any]]:
        display_report_path = self._display_storage_path(report_path)
        return self.update_job(
            job_id,
            {
                "status": "completed",
                "completed_timestamp": time.time(),
                "report_path": display_report_path,
            },
        )

    def get_report_text(self, job_id: str) -> Optional[str]:
        job = self.get_job(job_id)
        if not job:
            return None
        report_path = str(job.get("report_path") or "")
        resolved = self._resolve_storage_path(report_path)
        if not resolved or not file_io_utils.file_exists(resolved):
            return None
        return file_io_utils.load_text(resolved) or ""

    def has_active_sessions(self) -> bool:
        return bool(self.active_sessions)

    def has_pending_or_running(self) -> bool:
        if self.active_sessions:
            return True
        return any(str(job.get("status", "")).lower() in {"queued", "running"} for job in self.list_jobs())

    def poll(self):
        for job_id, session in list(self.active_sessions.items()):
            returncode = session.process.poll()
            if returncode is None:
                if self._session_timed_out(job_id):
                    self._terminate_session_process(session)
                    self.update_job(job_id, {"status": "failed", "failure_reason": "Visual analysis timed out."})
                    self._close_session_handles(session)
                    self.active_sessions.pop(job_id, None)
                continue

            self._close_session_handles(session)
            if file_io_utils.file_exists(session.report_path):
                self.mark_completed(job_id, report_path=session.report_path)
                job = self.get_job(job_id) or {}
                self._notify_completion(job_id, job)
            else:
                self.update_job(
                    job_id,
                    {
                        "status": "failed",
                        "failure_reason": f"Visual coder exited with code {returncode} without writing report.md.",
                    },
                )
            self.active_sessions.pop(job_id, None)

    def launch_next_queued(self) -> bool:
        if len(self.active_sessions) >= constants.RESEARCH_VISUAL_MAX_PARALLEL_WORKERS:
            return False
        for job in self.list_jobs():
            job_id = str(job.get("id"))
            if str(job.get("status", "")).lower() != "queued":
                continue
            if job_id in self.active_sessions:
                continue
            try:
                self.launch_job(job)
            except Exception as exc:
                self.update_job(
                    job_id,
                    {
                        "status": "blocked",
                        "failure_reason": f"Failed to launch visual analysis coder: {exc}",
                    },
                )
                self._push_log_event("research_visual_launch_failed", {"job_id": job_id, "error": str(exc)})
                return False
            return True
        return False

    def launch_job(self, job: Dict[str, Any]) -> bool:
        job_id = str(job.get("id"))
        backend = str(job.get("backend") or constants.RESEARCH_CODER_BACKEND).lower()
        model_name = job.get("model_name", constants.RESEARCH_CODER_MODEL_NAME)
        env = self._build_runtime_env()
        if backend == "codex":
            apply_codex_proxy_overrides(env)
        executable = self._detect_backend_executable(backend, env)
        backend_runner = get_cli_worker_backend(backend)
        run_dir = os.path.join(self.paths.coder_sessions_dir, f"visual_{backend}_{job_id}_{uuid.uuid4().hex[:8]}")
        file_io_utils.ensure_dir_exists(run_dir)
        output_dir = self._job_storage_dir(job_id)
        file_io_utils.ensure_dir_exists(output_dir)
        report_path = os.path.join(output_dir, "report.md")
        prompt = self._build_prompt(job, output_dir=output_dir)
        file_io_utils.save_text(prompt, os.path.join(run_dir, "prompt.txt"))
        prepared = backend_runner.prepare_launch(
            executable=executable,
            workspace_root=os.path.abspath(self.paths.research_root),
            research_root=os.path.abspath(self.paths.research_root),
            run_dir=run_dir,
            model_name=model_name,
            storage_root=self.paths.storage_root,
            prompt=prompt,
        )

        for key, value in (prepared.env_overrides or {}).items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value

        transcript_handle = open(prepared.transcript_path, "w", encoding="utf-8")
        stderr_handle = open(prepared.stderr_path, "w", encoding="utf-8")
        popen_kwargs = {
            "cwd": os.path.abspath(self.paths.research_root),
            "env": env,
            "stdout": transcript_handle,
            "stderr": stderr_handle,
            "text": True,
            "start_new_session": True,
        }
        if prepared.stdin_text is not None:
            popen_kwargs["stdin"] = subprocess.PIPE
        process = subprocess.Popen(prepared.command, **popen_kwargs)
        if prepared.stdin_text is not None:
            assert process.stdin is not None
            process.stdin.write(prepared.stdin_text)
            process.stdin.close()

        session_id = os.path.basename(run_dir)
        self.active_sessions[job_id] = ActiveVisualSession(
            job_id=job_id,
            session_id=session_id,
            process=process,
            transcript_handle=transcript_handle,
            stderr_handle=stderr_handle,
            report_path=report_path,
            transcript_path=prepared.transcript_path,
            stderr_path=prepared.stderr_path,
            last_message_path=prepared.last_message_path,
        )
        self.update_job(
            job_id,
            {
                "status": "running",
                "session_id": session_id,
                "active_pid": process.pid,
                "started_timestamp": time.time(),
                "report_path": self._display_storage_path(report_path),
            },
        )
        self._push_log_event("research_visual_job_started", {"job_id": job_id, "session_id": session_id})
        return True

    def _build_prompt(self, job: Dict[str, Any], *, output_dir: str) -> str:
        job_id = str(job.get("id"))
        lineage = str(job.get("lineage", "unknown")).lower()
        python_bin = detect_station_python_executable(self._build_runtime_env())
        images = job.get("images") or []
        image_lines = "\n".join(f"- {path}" for path in images) if images else "(none)"
        script = str(job.get("script") or "").rstrip() or "(none)"
        output_display = self._display_storage_path(output_dir)
        return f"""You are the Research Center Visual Analyst for visual job {job_id}.

This is a non-interactive visual analysis session. You cannot ask follow-up questions.

Your job:
- Help the agent understand plots or visual artifacts relevant to the active research task.
- If Python plotting code is provided, run or adapt it to generate clear image files.
- If image paths are provided, inspect those images directly.
- Explain only what is visible or strongly supported by the provided artifacts.
- Do not invent numerical claims that are not shown in the plot or supporting files.
- If the image is unreadable, explain how it should be redrawn.

Context
- Title: {job.get("title", "Untitled")}
- Author: {job.get("author", "Unknown")}
- Lineage: {lineage}
- Working directory: {os.path.abspath(self.paths.research_root)}
- Python executable: {python_bin}
- Output directory: `{output_display}`
- Final report path: `{output_display}/report.md`

Filesystem access
- Read/write: `storage/{lineage}`, `storage/shared`, `{output_display}`
- Read-only: `storage/system`, `evaluations/`, `evaluators/`, task spec files
- Never modify `storage/system`, `evaluations`, `evaluators`, or task spec files.

Agent context
{job.get("context", "")}

Agent question
{job.get("question", "")}

Existing image paths
{image_lines}

Python plotting script, if any
```python
{script}
```

Workflow
1. Inspect any provided image paths.
2. If a script is provided, write it to `{output_display}/visual_script.py` or create a safer equivalent, then run it with `{python_bin}`.
3. Save generated plots under `{output_display}/`.
4. Open and inspect the generated image files.
5. Write `{output_display}/report.md`.

Required report format
# Visual Analysis Report

## Summary
Brief answer to the agent's question.

## Visible Evidence
Describe the plot/image features you directly observed.

## Interpretation
Explain what the visual evidence suggests for the research task.

## Files Created Or Inspected
List relevant images/scripts.

## Suggested Next Step
One focused next experiment or analysis.

## Caveats
Mention unreadable labels, missing data, plotting limitations, or uncertainty.
"""

    def _detect_backend_executable(self, backend: str, env: Optional[Dict[str, str]] = None) -> str:
        return detect_cli_worker_executable(backend, env or self._build_runtime_env())

    def _build_runtime_env(self) -> Dict[str, str]:
        return build_cli_worker_runtime_env(constants.RESEARCH_EVAL_PYTHON_CONDA_ENV)

    def _session_timed_out(self, job_id: str) -> bool:
        job = self.get_job(job_id) or {}
        started = float(job.get("started_timestamp") or time.time())
        return (time.time() - started) > constants.RESEARCH_VISUAL_TIMEOUT_SECONDS

    def _terminate_session_process(self, session: ActiveVisualSession):
        try:
            session.process.terminate()
        except Exception:
            pass

    @staticmethod
    def _close_session_handles(session: ActiveVisualSession):
        for handle in (session.transcript_handle, session.stderr_handle):
            try:
                handle.close()
            except Exception:
                pass

    def _notify_completion(self, job_id: str, job: Dict[str, Any]):
        if not self.notification_callback:
            return
        report_text = self.get_report_text(job_id) or ""
        message = f"Your visual analysis job {job_id} has completed.\n\n{report_text}"
        self.notification_callback(str(job.get("author", "")), message)

    def _resolve_storage_path(self, display_path: str) -> Optional[str]:
        path = str(display_path or "").strip()
        if not path:
            return None
        if os.path.isabs(path):
            return path
        if path.startswith("storage/"):
            return os.path.join(self.paths.storage_real_root, path[len("storage/") :])
        return os.path.join(self.paths.research_root, path)

    def _display_storage_path(self, path: str) -> str:
        real_path = os.path.realpath(path)
        storage_root = os.path.realpath(self.paths.storage_real_root)
        try:
            rel = os.path.relpath(real_path, storage_root)
        except ValueError:
            return path
        if not rel.startswith(".."):
            return f"storage/{rel}"
        return path
