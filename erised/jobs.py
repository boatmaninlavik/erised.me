"""
Async job management for long-running generation tasks.
"""

import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum
from datetime import datetime


class JobStatus(str, Enum):
    PENDING = "pending"
    GENERATING_A = "generating_a"
    GENERATING_B = "generating_b"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    status: JobStatus
    prompt: str
    lyrics: str
    created_at: datetime
    result_a: Optional[Dict[str, Any]] = None
    result_b: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: int = 0  # 0-100


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(self, prompt: str, lyrics: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            prompt=prompt,
            lyrics=lyrics,
            created_at=datetime.now(),
        )
        with self._lock:
            self._jobs[job_id] = job
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                for k, v in kwargs.items():
                    setattr(job, k, v)

    def set_result_a(self, job_id: str, result: Dict[str, Any]):
        self.update_job(job_id, result_a=result, status=JobStatus.GENERATING_B, progress=50)

    def set_result_b(self, job_id: str, result: Dict[str, Any]):
        self.update_job(job_id, result_b=result, status=JobStatus.COMPLETE, progress=100)

    def set_failed(self, job_id: str, error: str):
        self.update_job(job_id, status=JobStatus.FAILED, error=error)


# Global instance
job_manager = JobManager()
