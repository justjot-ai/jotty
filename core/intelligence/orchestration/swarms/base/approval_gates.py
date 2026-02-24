"""
Approval Gates with Resume Tokens
===================================

Enables human-in-the-loop checkpoints in swarm execution.
When an approval gate is hit, execution pauses, state is serialized
to a checkpoint, and a resume token is returned. Execution continues
when the token is used to resume with an approval decision.

Inspired by OpenClaw's Lobster workflow engine approval pattern.

Usage in a SwarmTemplate subclass::

    class DeploySwarm(SwarmTemplate):
        AGENTS = TeamCoordinator.define(
            (BuildAgent, "Builder"),
            (DeployAgent, "Deployer"),
            pattern=CoordinationPattern.PIPELINE,
        )

        APPROVAL_GATES = {
            "pre_deploy": ApprovalGate(
                name="pre_deploy",
                description="Approve deployment to production",
                phase_name="Deploy",
            ),
        }

        async def _execute_domain(self, requirements: str, **kwargs):
            # Phase 1: Build
            build_result = await self._builder.build(requirements)

            # Gate: pause for human approval before deploying
            gate_result = await self.check_approval_gate(
                "pre_deploy",
                checkpoint_data={"build": build_result},
            )
            if gate_result.paused:
                return gate_result.to_swarm_result()  # Returns with resume_token

            # Phase 2: Deploy (only reached after approval)
            return await self._deployer.deploy(build_result)

    # Resume later:
    result = await swarm.execute(resume_token="abc123...")
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of an approval gate."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"


@dataclass
class ApprovalGate:
    """Declares an approval checkpoint in a swarm's execution flow.

    Attributes:
        name: Unique gate identifier within the swarm.
        description: Human-readable description shown to approver.
        phase_name: Which phase this gate follows (for logging).
        timeout_seconds: Auto-reject after this many seconds (0 = no timeout).
    """

    name: str
    description: str = ""
    phase_name: str = ""
    timeout_seconds: int = 0


@dataclass
class ApprovalCheckpoint:
    """Serializable snapshot of swarm state at an approval gate.

    Stored to disk so execution can resume hours/days later.
    """

    checkpoint_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    gate_name: str = ""
    swarm_class: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    original_args: Dict[str, Any] = field(default_factory=dict)
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver: str = ""
    approved_at: str = ""
    rejection_reason: str = ""

    @property
    def resume_token(self) -> str:
        """Generate a compact resume token (base64-encoded checkpoint ID)."""
        return base64.urlsafe_b64encode(self.checkpoint_id.encode()).decode()

    @staticmethod
    def id_from_token(token: str) -> str:
        """Extract checkpoint ID from a resume token."""
        return base64.urlsafe_b64decode(token.encode()).decode()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "gate_name": self.gate_name,
            "swarm_class": self.swarm_class,
            "created_at": self.created_at,
            "checkpoint_data": self.checkpoint_data,
            "original_args": self.original_args,
            "status": self.status.value,
            "approver": self.approver,
            "approved_at": self.approved_at,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalCheckpoint":
        data = dict(data)  # Don't mutate original
        data["status"] = ApprovalStatus(data.get("status", "pending"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GateResult:
    """Result of checking an approval gate.

    If paused=True, the swarm should stop and return the resume_token
    to the caller so they can resume execution later.
    """

    paused: bool = False
    approved: bool = False
    resume_token: str = ""
    checkpoint: Optional[ApprovalCheckpoint] = None
    rejection_reason: str = ""

    def to_swarm_result(self, swarm_name: str = "", domain: str = "") -> Any:
        """Convert to a SwarmResult for returning from _execute_domain."""
        from .._base.swarm_learning import SwarmResult

        if self.paused:
            return SwarmResult(
                success=True,  # Not a failure — just paused
                swarm_name=swarm_name,
                domain=domain,
                output={
                    "status": "awaiting_approval",
                    "resume_token": self.resume_token,
                    "gate": self.checkpoint.gate_name if self.checkpoint else "",
                },
                execution_time=0.0,
            )
        return None


class CheckpointStore:
    """Simple file-based checkpoint persistence.

    Stores checkpoints as JSON files in ~/.jotty/checkpoints/.
    """

    def __init__(self, store_dir: Optional[str] = None) -> None:
        self._dir = Path(store_dir or os.path.expanduser("~/.jotty/checkpoints"))
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: ApprovalCheckpoint) -> str:
        """Save checkpoint to disk. Returns the resume token."""
        path = self._dir / f"{checkpoint.checkpoint_id}.json"
        path.write_text(json.dumps(checkpoint.to_dict(), indent=2, default=str))
        logger.info(
            "Saved approval checkpoint %s for gate '%s'",
            checkpoint.checkpoint_id,
            checkpoint.gate_name,
        )
        return checkpoint.resume_token

    def load(self, checkpoint_id: str) -> Optional[ApprovalCheckpoint]:
        """Load checkpoint from disk."""
        path = self._dir / f"{checkpoint_id}.json"
        if not path.exists():
            logger.warning("Checkpoint not found: %s", checkpoint_id)
            return None
        try:
            data = json.loads(path.read_text())
            return ApprovalCheckpoint.from_dict(data)
        except Exception as e:
            logger.error("Failed to load checkpoint %s: %s", checkpoint_id, e)
            return None

    def load_from_token(self, resume_token: str) -> Optional[ApprovalCheckpoint]:
        """Load checkpoint using a resume token."""
        try:
            checkpoint_id = ApprovalCheckpoint.id_from_token(resume_token)
            return self.load(checkpoint_id)
        except Exception as e:
            logger.error("Invalid resume token: %s", e)
            return None

    def approve(self, checkpoint_id: str, approver: str = "human") -> bool:
        """Mark a checkpoint as approved."""
        checkpoint = self.load(checkpoint_id)
        if not checkpoint:
            return False
        checkpoint.status = ApprovalStatus.APPROVED
        checkpoint.approver = approver
        checkpoint.approved_at = datetime.now().isoformat()
        self.save(checkpoint)
        return True

    def reject(self, checkpoint_id: str, reason: str = "", approver: str = "human") -> bool:
        """Mark a checkpoint as rejected."""
        checkpoint = self.load(checkpoint_id)
        if not checkpoint:
            return False
        checkpoint.status = ApprovalStatus.REJECTED
        checkpoint.approver = approver
        checkpoint.approved_at = datetime.now().isoformat()
        checkpoint.rejection_reason = reason
        self.save(checkpoint)
        return True

    def list_pending(self) -> list[ApprovalCheckpoint]:
        """List all pending checkpoints."""
        pending = []
        for path in self._dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                cp = ApprovalCheckpoint.from_dict(data)
                if cp.status == ApprovalStatus.PENDING:
                    pending.append(cp)
            except Exception:
                continue
        return pending

    def cleanup(self, max_age_days: int = 7) -> int:
        """Remove old checkpoints. Returns count removed."""
        removed = 0
        cutoff = datetime.now()
        for path in self._dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                created = datetime.fromisoformat(data["created_at"])
                if (cutoff - created).days > max_age_days:
                    path.unlink()
                    removed += 1
            except Exception:
                continue
        return removed


# Module-level singleton
_checkpoint_store: Optional[CheckpointStore] = None


def get_checkpoint_store() -> CheckpointStore:
    """Get the singleton CheckpointStore."""
    global _checkpoint_store
    if _checkpoint_store is None:
        _checkpoint_store = CheckpointStore()
    return _checkpoint_store


__all__ = [
    "ApprovalGate",
    "ApprovalCheckpoint",
    "ApprovalStatus",
    "GateResult",
    "CheckpointStore",
    "get_checkpoint_store",
]
