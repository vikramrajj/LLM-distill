#!/usr/bin/env python3
"""Experience buffer for collecting agent trajectories (for distillation)."""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Experience:
    """A single agent experience (trajectory)."""
    task: str
    steps: list[dict]
    success: bool
    reward: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ExperienceBuffer:
    """Buffer for collecting and managing agent experiences."""

    def __init__(self, path: str = "experiences.jsonl"):
        self.path = Path(path)
        self.buffer: list[Experience] = []
        self._load_existing()

    def _load_existing(self):
        """Load existing experiences from disk."""
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self.buffer.append(Experience(**data))

    def add(self, experience: Experience):
        """Add an experience to the buffer and persist it."""
        self.buffer.append(experience)
        with open(self.path, "a") as f:
            f.write(json.dumps(experience.to_dict()) + "\n")

    def get_successes(self, min_reward: float = 0.0) -> list[Experience]:
        """Get successful experiences with reward >= min_reward."""
        return [e for e in self.buffer if e.success and e.reward >= min_reward]

    def get_all(self) -> list[Experience]:
        """Get all experiences."""
        return self.buffer

    def __len__(self):
        return len(self.buffer)
