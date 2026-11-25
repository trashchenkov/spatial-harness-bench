"""Private, evaluator-side diagnostics. Never pass this directory to a worker."""

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from uuid import uuid4

LOG_LIMIT = 256 * 1024
# Include only known literal environment secrets, not the environment itself.
SECRET_NAME = re.compile(r"(?:TOKEN|(?:API|ACCESS|SECRET|PRIVATE)[_-]?KEY|PASSWORD|SECRET)$", re.IGNORECASE)
BEARER = re.compile(r"\bBearer[ \t]+[^\s\"',;}]+", re.IGNORECASE)
HEADER = re.compile(r"(?im)(^\s*(?:proxy-)?authorization\s*:\s*|^\s*x-api-key\s*:\s*)[^\r\n]*")
JSON_SECRET = re.compile(
    r'("(?:authorization|proxy-authorization|x-api-key|api[_-]?key|'
    r'(?:access[_-]?|refresh[_-]?|id[_-]?)?token|password)"\s*[:,]\s*)'
    r'"(?:[^"\\]|\\.)*(?:"|\\?\Z)',
    re.IGNORECASE,
)


def redact(text, secrets, *, incomplete=False):
    # Also remove JSON-escaped versions of known literal values.
    for value in secrets:
        for spelling in {value, json.dumps(value, ensure_ascii=True)[1:-1]}:
            text = text.replace(spelling, "[REDACTED]")
            if incomplete:
                for length in range(min(len(spelling) - 1, len(text)), 7, -1):
                    if text.endswith(spelling[:length]):
                        text = text[:-length] + "[REDACTED]"
                        break
    text = HEADER.sub(r"\g<1>[REDACTED]", text)
    text = JSON_SECRET.sub(r'\g<1>"[REDACTED]"', text)
    return BEARER.sub("Bearer [REDACTED]", text)


class Attempt:
    def __init__(self, root, *, family, mode, stage):
        self.path = None
        self.clock = monotonic()
        self.record = {
            "schema_version": 1, "id": uuid4().hex, "family": family, "mode": mode, "stage": stage,
            "started_at": datetime.now(UTC).isoformat(), "status": "running", "streams": {},
        }
        if root:
            parent = Path(root)
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.path = parent / self.record["id"]
            self.path.mkdir(mode=0o700)
            self.secrets = sorted({v for k, v in os.environ.items() if SECRET_NAME.search(k) and len(v) >= 4},
                                  key=len, reverse=True)
            self.update()

    def update(self, **fields):
        if self.path is None:
            return
        self.record.update(fields)
        draft = self.path / ".attempt.tmp"
        draft.unlink(missing_ok=True)
        self.write(draft, (json.dumps(self.record, ensure_ascii=False, indent=2) + "\n").encode())
        draft.replace(self.path / "attempt.json")

    @staticmethod
    def write(path, data):
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            stream.write(data)

    def clean(self, data, *, incomplete=False):
        if self.path is None:
            return data
        return redact(data.decode("utf-8", errors="replace"), self.secrets, incomplete=incomplete).encode("utf-8")

    def capture(self, name, data, *, truncated=False, refresh=False):
        if self.path is None or not data:
            return
        artifact = self.path / name
        if artifact.exists() and not refresh:
            return
        # Redact before truncating so a clipped credential cannot leave a bare prefix.
        clean = self.clean(data, incomplete=truncated)
        limited = clean[:LOG_LIMIT].decode("utf-8", errors="ignore").encode("utf-8")
        draft = self.path / f".{name}.tmp"
        draft.unlink(missing_ok=True)
        self.write(draft, limited)
        draft.replace(artifact)
        self.record["streams"][name] = {"bytes": len(limited), "truncated": truncated or len(clean) > LOG_LIMIT}
        self.update()

    def finish(self, code):
        self.update(status="complete" if code == 0 else "error", returncode=code,
                    elapsed_seconds=round(monotonic() - self.clock, 3))
