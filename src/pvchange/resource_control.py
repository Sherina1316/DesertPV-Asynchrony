from __future__ import annotations

import getpass
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MemorySnapshot:
    total_gb: float
    available_gb: float
    swap_total_gb: float
    swap_free_gb: float
    competing_python_rss_gb: float
    competing_processes: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    path = Path("/proc/meminfo")
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        token = raw.strip().split()[0]
        try:
            result[key] = int(token) * 1024
        except (ValueError, IndexError):
            continue
    return result


def _competing_python_processes(project_root: str | Path, minimum_rss_mb: float) -> list[dict[str, Any]]:
    project_root = str(Path(project_root).resolve())
    current_pid = os.getpid()
    user = getpass.getuser()
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,user=,rss=,args="],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        pid_raw, process_user, rss_raw, command = parts
        try:
            pid = int(pid_raw)
            rss_kb = int(rss_raw)
        except ValueError:
            continue
        if pid == current_pid or process_user != user:
            continue
        lowered = command.lower()
        if "python" not in lowered:
            continue
        if project_root in command:
            continue
        rss_mb = rss_kb / 1024.0
        if rss_mb < float(minimum_rss_mb):
            continue
        rows.append({
            "pid": pid,
            "rss_mb": float(rss_mb),
            "command": command[:500],
        })
    rows.sort(key=lambda item: item["rss_mb"], reverse=True)
    return rows


def memory_snapshot(project_root: str | Path, minimum_competing_process_rss_mb: float = 512.0) -> MemorySnapshot:
    info = _read_meminfo()
    gib = float(1024**3)
    competing = _competing_python_processes(project_root, minimum_competing_process_rss_mb)
    return MemorySnapshot(
        total_gb=float(info.get("MemTotal", 0) / gib),
        available_gb=float(info.get("MemAvailable", info.get("MemFree", 0)) / gib),
        swap_total_gb=float(info.get("SwapTotal", 0) / gib),
        swap_free_gb=float(info.get("SwapFree", 0) / gib),
        competing_python_rss_gb=float(sum(item["rss_mb"] for item in competing) / 1024.0),
        competing_processes=competing,
    )


def resource_headroom_ok(snapshot: MemorySnapshot, cfg: dict[str, Any]) -> tuple[bool, list[str]]:
    """Admission gate based only on Linux MemAvailable and SwapFree.

    Summed per-process RSS is retained for diagnostics but is intentionally not
    used for admission because multiprocessing jobs double-count shared pages.
    """
    settings = cfg.get("resource_control", {})
    reasons: list[str] = []
    minimum_available = float(settings.get("minimum_available_memory_gb", 16.0))
    minimum_swap_free = float(settings.get("minimum_swap_free_gb", 0.25))
    if snapshot.available_gb < minimum_available:
        reasons.append(
            f"MemAvailable {snapshot.available_gb:.1f} GiB is below required {minimum_available:.1f} GiB"
        )
    if snapshot.swap_total_gb > 0 and snapshot.swap_free_gb < minimum_swap_free:
        reasons.append(
            f"SwapFree {snapshot.swap_free_gb:.2f} GiB is below required {minimum_swap_free:.2f} GiB"
        )
    return not reasons, reasons


def wait_for_resource_headroom(
    cfg: dict[str, Any],
    *,
    stage_label: str,
    poll_seconds: int | None = None,
) -> MemorySnapshot:
    settings = cfg.get("resource_control", {})
    project_root = cfg.get("paths", {}).get("project_root", ".")
    if not bool(settings.get("enabled", True)):
        return memory_snapshot(project_root)
    poll = int(poll_seconds or settings.get("poll_seconds", 30))
    deadline = time.monotonic() + float(settings.get("maximum_wait_minutes", 720.0)) * 60.0
    waited = False
    while True:
        snapshot = memory_snapshot(
            project_root,
            float(settings.get("minimum_competing_process_rss_mb", 512.0)),
        )
        ok, reasons = resource_headroom_ok(snapshot, cfg)
        if ok:
            if waited:
                print(
                    f"Resource headroom restored for {stage_label}: "
                    f"MemAvailable={snapshot.available_gb:.1f} GiB, "
                    f"SwapFree={snapshot.swap_free_gb:.2f} GiB",
                    flush=True,
                )
            return snapshot
        waited = True
        detail = "; ".join(reasons)
        print(
            f"Waiting for resource headroom before {stage_label}: {detail}. "
            f"Current MemAvailable={snapshot.available_gb:.1f} GiB, "
            f"SwapFree={snapshot.swap_free_gb:.2f} GiB.",
            flush=True,
        )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Resource wait timed out before {stage_label}. Last conditions: {detail}"
            )
        time.sleep(max(5, poll))

