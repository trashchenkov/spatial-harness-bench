"""Orchestrate the existing benchmark CLIs without changing their scores."""

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4


@dataclass(frozen=True)
class Block:
    name: str
    family: str
    mode: str
    options: tuple[str, ...] = ()


def catalog():
    blocks = [
        Block("ascii_recall.assigned", "ascii_recall", "file", ("--objects", "assigned")),
        Block("ascii_recall.free", "ascii_recall", "file"),
        Block("maze", "maze", "session", ("--suite",)),
        *(Block(f"svg_recall.{level}", "svg_recall", "source", ("--difficulty", level))
          for level in ("basic", "hard", "expert", "master")),
        *(Block(f"svg_scenes.{dataset}", "svg_scenes", "source", ("--set", dataset))
          for dataset in ("fixed", "generated")),
        Block("arcade_clips", "arcade_clips", "file"),
        Block("building", "building", "file"),
        Block("adnd_route", "adnd_route", "file"),
        Block("adnd_memory", "adnd_memory", "session", ("--save-private",)),
        Block("adnd_expedition", "adnd_expedition", "session"),
    ]
    if importlib.util.find_spec("spatial_harness_bench.engineering_geometry") is not None:
        blocks.append(Block("engineering_geometry", "engineering_geometry", "geometry",
                            ("--family", "all", "--count", "5")))
    return blocks


def select(selection, blocks):
    if selection is None:
        return blocks
    keys = {key.strip() for key in selection.split(",")}
    unknown = keys - {key for block in blocks for key in (block.name, block.family)}
    if unknown:
        raise ValueError("unknown selectors: " + ", ".join(sorted(unknown)))
    return [block for block in blocks if keys.intersection((block.name, block.family))]


def options(block, smoke, seed=None):
    selected = block.options
    if smoke:
        selected = {
            "maze": ("--level", "1"),
            "adnd_expedition": ("--max-turns", "2"),
            "building": ("--level", "1"),
            "svg_scenes.generated": (*block.options, "--per-tier", "1"),
            "engineering_geometry": ("--family", "add-core", "--count", "1"),
        }.get(block.name, block.options)
    if seed is not None and block.name in {
        "ascii_recall.assigned", "maze", "svg_recall.hard", "svg_recall.expert", "svg_recall.master",
        "svg_scenes.generated", "adnd_route", "adnd_memory", "adnd_expedition", "engineering_geometry",
    }:
        selected += ("--seed", str(seed))
    return selected

def docker_check(*args):
    result = subprocess.run(["docker", *args], check=False, capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise ValueError(f"Docker preflight failed ({args[0]}): {result.stderr.strip()[:400]}")
    return result.stdout.strip()


def preflight(blocks, args, binary):
    # Check evaluator-side dependencies before spending any model sessions.
    if any(block.family == "arcade_clips" for block in blocks):
        if sys.platform != "linux":
            raise ValueError("Arcade Clips evaluation requires Linux with Bubblewrap and system Python 3.12")
        from .arcade_clips.sandbox import query_command
        with TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.py"
            probe.write_text("import json\n", encoding="utf-8")
            subprocess.run(query_command(probe), env={"PATH": os.defpath}, check=True, timeout=15)
    if any(block.family == "svg_recall" for block in blocks):
        from .svg_recall import verifier  # noqa: F401
        languages = subprocess.check_output(["tesseract", "--list-langs"], text=True, timeout=15)
        if not {"rus", "eng"}.issubset(languages.splitlines()):
            raise ValueError("SVG Recall requires Tesseract with rus and eng")
    if any(block.family == "svg_scenes" for block in blocks):
        from .svg_scenes import __main__  # noqa: F401
    if args.network != "none" and docker_check("network", "inspect", "--format", "{{.Driver}} {{.Internal}}", args.network) != "bridge true":
        raise ValueError("the harness network must be a dedicated internal Docker bridge")
    image = docker_check("image", "inspect", "--format", "{{.Id}}", args.image)
    from .suite_adapter import container_args
    needs_python = any(block.mode in {"source", "session"} and args.agent in {"codex", "opencode", "pi"}
                       for block in blocks)
    probe = ["python3", "-c", "import shutil,sys;sys.exit(not shutil.which(sys.argv[1]))", binary]
    if not needs_python:
        # Non-native file agents need not ship Python in the harness image.
        probe = ["/bin/sh", "-c", 'command -v "$1" >/dev/null', "probe", binary]
    docker_check(*container_args("none", "spatial-probe-" + uuid4().hex)[1:],
                 "--entrypoint", probe[0], image, *probe[1:])
    version = None
    if args.agent:
        result = docker_check(*container_args("none", "spatial-probe-" + uuid4().hex)[1:],
                              "--entrypoint", binary, image, "--version")
        version = result.splitlines()[-1] if result else None
    return image, version


def invocation(block, args, command, image, output, run_id):
    runner = [sys.executable, "-m", f"spatial_harness_bench.{block.family}",
              "--output", str(output), *options(block, args.smoke, args.seed)]
    if block.mode == "geometry":
        from .suite_adapter import file_command
        return [*runner, "--image", image, "--network", args.network, "--timeout", str(args.timeout),
                "--", *file_command(args.agent, args.model, command)]
    if block.mode == "session":
        if block.family == "adnd_memory":
            runner += ["--io-timeout", str(args.timeout + 20)]
        elif block.family == "adnd_expedition":
            runner += ["--timeout", str(args.timeout + 20)]
    pi_bridge = args.agent == "pi" and block.family in {"maze", "adnd_expedition"}
    launcher = [sys.executable, str(Path(__file__).with_name("suite_adapter.py")), image, args.network,
                str(args.timeout), run_id, block.family,
                "stream" if pi_bridge else block.mode, args.agent or "external",
                args.model or "", *command]
    if pi_bridge:
        launcher.extend(("--offline", "--no-mcp"))
        if args.model:
            launcher.extend(("--model", args.model))
        module = {"maze": "maze.pi", "adnd_expedition": "adnd_expedition.pi"}[block.family]
        launcher = [sys.executable, "-m", f"spatial_harness_bench.{module}", "--", *launcher]
    return [*runner, "--", *launcher]


def summary(output, returncode, agent=None):
    diagnostics = output.parent / f"{output.name}.diagnostics.jsonl"
    records = []
    try:
        with diagnostics.open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
    except (OSError, ValueError):
        pass
    failure = next((r for r in reversed(records)
                    if r.get("code") not in {"provider_http", "process_guard_verified"}), None)
    last_http = next((r["http_status"] for r in reversed(records) if "http_status" in r), None)
    extra = {"native_diagnostic": failure} if failure else {}
    if last_http is not None:
        extra["last_provider_http_status"] = last_http

    path = output / "result.json"
    if not path.is_file():
        # Maze emits suite.json for a ten-level run.
        path = output / "suite.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "technical_error", "score": None,
                "reason": failure["code"] if failure else "missing_or_invalid_runner_report", **extra}
    if not isinstance(report, dict):
        return {"status": "technical_error", "score": None, "reason": "invalid_runner_report"}
    score = report.get("modes", {}).get("code", report)
    technical = score.get("status") in {"error", "technical_error"} or bool(report.get("technical_errors"))
    technical |= bool(returncode and not report.get("agent_errors"))
    relative = str(path.relative_to(output.parent))
    value = score.get("score")
    maximum = score.get("max_score")
    if report.get("task", "").startswith("02-maze-") and "suite" not in report:
        value, maximum = int(report.get("status") == "solved"), 1
    if "technical_errors" in report and agent in {"codex", "opencode", "pi"}:
        from . import suite_native
        for row in report.get("items", []):
            log = output / row["artifacts"] / "stdout.log"
            if log.is_file() and suite_native.event_failure(suite_native.events(log.read_bytes())):
                technical = True
    reason = score.get("reason") or next(
        (item["reason"] for item in report.get("items", []) if item.get("reason")), None)
    return {
        "status": "technical_error" if technical else ("agent_error" if report.get("agent_errors") else "complete"),
        "score": None if technical else value, "max_score": maximum, "artifact": relative,
        "reason": failure["code"] if technical and failure else reason,
        **(extra if technical else {}),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, prog="python -m spatial_harness_bench run")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--select", metavar="FAMILY[,BLOCK…]")
    parser.add_argument("--agent", choices=("codex", "opencode", "pi"))
    parser.add_argument("--model", help="model ID or provider/model selector for the native CLI")
    parser.add_argument("--model-label", help="resolved model name when the selector is an alias")
    parser.add_argument("--image", default=os.environ.get("SPATIAL_AGENT_IMAGE"), help="local configured harness image")
    parser.add_argument("--network", default=os.environ.get("SPATIAL_AGENT_NETWORK", "none"))
    parser.add_argument("--output", type=Path, help="new evaluator-only result directory")
    parser.add_argument("--timeout", type=float, default=85, help="seconds per native turn or file stage")
    parser.add_argument("--seed", type=int, help="public development seed shared by generated blocks")
    parser.add_argument("--smoke", action="store_true", help="shorten supported blocks; not a ranked profile")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without Docker or model calls")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND [ARG ...] for file-capable agents")
    args = parser.parse_args(argv)
    args.command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if bool(args.agent) == bool(args.command):
        parser.error("specify one of --agent NAME or -- COMMAND")
    if args.agent in {"codex", "opencode"} and not args.model:
        parser.error("specify --model for Codex/OpenCode")
    if args.agent == "opencode" and "/" not in args.model:
        parser.error("OpenCode requires --model PROVIDER/MODEL, as configured in the image")
    if args.model and not args.agent:
        parser.error("for an external command, select its model in the command itself")
    if args.seed is not None and not 0 <= args.seed < 2**62:
        parser.error("--seed must be in [0, 2^62)")
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be positive and finite")
    if not re.fullmatch(r"[A-Za-z0-9_.][A-Za-z0-9_.-]*", args.network) or args.network == "host":
        parser.error("use none or a dedicated Docker bridge")
    if args.image and args.image.startswith("-"):
        parser.error("invalid Docker image name")
    try:
        blocks = select(args.select, catalog())
        for block in blocks:
            if block.mode in {"source", "session"} and not args.agent:
                raise ValueError(f"{block.name} needs a {block.mode} transport; the selected CLI has none")
        command = [args.agent] if args.agent else args.command
        if args.dry_run:
            print(json.dumps({"agent": args.agent or "external", "model_selector": args.model, "smoke": args.smoke,
                              "blocks": [{"id": b.name, "family": b.family, "transport": b.mode,
                                          "options": options(b, args.smoke, args.seed)} for b in blocks]}, indent=2))
            return 0
        if not args.image or not args.output:
            raise ValueError("--image (or SPATIAL_AGENT_IMAGE) and --output are required")
        output = args.output.resolve()
        if output.exists():
            raise ValueError("choose a new output directory")
        image, version = preflight(blocks, args, command[0])
    except (ImportError, ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))

    run_id = uuid4().hex
    report = {"schema_version": 1, "status": "running", "agent": args.agent or "external",
              "executable": command[0], "model_selector": args.model,
              "model": args.model_label or args.model, "agent_version": version,
              "image_id": image, "network": args.network, "smoke": args.smoke,
              "blocks": [{"id": b.name, "status": "pending", "options": options(b, args.smoke, args.seed)} for b in blocks]}

    def save():
        draft = output / ".result.tmp"
        draft.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        draft.replace(output / "result.json")

    output.mkdir(parents=True, mode=0o700)
    save()
    try:
        for block, row in zip(blocks, report["blocks"], strict=True):
            target = output / block.name
            print(f"[{block.name}]", flush=True)
            with (output / f"{block.name}.log").open("w", encoding="utf-8") as logs:
                completed = subprocess.run(invocation(block, args, command, image, target, run_id),
                                           check=False, stdout=logs, stderr=subprocess.STDOUT,
                                           env={**os.environ, "SPATIAL_SUITE_DIAGNOSTICS":
                                                str(output / f"{block.name}.diagnostics.jsonl"),
                                                "SPATIAL_SUITE_ATTEMPTS":
                                                str(output / "attempts" / block.name)})
            row.update(summary(target, completed.returncode, args.agent), returncode=completed.returncode)
            attempt_dir = output / "attempts" / block.name
            if attempt_dir.is_dir():
                row["attempts"] = str(attempt_dir.relative_to(output))
            save()
        report["status"] = ("technical_error" if any(r["status"] == "technical_error" for r in report["blocks"])
                            else "complete")
    except (KeyboardInterrupt, OSError):
        report["status"] = "interrupted"
        raise
    finally:
        try:
            containers = docker_check("ps", "-aq", "--filter", f"label=spatial-suite={run_id}")
            if containers:
                docker_check("rm", "-fv", *containers.split())
        except (OSError, ValueError, subprocess.SubprocessError):
            report.update(status="technical_error", cleanup_error=True)
        save()
    print(output / "result.json")
    return 2 if report["status"] == "technical_error" else 0
