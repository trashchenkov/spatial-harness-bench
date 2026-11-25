"""Isolate native agents from legacy evaluators and exchange only public inputs."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from selectors import EVENT_READ, DefaultSelector
from stat import S_ISREG
from tempfile import TemporaryDirectory, TemporaryFile
from time import monotonic
from uuid import uuid4

# No verifier code or checkout is imported by the process inside Docker.
if __package__:
    from . import suite_native as native
    from .suite_attempt import Attempt
else:
    import suite_native as native
    from suite_attempt import Attempt


FILE_OUTPUTS = {
    "ascii_recall": ("desc.txt", *(f"{i}.txt" for i in range(10))),
    "arcade_clips": ("clips.rix", "query.py"),
    "building": ("answer.json",),
    "adnd_route": ("answer.json",),
}
CAP = 64 * 1024 * 1024


def container_args(network, name, run_id=None):
    flags = ["docker", "run", "--rm", "--pull=never", "--name", name, "-i", "--network", network,
             "--user", "65532:65532", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
             "--memory", "1g", "--memory-swap", "1g", "--cpus", "2", "--pids-limit", "256",
             "--ulimit", f"fsize={CAP}:{CAP}", "--log-driver", "none", "--env", "HOME=/tmp",
             "--workdir", "/tmp"]
    if run_id:
        flags.extend(("--label", f"spatial-suite={run_id}"))
    return flags


def file_command(agent, model, command):
    if agent == "codex":
        return [*command, "exec", "--ephemeral", "--json", "--skip-git-repo-check", "--ignore-rules",
                "--sandbox", "danger-full-access", "-c", 'approval_policy="never"',
                *native.codex_config(model), "-"]
    if agent == "opencode":
        return [*command, "run", "--standalone", "--auto", "--format", "json", "--model", model]
    if agent == "pi":
        return [*command, *native.pi_flags(model), "--print", "--mode", "json"]
    return command


def safe_artifact(folder, name):
    # The container has exited. Do not treat links, devices, pipes or oversized
    # files as submissions, and never copy back unrequested paths.
    try:
        path = folder / name
        attrs = path.lstat()
        if S_ISREG(attrs.st_mode) and attrs.st_nlink == 1 and attrs.st_size <= CAP:
            with path.open("rb") as fh:
                return fh.read(CAP + 1)
    except OSError:
        pass
    return None


def docker_command(image, network, run_id, command, *, stage=None, helper=False):
    name = "spatial-" + uuid4().hex
    flags = container_args(network, name, run_id)
    if stage is not None:
        flags += ["--mount", f"type=bind,src={stage},dst=/task", "--workdir", "/task"]
    if helper:
        source = Path(native.__file__).resolve()
        flags += ["--mount", f"type=bind,src={source},dst=/suite_native.py,readonly"]
        command = ["python3", "/suite_native.py", *command]
    return name, [*flags, "--entrypoint", command[0], image, *command[1:]]


def bounded_run(argv, prompt, timeout, attempt=None):
    """Drain both output pipes without buffering an unlimited agent transcript."""
    deadline = None if timeout is None else monotonic() + timeout
    with TemporaryFile() as stdin, DefaultSelector() as streams:
        if prompt is not None:
            stdin.write(prompt)
            stdin.seek(0)
        with subprocess.Popen(argv, stdin=None if prompt is None else stdin,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
            output, diagnostics = bytearray(), bytearray()
            streamed = prompt is None
            truncated = False
            stderr_truncated = False
            last_snapshot = -1.0
            streams.register(child.stdout, EVENT_READ, output)
            streams.register(child.stderr, EVENT_READ, diagnostics)
            try:
                while streams.get_map():
                    remaining = None if deadline is None else deadline - monotonic()
                    if remaining is not None and remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, timeout)
                    for key, _ in streams.select(remaining):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            streams.unregister(key.fileobj)
                            continue
                        if streamed and key.data is output:
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.buffer.flush()
                            # Do not make an episodic run fail merely because its transcript is long.
                            space = max(0, 1024 * 1024 - len(output))
                            output.extend(chunk[:space])
                            truncated |= len(chunk) > space
                            now = monotonic()
                            if space and attempt and now - last_snapshot >= 0.5:
                                attempt.capture("stdout.log", bytes(output), truncated=True, refresh=True)
                                last_snapshot = now
                        else:
                            key.data.extend(chunk)
                            cap = 8 * 1024 * 1024 if key.data is output else 128 * 1024
                            if len(key.data) > cap:
                                if not streamed:
                                    raise OSError("agent logs exceeded transport limit")
                                del key.data[:-cap]
                                stderr_truncated = True
                wait_timeout = None if deadline is None else max(0, deadline - monotonic())
                return subprocess.CompletedProcess(argv, child.wait(timeout=wait_timeout),
                                                   bytes(output), bytes(diagnostics))
            except BaseException:
                truncated = stderr_truncated = True
                child.kill()
                save_diagnostics(bytes(diagnostics))
                raise
            finally:
                if attempt:
                    capture_output(attempt, bytes(output), bytes(diagnostics),
                                   truncated=truncated, stderr_truncated=stderr_truncated)


def capture_output(attempt, stdout, stderr, *, truncated=False, stderr_truncated=False):
    attempt.capture("stdout.log", stdout, truncated=truncated, refresh=True)
    for line in stderr.splitlines():
        if line.startswith(native.PARTIAL_PREFIX):
            try:
                partial = json.loads(line[len(native.PARTIAL_PREFIX):])
            except (ValueError, RecursionError):
                continue
            if isinstance(partial, str):
                attempt.capture("partial-answer.txt", partial.encode("utf-8", errors="replace"), truncated=True)
    attempt.capture("stderr.log", ordinary_stderr(stderr), truncated=stderr_truncated)
    records = list(native.diagnostics(stderr))[-32:]
    if records:
        attempt.update(runtime=records)


def ordinary_stderr(stderr):
    return b"".join(line for line in stderr.splitlines(keepends=True) if not line.startswith(native.PARTIAL_PREFIX))


def save_diagnostics(stderr):
    filename = os.environ.get("SPATIAL_SUITE_DIAGNOSTICS")
    records = list(native.diagnostics(stderr))
    if filename and records:
        with Path(filename).open("a", encoding="utf-8") as log:
            log.writelines(json.dumps(record, separators=(",", ":")) + "\n" for record in records)


def invoke(image, network, run_id, command, *, stage=None, helper=False, prompt=None, timeout=None, attempt=None):
    name, argv = docker_command(image, network, run_id, command, stage=stage, helper=helper)
    try:
        result = bounded_run(argv, prompt, timeout, attempt)
        save_diagnostics(result.stderr)
        if attempt:
            attempt.update(worker_returncode=result.returncode)
        if result.stderr:
            stderr = ordinary_stderr(result.stderr)
            sys.stderr.buffer.write(attempt.clean(stderr, incomplete=prompt is None) if attempt else stderr)
        return result
    finally:
        # Handling a timeout in docker(1) does not necessarily stop the daemon job.
        removed = subprocess.run(["docker", "rm", "-fv", name], check=False, capture_output=True, timeout=15)
        if removed.returncode and b"No such container" not in removed.stderr:
            raise OSError("cannot clean up agent container")


def source_request(folder, prompt):
    stage, difficulty = (os.environ[k] for k in ("SVG_RECALL_STAGE", "SVG_RECALL_DIFFICULTY"))
    count = int(os.environ["SVG_RECALL_COUNT"])
    if stage == "draw":
        return (stage, count,
                prompt + '\nWrite no files. Return only JSON {"drawings":[{"name":"…","svg":"…"}]}, '
                f'one entry for each of {count} drawings in order. No tools.')
    if stage != "code":
        raise ValueError("invalid source stage")
    names = ["board"] if difficulty in {"hard", "expert", "master"} else [str(i) for i in range(count)]
    source = "\n".join(f"\n--- {name}.svg ---\n" + (folder / f"{name}.svg").read_text(encoding="utf-8")
                       for name in names)
    return (stage, count, prompt +
            f'\nReturn only JSON {{"answers":[...]}}, exactly {count} strings in question/file order. '
            'Use an empty string to skip. No tools.\n' + source)


def source_artifacts(folder, stage, count, data):
    reason = None
    names, rows = [], []
    try:
        payload = native.json_answer(data)
    except (TypeError, ValueError, RecursionError):
        reason = "invalid_json"
    else:
        key = "drawings" if stage == "draw" else "answers"
        if not isinstance(payload, dict) or set(payload) != {key}:
            reason = "unexpected_fields"
        elif not isinstance(payload[key], list) or len(payload[key]) != count:
            reason = "invalid_list_or_count"
        else:
            rows = payload[key]
            if stage == "draw":
                if any(not isinstance(x, dict) or set(x) != {"svg", "name"} or
                       not all(isinstance(v, str) for v in x.values()) for x in rows):
                    reason = "invalid_drawing"
                else:
                    names = [row["name"] for row in rows]
            else:
                names = rows
            if any(not isinstance(name, str) or "\n" in name or "\r" in name for name in names):
                reason = "invalid_answer_string"
    if reason:
        # Keep the historical empty-artifact fallback and let the verifier score it.
        names, rows = [], []
    files = []
    if stage == "draw":
        for i in range(count):
            filename = f"{i}.svg"
            (folder / filename).write_text(rows[i]["svg"] if i < len(rows) else "", encoding="utf-8")
            files.append(filename)
    (folder / "desc.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    files.append("desc.txt")
    return {"status": "rejected" if reason else "accepted", "reason": reason,
            "empty_artifact_fallback": bool(reason), "files": files}


def execute(attempt, image, network, seconds, run_id, family, mode, agent, model, *command):
    timeout = float(seconds)
    try:
        if mode == "session":
            return invoke(image, network, run_id,
                          ["session", agent, model, seconds, *command], helper=True, attempt=attempt).returncode
        if mode == "stream":
            # The stock Pi bridge owns the pipes and enforces its protocol.
            return invoke(image, network, run_id, command, attempt=attempt).returncode
        folder = Path.cwd()
        if mode == "source":
            stage, count, prompt = source_request(folder, sys.stdin.read())
            result = invoke(image, network, run_id, ["source", agent, model, seconds, *command],
                            helper=True, prompt=prompt.encode(), timeout=timeout + 15, attempt=attempt)
            if result.returncode == 0:
                attempt.update(parser={"status": "materializing"})
                try:
                    parser = source_artifacts(folder, stage, count, result.stdout.decode("utf-8"))
                except (OSError, ValueError) as exc:
                    attempt.update(parser={"status": "error", "reason": type(exc).__name__})
                    raise
                attempt.update(parser=parser)
            return result.returncode
        prompt = sys.stdin.buffer.read()
        with TemporaryDirectory(prefix="spatial-public-") as tmp:
            stage = Path(tmp) / "work"
            shutil.copytree(folder, stage, symlinks=True)
            stage.chmod(0o1777)
            for path in stage.rglob("*"):
                if path.is_symlink():
                    raise ValueError("public input is a symlink")
                if path.is_dir():
                    path.chmod(0o755)
                else:
                    path.chmod(0o644)
            result = invoke(image, network, run_id, file_command(agent, model, command),
                          stage=stage.resolve(), prompt=prompt, timeout=timeout, attempt=attempt)
            if agent in {"codex", "opencode", "pi"} and native.event_failure(native.events(result.stdout)):
                return 75
            if result.returncode == 0:
                files = []
                for filename in FILE_OUTPUTS[family]:
                    data = safe_artifact(stage, filename)
                    if data is not None:
                        (folder / filename).write_bytes(data)
                        files.append(filename)
                attempt.update(files=files)
            return result.returncode
    except subprocess.TimeoutExpired:
        save_diagnostics(b"spatial-native: sandbox_timeout\n")
        native.report_failure("sandbox_timeout")
        attempt.update(reason="sandbox_timeout")
        return 70
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Agent transport failed: {type(exc).__name__}", file=sys.stderr, flush=True)
        attempt.update(reason=type(exc).__name__)
        return 70


def main():
    args = sys.argv[1:]
    family, mode = args[4:6]
    attempt = Attempt(os.environ.get("SPATIAL_SUITE_ATTEMPTS"), family=family, mode=mode,
                      stage=os.environ.get("SVG_RECALL_STAGE", mode))
    try:
        code = execute(attempt, *args)
    except BaseException as exc:
        attempt.update(reason=type(exc).__name__)
        attempt.finish(None)
        raise
    attempt.finish(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
