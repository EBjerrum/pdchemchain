"""Write a diagnostic dump when a Schrodinger tool fails at the batch level.

Why this exists. A Glide/LigPrep failure is often undiagnosable after the fact: the link
logs ``result.stderr`` (which may carry only a warning), the tool's own log lives in a
temporary directory that is then deleted, and the Schrodinger job directory is named in a
file that job control never transferred back. A real case cost hours of manual work --
re-running the logged command by hand on a login node was the only way to see the actual
error message, which was on *stdout*.

What this writes. One small ``dump.log`` per batch-level failure, containing the full
command, stdout, stderr, the parsed job UUIDs, and copy-pasteable commands for finding the
job directories on the server. No file copying: on a cluster whose scratch is shared with
the login node, recording *where to look* is enough, and copying job directories is both
expensive (~25 MB each, several per step) and exactly the fragile part.

What this is NOT for. Per-row failures -- a molecule that cannot be prepared, a pose that
misses the constraint -- are normal stochastic output of a generative run (one campaign
measured ~21% of designs non-strippable). Dumping on those would write thousands of
directories. Only *batch-level* failures come here.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
from datetime import datetime
from pathlib import Path

# Schrodinger job ids are UUIDs. They appear in stderr inside the MmException, e.g.
#   mmjob_wait_and_download returned error code 27 (MMJOB_ERROR)
#     for arguments ('ea895631-d1ff-11f1-aebe-10ffe030ac0c',) with message: ...
# about 18 lines into the output, which is why they are easy to miss by eye.
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)

DEFAULT_DUMP_DIR = Path.home() / "pdchemchain_dumps"


def find_job_ids(*texts: str | None) -> list[str]:
    """Job UUIDs mentioned in the given output, in first-seen order."""
    seen: dict[str, None] = {}
    for text in texts:
        for match in _UUID_RE.findall(text or ""):
            seen.setdefault(match, None)
    return list(seen)


def _job_root() -> str:
    """Where the job server puts job directories, per the hosts file ``tmpdir:``.

    Read from the hosts file rather than assumed: it differs between installations
    (``/scratch/schrodinger`` on one machine, ``/projects/scratch`` on another) and recent
    Schrodinger versions ignore user-supplied host files, so it is not ours to control.
    """
    schrodinger = os.environ.get("SCHRODINGER", "")
    hosts = Path(schrodinger) / "schrodinger.hosts"
    try:
        for line in hosts.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("tmpdir:"):
                return stripped.split(":", 1)[1].strip()
    except OSError:
        pass
    return "<schrodinger tmpdir, see $SCHRODINGER/schrodinger.hosts>"


def _hints(job_ids: list[str], tmpdir: Path) -> str:
    """Copy-pasteable recovery commands.

    Emitted unconditionally, without checking whether the paths resolve here: the dump is
    read later and often from a different machine (a login node), where they do.
    """
    root = f"{_job_root()}/{os.environ.get('USER', '$USER')}"
    host = socket.gethostname()
    lines = [
        "# --- the pdchemchain tempdir (kept, not deleted, because this run failed) ---",
        f"ls -la {tmpdir}",
        "",
        "# --- ask the job server; authoritative, and survives tmpdir: moving ---",
    ]
    if job_ids:
        for job_id in job_ids:
            lines += [
                f"$SCHRODINGER/jsc info {job_id}",
                f"$SCHRODINGER/jsc list-files {job_id}",
                f"$SCHRODINGER/jsc download {job_id}",
                f"ls -d {root}/jobs/{job_id}",
                "# one run spawns MANY job dirs; find the whole family via the shared token:",
                f"ls -l {root}/jobs/by-name/ | grep {job_id[:8]}",
            ]
    else:
        lines += [
            "# NO job id was found in the output. Find the job by time instead:",
            "$SCHRODINGER/jsc list",
            f"ls -lt {root}/jobs/ | head",
        ]
    lines += [
        "",
        "# --- the job server's own logs (per node) ---",
        f"tail -100 ~/.schrodinger/.jobdb2/jserver.{host}.log",
        f"tail -100 ~/.schrodinger/.jobdb2/jnanny.{host}.log",
        f"ls -d {root}/jobserverd-{host}",
        "",
        "# --- and the single most useful thing: re-run the command by hand ---",
        "# (copy it from 'command:' above; the real error is often only on stdout)",
    ]
    return "\n".join(lines)


def write_dump(
    link_name: str,
    reason: str,
    command: list[str] | None,
    tmpdir: Path,
    stdout: str | None = None,
    stderr: str | None = None,
    dump_dir: Path | str | None = None,
    extra: str | None = None,
) -> Path | None:
    """Write ``dump.log`` for a batch-level failure. Returns its path, or None.

    Never raises: a diagnostics helper must not turn a recoverable failure into a crash.
    """
    try:
        base = Path(dump_dir) if dump_dir is not None else DEFAULT_DUMP_DIR
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = base / f"{stamp}_{link_name}_{os.getpid()}"
        target.mkdir(parents=True, exist_ok=True)

        job_ids = find_job_ids(stderr, stdout)
        parts = [
            f"pdchemchain {link_name} failure",
            f"when:     {datetime.now().isoformat(timespec='seconds')}",
            f"host:     {socket.gethostname()}",
            f"pid:      {os.getpid()}",
            f"reason:   {reason}",
            f"tmpdir:   {tmpdir}   (KEPT for inspection)",
            f"job ids:  {', '.join(job_ids) if job_ids else '(none found in output)'}",
            "",
            "command:",
            "  " + (subprocess.list2cmdline(command) if command else "(not recorded)"),
            "",
            "=== recovery hints " + "=" * 58,
            _hints(job_ids, tmpdir),
            "",
            "=== stdout " + "=" * 66,
            (stdout or "(empty)").rstrip(),
            "",
            "=== stderr " + "=" * 66,
            (stderr or "(empty)").rstrip(),
        ]
        if extra:
            parts += ["", "=== extra " + "=" * 67, extra.rstrip()]

        (target / "dump.log").write_text("\n".join(parts) + "\n")
        return target / "dump.log"
    except Exception:  # diagnostics must never mask the original failure
        return None


class DumpOnFailureMixin:
    """Shared failure-dump behaviour for the Schrodinger links.

    Provides ``_record_failure()`` for the error paths and ``_cleanup_tempdir()`` for the
    ``finally`` block, which keeps the directory when the batch failed.

    The links need a flag rather than an exception because a non-zero return code does
    NOT raise here: the inner method logs the error and returns a DataFrame with
    ``__error__`` set, so per-row failures degrade instead of killing a 500-step run.
    That design is deliberate, which is why ``finally`` cannot otherwise tell success from
    failure.
    """

    # Set per call, not in __post_init__: these links are frozen dataclasses in some
    # configurations and are reused across batches, so state must not persist.
    _failure_reason: str | None = None

    def _record_failure(
        self,
        reason: str,
        command: list[str] | None,
        tmpdir: Path,
        result=None,
        extra: str | None = None,
    ) -> None:
        """Note a BATCH-level failure and write its dump. Safe to call more than once."""
        if getattr(self, "_failure_reason", None) is not None:
            return  # first reason is the useful one; a cascade follows from it
        self._failure_reason = reason

        path = write_dump(
            link_name=type(self).__name__,
            reason=reason,
            command=command,
            tmpdir=tmpdir,
            stdout=getattr(result, "stdout", None),
            stderr=getattr(result, "stderr", None),
            dump_dir=getattr(self, "dump_dir", None),
            extra=extra,
        )
        if path is not None:
            self.logger.error(
                f"{type(self).__name__}: wrote failure diagnostics to {path} "
                f"(tempdir kept at {tmpdir})"
            )

    def _cleanup_tempdir(self, tmpdir: Path) -> None:
        """Remove the tempdir unless it is wanted for inspection.

        Kept on a batch-level failure as well as on ``keep_tempdir``: it holds the .in
        file, the input SDF and any partial output, which is the cheapest possible
        evidence and is otherwise destroyed before anyone can look.
        """
        failed = getattr(self, "_failure_reason", None) is not None
        self._failure_reason = None  # reset before the next batch

        if self.keep_tempdir:
            self.logger.info(f"{type(self).__name__} temp directory kept at: {tmpdir}")
        elif failed:
            self.logger.warning(
                f"{type(self).__name__} temp directory KEPT after failure: {tmpdir}"
            )
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)
