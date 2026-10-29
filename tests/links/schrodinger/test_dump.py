"""Failure-dump policy for the Schrodinger links.

No Schrodinger installation needed: the policy is exercised through the mixin's two entry
points rather than by running a tool.

The distinction being tested is the whole point of the feature. A BATCH-level failure (the
tool exited non-zero, or produced nothing for any molecule) is rare and needs
investigating, so it writes a dump and keeps the tempdir. A PER-ROW failure (one molecule
that cannot be prepared, one pose that misses a constraint) is ordinary stochastic output
of a generative run -- one campaign measured ~21% of designs non-strippable -- so dumping
on those would write thousands of directories across a 500-step run and fill the disk.
"""
import types

import pytest

from pdchemchain.links.schrodinger import LigPrep
from pdchemchain.links.schrodinger._dump import find_job_ids

# Verbatim from a real failure (GLS1 run3_dissim, 2026-10-27). The job id sits ~18 lines
# into stderr, behind an unrelated warning and a Python traceback, which is exactly why it
# was missed by eye and has to be parsed out.
REAL_STDERR = """WARNING The custom schrodinger.hosts file SCHRODINGER_HOSTS(/homes/x/schrodinger.hosts) is being ignored.
FATAL getJobRecord(localhost:41371): Channel stuck in TRANSIENT_FAILURE. retry started: 2026-10-27T07:14:24-05:00
[GRPC connection error 14, retry_min_interval(s)=1 retry_max_time(s)=7200]
Traceback (most recent call last):
  File "ligprep3_startup.py", line 200, in <module>
    main()
schrodinger.infra.mmcheck.MmException: mmjob_wait_and_download returned error code 27 \
(MMJOB_ERROR) for arguments ('ea895631-d1ff-11f1-aebe-10ffe030ac0c',) with message: "..."
"""


def test_finds_job_id_in_real_failure_output():
    assert find_job_ids(REAL_STDERR) == ["ea895631-d1ff-11f1-aebe-10ffe030ac0c"]


def test_job_ids_deduplicated_and_ordered():
    a, b = "aaaaaaaa-1111-2222-3333-444444444444", "bbbbbbbb-1111-2222-3333-444444444444"
    assert find_job_ids(f"{a} {b} {a}") == [a, b]


def test_no_job_id_is_not_an_error():
    """The 2026-10-27 crash had none in some streams; absence must not raise."""
    assert find_job_ids("nothing here", None) == []


def test_per_row_failure_removes_tempdir(tmp_path):
    """No failure recorded -> ordinary cleanup. This is the common case."""
    link = LigPrep(dump_dir=str(tmp_path / "dumps"))
    tmpdir = tmp_path / "td"
    tmpdir.mkdir()

    link._cleanup_tempdir(tmpdir)

    assert not tmpdir.exists()
    assert not list((tmp_path / "dumps").glob("*/dump.log"))


def test_batch_failure_writes_dump_and_keeps_tempdir(tmp_path):
    link = LigPrep(dump_dir=str(tmp_path / "dumps"))
    tmpdir = tmp_path / "td"
    tmpdir.mkdir()
    (tmpdir / "input.sdf").write_text("placeholder")
    result = types.SimpleNamespace(stdout="the real error was on stdout", stderr=REAL_STDERR)

    link._record_failure("exited rc=1", ["/bin/true", "-x"], tmpdir, result)
    link._cleanup_tempdir(tmpdir)

    assert tmpdir.exists(), "tempdir must survive a batch failure for inspection"
    dumps = list((tmp_path / "dumps").glob("*/dump.log"))
    assert len(dumps) == 1
    text = dumps[0].read_text()
    assert "ea895631-d1ff-11f1-aebe-10ffe030ac0c" in text
    assert "the real error was on stdout" in text, "stdout is where the cause often is"
    assert "/bin/true" in text, "the command must be re-runnable by hand"
    assert "jsc info" in text, "recovery hints must be present"


def test_failure_flag_resets_between_batches(tmp_path):
    """A link instance is reused for every batch; one failure must not pin the flag."""
    link = LigPrep(dump_dir=str(tmp_path / "dumps"))
    failed = tmp_path / "failed"
    failed.mkdir()
    link._record_failure("exited rc=1", None, failed, None)
    link._cleanup_tempdir(failed)
    assert failed.exists()

    later = tmp_path / "later"
    later.mkdir()
    link._cleanup_tempdir(later)
    assert not later.exists(), "a later successful batch must still clean up"


def test_first_failure_reason_wins(tmp_path):
    """A cascade follows from the first cause; one dump per batch, not three."""
    link = LigPrep(dump_dir=str(tmp_path / "dumps"))
    tmpdir = tmp_path / "td"
    tmpdir.mkdir()

    link._record_failure("exited rc=1", None, tmpdir, None)
    link._record_failure("no output produced", None, tmpdir, None)

    dumps = list((tmp_path / "dumps").glob("*/dump.log"))
    assert len(dumps) == 1
    assert "exited rc=1" in dumps[0].read_text()


def test_keep_tempdir_still_honoured(tmp_path):
    link = LigPrep(keep_tempdir=True, dump_dir=str(tmp_path / "dumps"))
    tmpdir = tmp_path / "td"
    tmpdir.mkdir()

    link._cleanup_tempdir(tmpdir)

    assert tmpdir.exists()


def test_dump_failure_never_raises(tmp_path, monkeypatch):
    """Diagnostics must not turn a recoverable failure into a crash."""
    link = LigPrep(dump_dir="/proc/cannot/write/here")
    tmpdir = tmp_path / "td"
    tmpdir.mkdir()

    link._record_failure("exited rc=1", None, tmpdir, None)  # must not raise
    link._cleanup_tempdir(tmpdir)

    assert tmpdir.exists(), "tempdir still kept even though the dump could not be written"
