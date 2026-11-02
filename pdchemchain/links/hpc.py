import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import psutil

from pdchemchain.base import Link
from pdchemchain.typing import Partitionable


@dataclass(kw_only=True)
class PartitionProcessorBase(Link):
    """Base class for partition processors"""

    partition_size: Optional[int] = field(default=None)
    num_partitions: Optional[int] = field(default=None)
    allow_non_partitionable: bool = False

    def __post_init__(self):
        super().__post_init__()
        if (self.partition_size is not None) and (self.num_partitions is not None):
            raise ValueError(
                "Specify either 'partition_size' or 'num_partitions', not both."
            )
        # Check that the wrapped link is partitionable
        if hasattr(self, 'link') and self.link is not None:
            self._check_partitionable()

    def _check_partitionable(self):
        """Check if the wrapped link/chain is safe to partition.

        Uses ternary logic: NO raises, MAYBE warns, YES passes silently.
        """
        level = self.link._partitionable
        if level == Partitionable.YES:
            return

        bad_links = self.link._non_partitionable_links()
        move_hint = (
            "Move them outside the partition processor — either before it "
            "(to preprocess the full dataset) or after it (to postprocess "
            "the reassembled result)."
        )

        if level == Partitionable.NO:
            msg = (
                f"The following link(s) are not partitionable: {bad_links}. "
                f"They require the full dataset to produce correct results "
                f"(e.g. clustering, deduplication, dimensionality reduction). "
                f"{move_hint}"
            )
            if self.allow_non_partitionable:
                self.logger.warning(msg)
            else:
                raise ValueError(
                    f"{msg} If you understand the risk, set "
                    f"allow_non_partitionable=True to override."
                )
        else:  # MAYBE
            self.logger.warning(
                f"The following link(s) may not be partitionable depending on "
                f"their configuration: {bad_links}. If they perform cross-row "
                f"operations (e.g. ranking, aggregation), partitioning will "
                f"produce incorrect results. {move_hint}"
            )

    def _partition(self, dataframe):
        if self.partition_size is not None:
            self.logger.debug(
                f"Partition dataframe into partitions with approximate size {self.partition_size}"
            )
            partitions = self._partition_by_size(dataframe)
            self.logger.debug(f"Made {len(partitions)} partitions")
        elif self.num_partitions is not None:
            self.logger.debug(
                f"Partition dataframe into {self.num_partitions} partitions"
            )
            partitions = self._partition_by_num(dataframe)
            self.logger.debug(f"First partition is {len(partitions[0])} rows")
        else:
            raise ValueError("Specify either 'partition_size' or 'num_partitions'.")
        return partitions

    def _partition_by_size(self, dataframe):
        return np.array_split(dataframe, len(dataframe) // self.partition_size + 1)

    def _partition_by_num(self, dataframe):
        return np.array_split(dataframe, self.num_partitions)


class SafePoolLinkMapper:
    def __init__(self, link: Link):
        self.config = link.get_params()  # can safely and efficiently transport the config to threads/processes as text only

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - [Process %(process)d] - %(name)s: %(message)s"
        )
        link = Link.from_params(
            self.config
        )  # Link recreated from config in the thread/process
        process_id = os.getpid()
        link.logger.debug(f"Starting process {process_id}")
        return link(df)


@dataclass
class ParallelPartitionProcessor(PartitionProcessorBase):
    """Process dataframe in parallel on several processors

    Partitions the dataframe into chunks and uses Pool.map to process them in parallel and concatenate the result"""

    link: Link
    num_processes: int = psutil.cpu_count(logical=False)

    def _apply(self, dataframe):
        partitions = self._partition(dataframe)

        safemapper = SafePoolLinkMapper(self.link)

        # Initialize a multiprocessing pool
        with Pool(self.num_processes) as pool:
            # Process each chunk using the multiprocessing pool
            self.logger.debug(
                f"Will process partitions in {self.num_processes} parallel processes"
            )
            partitions = pool.map(safemapper.apply, partitions)

        # Process each partition using the supplied function and concatenate the processed partitions into the final DataFrame
        self.logger.debug("Joining processed partition to one dataframe")
        result_dataframe = pd.concat(partitions, ignore_index=True)

        return result_dataframe


@dataclass
class SerialPartitionProcessor(PartitionProcessorBase):
    """Processes smaller chunks of the dataframe serially

    After partition the dataframe chunks are processed one after another and the output concatenated
    Can save intermediate memory needs"""

    link: Link

    def _apply(self, dataframe):
        partitions = self._partition(dataframe)
        # Process each partition using the supplied Link or Chain and concatenate the processed partitions into the final DataFrame
        self.logger.debug("Processing partitions one by one")
        # TODO make a loop that pops from partitions, and append to processed partition list. Can maybe save memory, and allow for better logging
        result_dataframe = pd.concat(
            [self.link(partition) for partition in partitions], ignore_index=True
        )
        self.logger.debug("Done processing partitions, join to a single dataframe")
        return result_dataframe


# The worker body, as a module-level template rather than a separate installed script: the
# submit directory is then self-contained and inspectable, and a site can edit it without
# reinstalling pdchemchain. `$SLURM_ARRAY_TASK_ID` picks the shard.
_WORKER_PY = '''"""Worker for one SLURM array task -- written by SlurmPartitionProcessor."""
import os
import sys

from pdchemchain.base import Link
from pdchemchain.links.io import enable_mol_property_pickling

# MUST happen before any molecule is unpickled, or the Mol properties (Glide SD tags and
# friends) come back silently missing. It is a global RDKit setting, so every process in
# the array needs it, not just the submitter.
enable_mol_property_pickling()

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
work = os.path.dirname(os.path.abspath(__file__))
in_pkl = os.path.join(work, f"shard_{task_id:05d}.in.pkl")
out_pkl = os.path.join(work, f"shard_{task_id:05d}.out.pkl")

chain = Link.from_config_file(os.path.join(work, "chain.yaml"))
print(f"task {task_id}: {in_pkl} -> {out_pkl}", flush=True)

import pandas as pd

df = pd.read_pickle(in_pkl)
print(f"task {task_id}: {len(df)} rows in", flush=True)
out = chain(df)
out.to_pickle(out_pkl)
print(f"task {task_id}: {len(out)} rows out, wrote {out_pkl}", flush=True)
'''

_SUBMIT_SH = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=0-{last_index}{throttle}
#SBATCH --output={work}/slurm-%A_%a.out
#SBATCH --error={work}/slurm-%A_%a.out
{directives}
set -euo pipefail
{preamble}
{python} {work}/worker.py
"""


@dataclass
class SlurmPartitionProcessor(PartitionProcessorBase):
    """Process a dataframe by partitioning it across a SLURM array job.

    The same idea as :class:`ParallelPartitionProcessor`, but the partitions cross a
    MACHINE boundary rather than a process boundary. That changes three things and nothing
    else:

    1. Partitions travel as pickles in ``work_dir`` rather than through ``Pool.map``.
       Pickle is the only format that round-trips a frame faithfully -- several Mol
       columns, 3D conformers, molecule properties, fingerprints (see
       :class:`~pdchemchain.links.io.ToPickle`).
    2. The inner link travels as its own config YAML, using the same serialisation
       ``SafePoolLinkMapper`` already relies on, so nothing of the link itself is pickled.
    3. Each array task runs a small generated ``worker.py`` that reads its shard, applies
       the chain and writes its output shard.

    The call BLOCKS until the array finishes, because a Link must return a dataframe.
    Expect a docking array to block for hours; run it from a script rather than an
    interactive session.

    FAILURE POLICY (MVP): if any task fails, the whole call raises and ``work_dir`` is left
    in place for inspection. For a 10k redock that is what you want -- a missing shard is a
    hole in the deliverable. A future version should allow a bounded retry (SLURM can
    requeue individual array indices) and a partial-result mode, which is the sane policy
    for a 10M-compound VLS where a few lost shards do not matter.

    Parameters
    ----------
    link
        The inner Link or Chain to apply to each partition.
    work_dir
        Directory for shards, the chain config, the worker and the SLURM logs. Must be on
        a filesystem the compute nodes can see. Kept on failure; see ``cleanup``.
    python_executable
        Interpreter the array tasks run. Defaults to the submitting interpreter, which is
        usually right and is wrong in exactly one interesting case: when the submitter is
        not on a shared filesystem.
    sbatch_directives
        Extra ``#SBATCH`` lines, without the prefix, e.g.
        ``["--time=04:00:00", "--mem=16G", "--partition=cpu"]``. A template beats
        individual parameters here because site-specific SLURM quirks are where cluster
        jobs actually lose time.
    preamble
        Shell lines to run before the worker, e.g. ``conda activate`` or
        ``module load schrodinger``. The array task starts in a fresh shell, so anything
        the chain needs on PATH has to be set up here.
    max_concurrent
        Cap on simultaneously running tasks (``--array=0-N%max``). Use it to stay inside a
        licence limit -- Glide tokens, for instance.
    poll_interval
        Seconds between ``squeue`` checks.
    timeout
        Give up waiting after this many seconds. None waits indefinitely.
    cleanup
        Remove ``work_dir`` on success. Always kept on failure.

    Examples
    --------
    >>> dock = LigPrep(njobs=4) + GlideDock(grid_file=grid, njobs=4) + AggregateByScore()
    >>> runner = SlurmPartitionProcessor(
    ...     link=dock,
    ...     num_partitions=40,
    ...     work_dir="redock_work",
    ...     sbatch_directives=["--time=08:00:00", "--mem=16G", "--cpus-per-task=4"],
    ...     preamble="source ~/miniforge3/etc/profile.d/conda.sh\\nconda activate chem",
    ... )
    >>> result = runner(df)        # blocks until the whole array is done
    """

    link: Link
    work_dir: str = "slurm_partitions"
    python_executable: Optional[str] = None
    sbatch_directives: List[str] = field(default_factory=list)
    preamble: str = ""
    max_concurrent: Optional[int] = None
    poll_interval: int = 30
    timeout: Optional[int] = None
    cleanup: bool = False
    job_name: str = "pdchemchain"

    def _write_inputs(self, partitions: List[pd.DataFrame], work: Path) -> None:
        from pdchemchain.links.io import enable_mol_property_pickling

        enable_mol_property_pickling()
        for i, part in enumerate(partitions):
            part.to_pickle(work / f"shard_{i:05d}.in.pkl")
        self.link.to_config_file(str(work / "chain.yaml"))
        (work / "worker.py").write_text(_WORKER_PY)
        self.logger.info(
            f"staged {len(partitions)} shards, chain.yaml and worker.py in {work}"
        )

    def _write_submit(self, n: int, work: Path) -> Path:
        script = work / "submit.sh"
        script.write_text(
            _SUBMIT_SH.format(
                job_name=self.job_name,
                last_index=n - 1,
                throttle=f"%{self.max_concurrent}" if self.max_concurrent else "",
                work=work,
                directives="\n".join(f"#SBATCH {d}" for d in self.sbatch_directives),
                preamble=self.preamble,
                python=self.python_executable or _default_python(),
            )
        )
        script.chmod(0o755)
        return script

    def _submit(self, script: Path) -> str:
        proc = subprocess.run(
            ["sbatch", "--parsable", str(script)], capture_output=True, text=True
        )
        if proc.returncode != 0:
            # The sbatch BINARY existing proves nothing: it is installed on machines with
            # no reachable controller (a workstation with the client packages), where it
            # fails here rather than at the which() check. Surface sbatch's own words --
            # "Unexpected message received" means no controller, and that is a different
            # problem from a rejected job.
            raise RuntimeError(
                f"sbatch failed (exit {proc.returncode}): "
                f"{(proc.stderr or proc.stdout).strip()}\n"
                f"If this reads like the controller is unreachable, you are not on a "
                f"submit host -- run from a login node, or use "
                f"ParallelPartitionProcessor locally. Script kept at {script}."
            )
        job_id = proc.stdout.strip().split(";")[0]  # --parsable may append ";cluster"
        self.logger.info(f"submitted array job {job_id} ({script})")
        return job_id

    def _wait(self, job_id: str) -> None:
        """Block until no task of the array is pending or running."""
        t0 = time.time()
        while True:
            # squeue only lists live tasks, so an empty answer means the array is done --
            # it says nothing about whether the tasks SUCCEEDED, which is checked after.
            live = subprocess.run(
                ["squeue", "-h", "-j", job_id, "-o", "%T"],
                capture_output=True, text=True,
            )
            states = [s for s in live.stdout.split() if s]
            if not states:
                self.logger.info(f"array {job_id} no longer in the queue")
                return
            if self.timeout is not None and time.time() - t0 > self.timeout:
                raise TimeoutError(
                    f"array {job_id} still has {len(states)} task(s) "
                    f"{sorted(set(states))} after {self.timeout}s; work kept in "
                    f"{self.work_dir}"
                )
            self.logger.debug(
                f"array {job_id}: {len(states)} task(s) live {sorted(set(states))}, "
                f"waited {int(time.time()-t0)}s"
            )
            time.sleep(self.poll_interval)

    def _collect(self, n: int, work: Path, job_id: str) -> List[pd.DataFrame]:
        from pdchemchain.links.io import enable_mol_property_pickling

        enable_mol_property_pickling()
        missing, out = [], []
        for i in range(n):
            f = work / f"shard_{i:05d}.out.pkl"
            if not f.exists():
                missing.append(i)
                continue
            out.append(pd.read_pickle(f))
        if missing:
            # MVP policy: all or nothing. The shard logs are the only way to find out why,
            # so point at them rather than guessing.
            raise RuntimeError(
                f"{len(missing)} of {n} shards produced no output: "
                f"{missing[:10]}{' ...' if len(missing) > 10 else ''}. "
                f"Nothing was collected. Inspect {work}/slurm-{job_id}_<task>.out, and "
                f"`sacct -j {job_id} --format=JobID,State,ExitCode,MaxRSS`."
            )
        return out

    def _apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if shutil.which("sbatch") is None:
            raise RuntimeError(
                "sbatch not found: SlurmPartitionProcessor must run where SLURM can be "
                "submitted to (a login node), not on a workstation. Use "
                "ParallelPartitionProcessor locally."
            )
        work = Path(self.work_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)

        partitions = self._partition(dataframe)
        n = len(partitions)
        self._write_inputs(partitions, work)
        job_id = self._submit(self._write_submit(n, work))
        self._wait(job_id)
        results = self._collect(n, work, job_id)

        result = pd.concat(results, ignore_index=True)
        if len(result) != len(dataframe):
            # Not fatal: an inner chain may legitimately aggregate or filter rows. Say so
            # loudly, because the usual cause is a chain that was not row-preserving when
            # the caller assumed it was.
            self.logger.warning(
                f"row count changed: {len(dataframe)} in, {len(result)} out. Expected if "
                f"the inner chain filters or aggregates; a bug if it should not."
            )
        if self.cleanup:
            shutil.rmtree(work)
            self.logger.info(f"removed {work}")
        else:
            self.logger.info(f"work kept in {work}")
        return result


def _default_python() -> str:
    import sys

    return sys.executable
