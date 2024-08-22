import logging
import os
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pandas as pd
import psutil

from pdchemchain.base import Link


@dataclass(kw_only=True)
class PartitionProcessorBase(Link):
    """Base class for partition processors"""

    partition_size: Optional[int] = field(default=None)
    num_partitions: Optional[int] = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if (self.partition_size is not None) and (self.num_partitions is not None):
            raise ValueError(
                "Specify either 'partition_size' or 'num_partitions', not both."
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
