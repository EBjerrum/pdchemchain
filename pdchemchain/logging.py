import logging
import inspect
import pandas as pd

# Configure the logging settings with a custom formatter
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s"
)


logger = logging.getLogger("pdchemchain.logging")


class RowLogger:
    """Logger for pandas.Series named 'row' in the call stack or supplied as argument.
    the various log methods will add a column '__log__' to the row containing the log message,
    if the log level is effective for the parent objects logger"""

    def __init__(self, parent):
        self.parent = parent

    def _log(self, row: pd.Series, message: str, level: str) -> pd.Series:
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        current_level = self.parent.logger.getEffectiveLevel()
        message_level = log_levels[level.upper()]

        if message_level >= current_level:
            log_column = "__log__"
            return self.parent._create_or_append(
                row, f"{level}: {message}", column_name=log_column
            )
        return row

    def debug(self, message: str, row: pd.Series = None):
        if row is None:
            row = self._get_outer_row()
        return self._log(row, message, "DEBUG")

    def info(self, message: str, row: pd.Series = None):
        if row is None:
            row = self._get_outer_row()
        return self._log(row, message, "INFO")

    def warning(self, message: str, row: pd.Series = None):
        if row is None:
            row = self._get_outer_row()
        return self._log(row, message, "WARNING")

    def error(self, message: str, row: pd.Series = None):
        if row is None:
            row = self._get_outer_row()
        return self._log(row, message, "ERROR")

    def critical(self, message: str, row: pd.Series = None):
        if row is None:
            row = self._get_outer_row()
        return self._log(row, message, "CRITICAL")

    def _get_outer_row(self) -> pd.Series:
        current_frame = inspect.currentframe()

        def _get_row_recursive(frame):
            """Get first 'row' in the call stack that is not None"""
            if frame.f_locals.get("row", None) is not None:
                return frame.f_locals["row"]
            if frame.f_back is None:
                raise ValueError("Row not found in the call stack")
            return _get_row_recursive(frame.f_back)

        return _get_row_recursive(current_frame.f_back)
