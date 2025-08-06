import logging
import subprocess
import threading


def _stream_subprocess_output(pipe, log_func):
    """Reads lines from a subprocess pipe (stdout or stderr) and calls log_func(line) in real time."""
    for line in iter(pipe.readline, ""):
        if line:
            log_func(line.rstrip("\n"))
    pipe.close()


def run_subprocess_in_real_time(cmd, logger):
    """Starts a subprocess with real-time capture of stdout and stderr.
    Logs each line immediately via the provided logger.

    Returns:
      The subprocess's return code (0 if success, non-zero if an error occurred).

    """
    logger.info("Running command: %s", " ".join(cmd))

    # Create the Popen object with pipes for stdout/stderr
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Threads to consume stdout/stderr in real time
    stdout_thread = threading.Thread(
        target=_stream_subprocess_output, args=(process.stdout, logger.info)
    )
    stderr_thread = threading.Thread(
        target=_stream_subprocess_output, args=(process.stderr, logger.error)
    )

    # Start the threads
    stdout_thread.start()
    stderr_thread.start()

    # Wait for the process to finish
    process.wait()
    # Wait for threads to finish reading
    stdout_thread.join()
    stderr_thread.join()

    return process.returncode


class InfoOnlyFilter(logging.Filter):
    """A filter to allow only messages containing '[INFO]' in them."""

    def filter(self, record):
        return "[INFO]" in record.getMessage()


def setup_logger(run_name: str, log_all="logs/all_logs.log", log_info="logs/info_logs.log"):
    """Sets up a logger that writes:
    - All logs to `log_all`
    - Only logs explicitly containing `[INFO]` to `log_info`
    - Logs to console.
    """
    logger_name = f"STaSC_logger_{run_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if already set
    if not logger.handlers:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format))

        # File handler for ALL logs
        file_handler_all = logging.FileHandler(log_all, mode="a", encoding="utf-8")
        file_handler_all.setLevel(logging.INFO)
        file_handler_all.setFormatter(logging.Formatter(log_format))

        # File handler for INFO logs with "[INFO]"
        file_handler_info = logging.FileHandler(log_info, mode="a", encoding="utf-8")
        file_handler_info.setLevel(logging.INFO)
        file_handler_info.setFormatter(logging.Formatter(log_format))
        file_handler_info.addFilter(InfoOnlyFilter())  # Only log messages containing "[INFO]"

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler_all)
        logger.addHandler(file_handler_info)

    return logger
