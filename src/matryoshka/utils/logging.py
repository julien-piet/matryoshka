import logging
import sys

# Define ANSI escape sequences for coloring
RESET = "\033[0m"
COLORS = {
    "DEBUG": "\033[94m",  # Blue
    "INFO": "\033[92m",  # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",  # Red
    "CRITICAL": "\033[95m",  # Magenta
}


# Custom formatter to add color to log levels
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        log_color = COLORS.get(record.levelname, RESET)
        record.levelname = f"{log_color}{record.levelname}{RESET}"
        return super().format(record)


def get_logger():
    return logging.getLogger("logparser")


def setup_logger(level=logging.DEBUG):
    # Create a logger (you can use the root logger by just calling logging.getLogger())
    logger = get_logger()
    logger.setLevel(
        level
    )  # Set the logging level to DEBUG for the root logger

    # Create a stream handler to send log output to sys.stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(
        level
    )  # Set the handler logging level to DEBUG

    # Create a formatter and set it for the handler
    formatter = ColoredFormatter("%(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(stream_handler)
    return logger
