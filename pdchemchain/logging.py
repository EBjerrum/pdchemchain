import logging

# Configure the logging settings with a custom formatter
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s"
)


logger = logging.getLogger("pdchemchain.logging")
