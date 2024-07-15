import logging
import coloredlogs


def config_logger(logger):
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s %(name)s\t: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    coloredlogs.DEFAULT_FIELD_STYLES["levelname"] = {"color": "white"}
    coloredlogs.install(logger=logger)
