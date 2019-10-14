import os

from bfgn.utils import logging


def test_get_bfgn_logger_logs_out(tmp_path) -> None:
    path_log = os.path.join(tmp_path, "log.out")
    logger = logging.get_bfgn_logger("", "INFO", path_log)
    logger.debug("test debug")
    logger.info("test info")
    logger.warning("test warning")
    logger.error("test error")
    with open(path_log) as file_:
        lines = file_.readlines()
    assert not any(["test debug" in line for line in lines])
    for msg in ("test info", "test warning", "test error"):
        assert any([msg in line for line in lines])
