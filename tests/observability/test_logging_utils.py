import logging
import sys
import types

sys.modules.setdefault("wandb", types.ModuleType("wandb"))

from slime.observability import logging_utils


def test_configure_logger_disables_arca_propagation(monkeypatch):
    monkeypatch.setattr(logging_utils, "_LOGGER_CONFIGURED", False)
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)

    arca_logger = logging.getLogger("arca")
    monkeypatch.setattr(arca_logger, "propagate", True)

    logging_utils.configure_logger()

    assert arca_logger.propagate is False
