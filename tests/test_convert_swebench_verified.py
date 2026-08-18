import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/convert_swebench_verified.py"


def load_converter():
    assert SCRIPT.exists()
    spec = importlib.util.spec_from_file_location("convert_swebench_verified", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_row(instance_id: str) -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "repo": "astropy/astropy",
        "version": "4.3",
        "base_commit": "abc",
        "problem_statement": "problem",
        "hints_text": "",
        "test_patch": "patch",
        "FAIL_TO_PASS": '["test_fail"]',
        "PASS_TO_PASS": "[]",
        "environment_setup_commit": "def",
        "image": "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
        "eval_script": "#!/bin/bash\nexit 0\n",
        "log_parser": "parse_log_astropy",
        "eval_type": "pass_and_fail",
    }


def test_convert_rows_keeps_v5_fields_and_uses_runtime_image_key() -> None:
    converter = load_converter()
    output = io.StringIO()
    row = make_row("astropy__astropy-12907")

    assert (
        converter.convert_rows(
            [row],
            {"astropy__astropy-12907": "registry/swebench:mutable-tag"},
            output,
        )
        == 1
    )

    record = json.loads(output.getvalue())
    info = record["metadata"]["remote_env_info"]
    assert info["image"] == "local/astropy__astropy-12907"
    assert info["dataset_image"] == row["image"]
    for field in converter.SWEBENCH_V5_FIELDS:
        assert field in info
    assert info["eval_script"] == row["eval_script"]
    assert info["log_parser"] == row["log_parser"]
    assert info["eval_type"] == row["eval_type"]
    assert info["workdir"] == "/testbed"


def test_convert_rows_fails_when_image_map_is_not_an_exact_id_match() -> None:
    converter = load_converter()
    row = make_row("astropy__astropy-12907")

    with pytest.raises(ValueError, match="missing_images"):
        converter.convert_rows([row], {}, io.StringIO())

    with pytest.raises(ValueError, match="extra_images"):
        converter.convert_rows(
            [row],
            {
                "astropy__astropy-12907": "registry/swebench:tag",
                "astropy__astropy-13033": "registry/swebench:tag",
            },
            io.StringIO(),
        )


def test_convert_rows_fails_with_old_swebench_schema() -> None:
    converter = load_converter()
    row = make_row("astropy__astropy-12907")
    del row["eval_script"]
    del row["log_parser"]
    del row["eval_type"]

    with pytest.raises(ValueError, match="eval_script.*log_parser.*eval_type"):
        converter.convert_rows(
            [row],
            {"astropy__astropy-12907": "registry/swebench:tag"},
            io.StringIO(),
        )


def test_load_image_map_rejects_invalid_json(tmp_path: Path) -> None:
    converter = load_converter()
    path = tmp_path / "arca-images.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        converter.load_image_map(path)
