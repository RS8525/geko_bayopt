from __future__ import annotations

from geko_bayesopt.config import ExperimentConfig


def _config(**overrides) -> ExperimentConfig:
    data = {
        "experiment_id": "baseline-test",
        "case": {
            "kind": "periodic_hills",
            "options": {
                "hill_height": 1.0,
                "re_h": 5600,
                "dns_path": "dns.dat",
            },
        },
        "parameters": [
            {"name": "geko_csep", "low": 0.5, "high": 2.5},
        ],
        "objective": {"kind": "mse_cp"},
        "optimizer": {"kind": "skopt_gp"},
    }
    data.update(overrides)
    return ExperimentConfig.model_validate(data)


def test_baseline_and_ascii_retention_defaults() -> None:
    cfg = _config()

    assert cfg.evaluate_default_first is True
    assert cfg.keep_only_best_case_files is True
    assert cfg.keep_all_ascii_files is False


def test_ascii_retention_defaults_to_negated_case_retention() -> None:
    cfg = _config(keep_only_best_case_files=False)

    assert cfg.keep_all_ascii_files is True


def test_explicit_ascii_retention_overrides_derived_default() -> None:
    cfg = _config(
        keep_only_best_case_files=False,
        keep_all_ascii_files=False,
        evaluate_default_first=False,
    )

    assert cfg.keep_all_ascii_files is False
    assert cfg.evaluate_default_first is False
