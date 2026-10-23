"""Regression guard: the reward must not be gameable."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("reward_audit", Path(__file__).parents[1] / "scripts" / "reward_audit.py")
audit_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_mod)

RESULTS = audit_mod.audit(n_tasks=100)


def test_degenerate_strategies_earn_nothing():
    for name in audit_mod.DEGENERATE:
        for difficulty, (reward, pass_rate) in RESULTS[name].items():
            assert reward < 0.02 and pass_rate == 0, (name, difficulty, reward)


def test_every_shortcut_fails_hard_tasks_often():
    for name in audit_mod.SHORTCUT:
        reward, pass_rate = RESULTS[name]["hard"]
        assert pass_rate <= 0.65, (name, pass_rate)
