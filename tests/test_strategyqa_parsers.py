"""Contract tests for the StrategyQA yes/no parsers in the runner.

Loads the runner with ``AJAR_DATASET_KIND=strategyqa`` and verifies:
  * ``parse_gsm8k_answer`` returns "yes" / "no" instead of a number
  * ``extract_predicted_number`` (rebound to extract_predicted_yesno)
    extracts yes/no from common model-output shapes
  * Numeric mode still works under the default kind
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "run_qwen3_gsm8k_mi.py"


def _import_runner(kind: str, name: str):
    os.environ["AJAR_DATASET_KIND"] = kind
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def runner_strategyqa():
    return _import_runner("strategyqa", "ajar_runner_strategyqa")


@pytest.fixture(scope="module")
def runner_gsm8k():
    return _import_runner("gsm8k", "ajar_runner_gsm8k")


@pytest.mark.parametrize("answer,expected", [
    ("Some rationale\n#### yes", "yes"),
    ("Some rationale\n#### no", "no"),
    ("Some rationale\n#### Yes.", "yes"),
    ("only the verdict yes", "yes"),
    ("the answer is no, definitely.", "no"),
])
def test_strategyqa_parse_gold(runner_strategyqa, answer, expected):
    assert runner_strategyqa.parse_gsm8k_answer(answer) == expected


@pytest.mark.parametrize("text,expected", [
    ("The answer is yes.", "yes"),
    ("Final answer: no", "no"),
    ("Final Answer = Yes.", "yes"),
    ("Reasoning... \\boxed{yes}", "yes"),
    ("Reasoning... \\boxed{no}", "no"),
    # Binary mapping seen under explicit_no_cot prompts that demand a
    # numeric \boxed{} answer. 0=no, 1=yes (verified consistent across
    # 73 StrategyQA explicit_no_cot outputs in the 2026-05-06 run).
    ("\\boxed{0}", "no"),
    ("\\boxed{1}", "yes"),
    ("Reasoning. \\boxed{1}", "yes"),
    (
        "<think>let me think no maybe yes</think>\nThe answer is yes.",
        "yes",
    ),
    ("there are reasons it could be yes or no, the model says no.", "no"),
])
def test_strategyqa_extract_prediction(runner_strategyqa, text, expected):
    pred, _ = runner_strategyqa.extract_predicted_number(text)
    assert pred == expected


def test_strategyqa_boxed_two_is_unparsed(runner_strategyqa):
    """`\\boxed{2}` (rare model error) should NOT be coerced to yes/no.
    Only \\boxed{0}/\\boxed{1} are interpreted under the binary mapping."""
    pred, boxed = runner_strategyqa.extract_predicted_number("\\boxed{2}")
    assert pred is None
    assert boxed == "2"


def test_gsm8k_mode_unaffected(runner_gsm8k):
    pred, _ = runner_gsm8k.extract_predicted_number("The answer is 42.")
    assert pred == "42"
    assert runner_gsm8k.parse_gsm8k_answer("Reasoning\n#### 18") == "18"
