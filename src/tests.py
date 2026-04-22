import sys

import pandas as pd

from analyze import compute_correlation, pole_to_win_rate, run_linear_regression
from kaggle_loader import load_kaggle_dataset
from load import load_multiple_seasons
from openf1_api import OpenF1APIError
from process import finalize_results

# AI generated: initial draft of this script was created with the help ofAI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.


class SkipTest(Exception):
    pass


def test_openf1_load():
    print("Test 1: load_multiple_seasons([2023])...")
    try:
        df = load_multiple_seasons([2023])
    except OpenF1APIError as exc:
        raise SkipTest(f"OpenF1 API is unavailable: {exc}") from exc

    assert not df.empty
    assert "qualifying_position" in df.columns
    assert "race_position" in df.columns
    print(f"  PASSED - loaded {len(df)} rows")


def test_compute_correlation():
    print("Test 2: compute_correlation()...")
    df = pd.DataFrame(
        {
            "qualifying_position": list(range(1, 21)),
            "race_position": list(range(1, 21)),
        }
    )
    corr = compute_correlation(df)
    assert abs(corr - 1.0) < 1e-6
    print(f"  PASSED - correlation = {corr:.4f}")


def test_run_linear_regression():
    print("Test 3: run_linear_regression()...")
    df = pd.DataFrame(
        {
            "qualifying_position": list(range(1, 21)),
            "race_position": list(range(1, 21)),
        }
    )
    results = run_linear_regression(df)
    assert abs(results["coefficient"] - 1.0) < 0.01
    assert results["mae"] < 1.0
    print(f"  PASSED - coefficient = {results['coefficient']:.4f}")


def test_pole_to_win_rate():
    print("Test 4: pole_to_win_rate()...")
    df = pd.DataFrame(
        {
            "qualifying_position": [1, 1, 1, 1, 2],
            "race_position": [1, 1, 2, 3, 1],
        }
    )
    rate = pole_to_win_rate(df)
    assert abs(rate - 0.5) < 1e-6
    print(f"  PASSED - pole-to-win rate = {rate:.2%}")


def test_finalize_results():
    print("Test 5: finalize_results()...")
    df = pd.DataFrame(
        {
            "driver_number": ["1", "16"],
            "qualifying_position": ["1", "2"],
            "race_position": ["2", "1"],
            "year": [2024, 2024],
            "circuit": ["Monza", "Monza"],
            "meeting_key": ["test_1", "test_1"],
        }
    )
    cleaned = finalize_results(df)
    assert cleaned["driver_number"].dtype.kind in "iu"
    assert cleaned["qualifying_position"].dtype.kind in "iu"
    assert cleaned["race_position"].dtype.kind in "iu"
    print(f"  PASSED - cleaned {len(cleaned)} rows")


def test_kaggle_load():
    print("Test 6: load_kaggle_dataset([2023, 2024, 2025])...")
    df = load_kaggle_dataset([2023, 2024, 2025])
    assert not df.empty
    assert "qualifying_position" in df.columns
    assert "race_position" in df.columns
    print(f"  PASSED - loaded {len(df)} rows")


if __name__ == "__main__":
    tests = [
        test_openf1_load,
        test_compute_correlation,
        test_run_linear_regression,
        test_pole_to_win_rate,
        test_finalize_results,
        test_kaggle_load,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            test()
            passed += 1
        except SkipTest as exc:
            print(f"  SKIPPED - {test.__name__}: {exc}")
            skipped += 1
        except Exception as exc:
            print(f"  FAILED - {test.__name__}: {exc}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        sys.exit(1)
