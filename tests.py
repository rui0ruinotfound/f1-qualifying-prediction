"""
tests.py
Test suite for the F1 Qualifying Prediction project.
Tests API connectivity, data fetching, and analysis functions.

Run with: python tests.py
"""

import sys
from typing import Callable

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    from src.analysis import (
        compute_correlation,
        pole_to_win_rate,
        run_linear_regression,
    )
    ANALYSIS_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    compute_correlation = None
    pole_to_win_rate = None
    run_linear_regression = None
    ANALYSIS_IMPORT_ERROR = exc

from src.openf1_api import OpenF1APIError, OpenF1Client

# Shared client instance for all API tests
_client = OpenF1Client()


class SkipTest(Exception):
    """Raised when a test cannot run in the current environment."""


def require_pandas() -> None:
    if pd is None:
        raise SkipTest(
            "pandas is not installed for this Python interpreter. "
            "Use the Anaconda Python or install project dependencies first."
        )


def require_analysis_dependencies() -> None:
    require_pandas()
    if ANALYSIS_IMPORT_ERROR is not None:
        raise SkipTest(
            "analysis dependencies are missing for this Python interpreter: "
            f"{ANALYSIS_IMPORT_ERROR}"
        )


def fetch_2023_race_sessions() -> list[dict]:
    try:
        race_sessions = [
            session
            for session in _client.get_sessions(year=2023)
            if session.get("session_name") == "Race"
        ]
    except OpenF1APIError as exc:
        raise SkipTest(f"OpenF1 API is unavailable: {exc}") from exc

    if not race_sessions:
        raise SkipTest("No 2023 race sessions were returned by OpenF1.")
    return race_sessions


# ──────────────────────────────────────────────
# Test 1: OpenF1 API – fetch sessions for a season
# ──────────────────────────────────────────────
def test_get_sessions():
    """Test that we can retrieve session data from the OpenF1 API."""
    print("Test 1: get_sessions(year=2023)...")
    try:
        sessions = _client.get_sessions(year=2023)
    except OpenF1APIError as exc:
        raise SkipTest(f"OpenF1 API is unavailable: {exc}") from exc

    assert isinstance(sessions, list), "Expected a list"
    assert len(sessions) > 0, "Should have at least one session"
    assert "session_key" in sessions[0], "Missing 'session_key'"
    assert "session_name" in sessions[0], "Missing 'session_name'"
    assert "year" in sessions[0], "Missing 'year'"

    print(f"  PASSED – {len(sessions)} sessions retrieved for 2023")


# ──────────────────────────────────────────────
# Test 2: OpenF1 API – convenience qualifying helper
# ──────────────────────────────────────────────
def test_get_qualifying_sessions():
    """Test the get_qualifying_sessions convenience wrapper."""
    print("Test 2: get_qualifying_sessions(year=2023)...")
    try:
        sessions = _client.get_qualifying_sessions(year=2023)
    except OpenF1APIError as exc:
        raise SkipTest(f"OpenF1 API is unavailable: {exc}") from exc

    assert isinstance(sessions, list), "Expected a list"
    assert len(sessions) > 0, "Should have qualifying sessions"
    assert all(s.get("session_name") == "Qualifying" for s in sessions), \
        "All sessions should be Qualifying"

    print(f"  PASSED – {len(sessions)} qualifying sessions found")


# ──────────────────────────────────────────────
# Test 3: OpenF1 API – fetch position data for a race session
# ──────────────────────────────────────────────
def test_get_position():
    """Test that position data can be fetched for a known 2023 race session."""
    print("Test 3: get_position() for a 2023 race session...")
    race_sessions = fetch_2023_race_sessions()

    session_key = race_sessions[0]["session_key"]
    try:
        positions = _client.get_position(session_key=session_key)
    except OpenF1APIError as exc:
        raise SkipTest(f"OpenF1 API is unavailable: {exc}") from exc

    assert isinstance(positions, list), "Expected a list"
    assert len(positions) > 0, "Position data should not be empty"
    assert "driver_number" in positions[0], "Missing 'driver_number'"
    assert "position" in positions[0], "Missing 'position'"

    print(f"  PASSED – {len(positions)} position records for session_key={session_key}")


# ──────────────────────────────────────────────
# Test 4: OpenF1 API – fetch drivers for a session
# ──────────────────────────────────────────────
def test_get_drivers():
    """Test that driver info can be fetched for a session."""
    print("Test 4: get_drivers() for a 2023 race session...")
    race_sessions = fetch_2023_race_sessions()

    session_key = race_sessions[0]["session_key"]
    try:
        drivers = _client.get_drivers(session_key=session_key)
    except OpenF1APIError as exc:
        raise SkipTest(f"OpenF1 API is unavailable: {exc}") from exc

    assert isinstance(drivers, list), "Expected a list"
    assert len(drivers) > 0, "Drivers list should not be empty"
    assert "driver_number" in drivers[0], "Missing 'driver_number'"

    print(f"  PASSED – {len(drivers)} drivers found for session_key={session_key}")


# ──────────────────────────────────────────────
# Test 5: OpenF1 API – fetch laps for a single driver
# ──────────────────────────────────────────────
def test_get_laps():
    """Test lap data retrieval for a specific driver in a session."""
    print("Test 5: get_laps() for driver 1 in a 2023 race session...")
    race_sessions = fetch_2023_race_sessions()

    session_key = race_sessions[0]["session_key"]
    try:
        laps = _client.get_laps(session_key=session_key, driver_number=1)
    except OpenF1APIError as exc:
        raise SkipTest(f"OpenF1 API is unavailable: {exc}") from exc

    assert isinstance(laps, list), "Expected a list"
    # Driver 1 may not be in every race; just verify the call succeeds
    print(f"  PASSED – {len(laps)} lap records for driver 1 in session_key={session_key}")


# ──────────────────────────────────────────────
# Test 6: Analysis – correlation on synthetic data
# ──────────────────────────────────────────────
def test_compute_correlation():
    """Test correlation function with perfectly correlated synthetic data."""
    require_analysis_dependencies()

    print("Test 6: compute_correlation() with synthetic data...")
    df = pd.DataFrame({
        "qualifying_position": list(range(1, 21)),
        "race_position":       list(range(1, 21)),
    })
    corr = compute_correlation(df)
    assert abs(corr - 1.0) < 1e-6, f"Expected perfect correlation, got {corr}"
    print(f"  PASSED – correlation = {corr:.4f}")


# ──────────────────────────────────────────────
# Test 7: Analysis – linear regression on synthetic data
# ──────────────────────────────────────────────
def test_run_linear_regression():
    """Test linear regression with perfectly correlated synthetic data."""
    require_analysis_dependencies()

    print("Test 7: run_linear_regression() with synthetic data...")
    df = pd.DataFrame({
        "qualifying_position": list(range(1, 21)),
        "race_position":       list(range(1, 21)),
    })
    results = run_linear_regression(df)
    assert abs(results["coefficient"] - 1.0) < 0.01, "Coefficient should be ~1.0"
    assert results["mae"] < 1.0, "MAE should be very low for perfect data"
    print(f"  PASSED – coefficient={results['coefficient']:.4f}, MAE={results['mae']:.4f}")


# ──────────────────────────────────────────────
# Test 8: Analysis – pole-to-win rate
# ──────────────────────────────────────────────
def test_pole_to_win_rate():
    """Test pole-to-win calculation with known data."""
    require_analysis_dependencies()

    print("Test 8: pole_to_win_rate()...")
    df = pd.DataFrame({
        "qualifying_position": [1, 1, 1, 1, 2],
        "race_position":       [1, 1, 2, 3, 1],
    })
    rate = pole_to_win_rate(df)
    assert abs(rate - 0.5) < 1e-6, f"Expected 0.5, got {rate}"
    print(f"  PASSED – pole-to-win rate = {rate:.2%}")


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────
if __name__ == "__main__":
    tests: list[Callable[[], None]] = [
        test_get_sessions,
        test_get_qualifying_sessions,
        test_get_position,
        test_get_drivers,
        test_get_laps,
        test_compute_correlation,
        test_run_linear_regression,
        test_pole_to_win_rate,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            test()
            passed += 1
        except SkipTest as e:
            print(f"  SKIPPED – {test.__name__}: {e}")
            skipped += 1
        except Exception as e:
            print(f"  FAILED – {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(
        f"Results: {passed} passed, {failed} failed, {skipped} skipped "
        f"out of {len(tests)} tests"
    )
    if failed > 0:
        sys.exit(1)
