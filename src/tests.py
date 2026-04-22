# AI generated: initial draft of this script was created with the help ofAI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.
import sys

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    from analyze import (
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

from openf1_api import OpenF1APIError, OpenF1Client

_client = OpenF1Client()


class SkipTest(Exception):
    """Used when a test cannot run in the current environment."""


def require_pandas():
    if pd is None:
        raise SkipTest(
            "pandas is not installed for this Python interpreter. "
            "Use the Anaconda Python or install project dependencies first."
        )


def require_analysis_dependencies():
    require_pandas()
    if ANALYSIS_IMPORT_ERROR is not None:
        raise SkipTest(
            "analysis dependencies are missing for this Python interpreter: "
            f"{ANALYSIS_IMPORT_ERROR}"
        )


def fetch_2023_race_sessions():
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


def test_get_sessions():
    """Check that the API returns session data for a season."""
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

    print(f"  PASSED - {len(sessions)} sessions retrieved for 2023")


def test_get_qualifying_sessions():
    """Check that the qualifying-session helper filters correctly."""
    print("Test 2: get_qualifying_sessions(year=2023)...")
    try:
        sessions = _client.get_qualifying_sessions(year=2023)
    except OpenF1APIError as exc:
        raise SkipTest(f"OpenF1 API is unavailable: {exc}") from exc

    assert isinstance(sessions, list), "Expected a list"
    assert len(sessions) > 0, "Should have qualifying sessions"
    assert all(s.get("session_name") == "Qualifying" for s in sessions), \
        "All sessions should be Qualifying"

    print(f"  PASSED - {len(sessions)} qualifying sessions found")


def test_get_session_result():
    """Check that session_result data can be fetched for one race session."""
    print("Test 3: get_session_result() for a 2023 race session...")
    race_sessions = fetch_2023_race_sessions()

    session_key = race_sessions[0]["session_key"]
    try:
        session_results = _client.get_session_result(session_key=session_key)
    except OpenF1APIError as exc:
        raise SkipTest(f"OpenF1 API is unavailable: {exc}") from exc

    assert isinstance(session_results, list), "Expected a list"
    assert len(session_results) > 0, "Session result data should not be empty"
    assert "driver_number" in session_results[0], "Missing 'driver_number'"
    assert "position" in session_results[0], "Missing 'position'"

    print(
        f"  PASSED - {len(session_results)} session result records "
        f"for session_key={session_key}"
    )


def test_get_drivers():
    """Check that driver information can be fetched for a session."""
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

    print(f"  PASSED - {len(drivers)} drivers found for session_key={session_key}")


def test_get_laps():
    """Check that lap data can be fetched for one driver in a session."""
    print("Test 5: get_laps() for driver 1 in a 2023 race session...")
    race_sessions = fetch_2023_race_sessions()

    session_key = race_sessions[0]["session_key"]
    try:
        laps = _client.get_laps(session_key=session_key, driver_number=1)
    except OpenF1APIError as exc:
        raise SkipTest(f"OpenF1 API is unavailable: {exc}") from exc

    assert isinstance(laps, list), "Expected a list"
    # Driver 1 may not appear in every session, so the main check is that the request works.
    print(f"  PASSED - {len(laps)} lap records for driver 1 in session_key={session_key}")


def test_compute_correlation():
    """Check correlation on a simple dataset with a perfect relationship."""
    require_analysis_dependencies()

    print("Test 6: compute_correlation() with synthetic data...")
    df = pd.DataFrame(
        {
            "qualifying_position": list(range(1, 21)),
            "race_position": list(range(1, 21)),
        }
    )
    corr = compute_correlation(df)
    assert abs(corr - 1.0) < 1e-6, f"Expected perfect correlation, got {corr}"
    print(f"  PASSED - correlation = {corr:.4f}")


def test_run_linear_regression():
    """Check linear regression on a simple dataset with a perfect relationship."""
    require_analysis_dependencies()

    print("Test 7: run_linear_regression() with synthetic data...")
    df = pd.DataFrame(
        {
            "qualifying_position": list(range(1, 21)),
            "race_position": list(range(1, 21)),
        }
    )
    results = run_linear_regression(df)
    assert abs(results["coefficient"] - 1.0) < 0.01, "Coefficient should be ~1.0"
    assert results["mae"] < 1.0, "MAE should be very low for perfect data"
    print(f"  PASSED - coefficient={results['coefficient']:.4f}, MAE={results['mae']:.4f}")


def test_pole_to_win_rate():
    """Check pole-to-win rate on a small example with a known answer."""
    require_analysis_dependencies()

    print("Test 8: pole_to_win_rate()...")
    df = pd.DataFrame(
        {
            "qualifying_position": [1, 1, 1, 1, 2],
            "race_position": [1, 1, 2, 3, 1],
        }
    )
    rate = pole_to_win_rate(df)
    assert abs(rate - 0.5) < 1e-6, f"Expected 0.5, got {rate}"
    print(f"  PASSED - pole-to-win rate = {rate:.2%}")


if __name__ == "__main__":
    tests = [
        test_get_sessions,
        test_get_qualifying_sessions,
        test_get_session_result,
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
            print(f"  SKIPPED - {test.__name__}: {e}")
            skipped += 1
        except Exception as e:
            print(f"  FAILED - {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(
        f"Results: {passed} passed, {failed} failed, {skipped} skipped "
        f"out of {len(tests)} tests"
    )
    if failed > 0:
        sys.exit(1)
