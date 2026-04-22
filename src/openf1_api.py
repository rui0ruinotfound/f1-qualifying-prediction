import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from config import (
    MAX_RETRIES,
    OPENF1_BASE_URL,
    QUALIFYING_SESSION_NAME,
    REQUEST_PAUSE_SECONDS,
    REQUEST_TIMEOUT,
    RETRY_DELAY_SECONDS,
)


class OpenF1APIError(RuntimeError):
    """Raised when an OpenF1 request fails."""

# AI generated: initial draft of this script was created with the help ofAI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

class OpenF1Client:
    """Small helper class for reading data from the OpenF1 API."""

    def __init__(
        self,
        base_url=OPENF1_BASE_URL,
        timeout=REQUEST_TIMEOUT,
        pause_seconds=REQUEST_PAUSE_SECONDS,
        max_retries=MAX_RETRIES,
        retry_delay=RETRY_DELAY_SECONDS,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.pause_seconds = pause_seconds
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _build_url(self, endpoint, **params):
        clean_endpoint = endpoint.strip("/")
        query_params = {
            key: self._stringify_value(value)
            for key, value in params.items()
            if value is not None
        }
        query_string = urlencode(query_params, doseq=True)
        url = f"{self.base_url}/{clean_endpoint}"
        return f"{url}?{query_string}" if query_string else url

    @staticmethod
    def _stringify_value(value):
        if isinstance(value, bool):
            return str(value).lower()
        return value

    def _get(self, endpoint, **params):
        url = self._build_url(endpoint, **params)
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(url, timeout=self.timeout) as response:
                    payload = response.read().decode("utf-8")
                time.sleep(self.pause_seconds)
                break
            except HTTPError as exc:
                if exc.code == 429 and attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise OpenF1APIError(
                    f"OpenF1 request failed with status {exc.code}: {url}"
                ) from exc
            except URLError as exc:
                raise OpenF1APIError(
                    f"Could not reach OpenF1 API: {exc.reason}"
                ) from exc

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise OpenF1APIError("OpenF1 returned invalid JSON.") from exc

        if not isinstance(data, list):
            raise OpenF1APIError(
                f"Expected a list response from OpenF1, got {type(data).__name__}."
            )
        return data

    def get(self, endpoint, **params):
        """Fetch any OpenF1 endpoint by path."""
        return self._get(endpoint, **params)

    def get_meetings(self, **params):
        return self._get("meetings", **params)

    def get_sessions(
        self,
        *,
        year=None,
        meeting_key=None,
        session_key=None,
        session_name=None,
        country_name=None,
        **params,
    ):
        return self._get(
            "sessions",
            year=year,
            meeting_key=meeting_key,
            session_key=session_key,
            session_name=session_name,
            country_name=country_name,
            **params,
        )

    def get_drivers(
        self,
        *,
        session_key=None,
        meeting_key=None,
        driver_number=None,
        team_name=None,
        **params,
    ):
        return self._get(
            "drivers",
            session_key=session_key,
            meeting_key=meeting_key,
            driver_number=driver_number,
            team_name=team_name,
            **params,
        )

    def get_laps(
        self,
        *,
        session_key,
        driver_number=None,
        lap_number=None,
        **params,
    ):
        return self._get(
            "laps",
            session_key=session_key,
            driver_number=driver_number,
            lap_number=lap_number,
            **params,
        )

    def get_position(
        self,
        *,
        session_key,
        driver_number=None,
        **params,
    ):
        return self._get(
            "positions",
            session_key=session_key,
            driver_number=driver_number,
            **params,
        )

    def get_session_result(
        self,
        *,
        session_key,
        driver_number=None,
        **params,
    ):
        return self._get(
            "session_result",
            session_key=session_key,
            driver_number=driver_number,
            **params,
        )

    def get_car_data(
        self,
        *,
        session_key,
        driver_number=None,
        **params,
    ):
        return self._get(
            "car_data",
            session_key=session_key,
            driver_number=driver_number,
            **params,
        )

    def get_pit(
        self,
        *,
        session_key,
        driver_number=None,
        **params,
    ):
        return self._get(
            "pit",
            session_key=session_key,
            driver_number=driver_number,
            **params,
        )

    def get_weather(self, *, session_key, **params):
        return self._get("weather", session_key=session_key, **params)

    def get_qualifying_sessions(
        self,
        *,
        year=None,
        meeting_key=None,
        country_name=None,
        **params,
    ):
        """Return only sessions labeled as qualifying sessions."""
        return self.get_sessions(
            year=year,
            meeting_key=meeting_key,
            country_name=country_name,
            session_name=QUALIFYING_SESSION_NAME,
            **params,
        )
