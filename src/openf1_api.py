import json
import socket
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
    pass


class OpenF1Client:
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
        params = {key: value for key, value in params.items() if value is not None}
        query = urlencode(params, doseq=True)
        url = f"{self.base_url}/{endpoint.strip('/')}"
        if query:
            return f"{url}?{query}"
        return url

# AI generated: initial draft of this script was created with the help of AI assistance
# and then reviewed and refactored by Rui Chen. The final version was manually edited.

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
                raise OpenF1APIError(f"OpenF1 request failed with status {exc.code}") from exc
            except URLError as exc:
                raise OpenF1APIError(f"Could not reach OpenF1 API: {exc.reason}") from exc
            except (TimeoutError, socket.timeout) as exc:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise OpenF1APIError("OpenF1 request timed out") from exc

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise OpenF1APIError("OpenF1 returned invalid JSON") from exc

        if not isinstance(data, list):
            raise OpenF1APIError("OpenF1 response was not a list")

        return data

    def get_sessions(
        self,
        *,
        year=None,
        meeting_key=None,
        session_key=None,
        session_name=None,
        country_name=None,
    ):
        return self._get(
            "sessions",
            year=year,
            meeting_key=meeting_key,
            session_key=session_key,
            session_name=session_name,
            country_name=country_name,
        )

    def get_drivers(self, *, session_key=None, meeting_key=None, driver_number=None):
        return self._get(
            "drivers",
            session_key=session_key,
            meeting_key=meeting_key,
            driver_number=driver_number,
        )

    def get_laps(self, *, session_key, driver_number=None, lap_number=None):
        return self._get(
            "laps",
            session_key=session_key,
            driver_number=driver_number,
            lap_number=lap_number,
        )

    def get_session_result(self, *, session_key, driver_number=None):
        return self._get(
            "session_result",
            session_key=session_key,
            driver_number=driver_number,
        )

    def get_pit(self, *, session_key, driver_number=None):
        return self._get(
            "pit",
            session_key=session_key,
            driver_number=driver_number,
        )

    def get_qualifying_sessions(self, *, year=None, meeting_key=None, country_name=None):
        return self.get_sessions(
            year=year,
            meeting_key=meeting_key,
            country_name=country_name,
            session_name=QUALIFYING_SESSION_NAME,
        )
