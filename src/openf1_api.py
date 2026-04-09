"""
Small client for the OpenF1 REST API.

This module provides a lightweight interface for accessing OpenF1 data.
All requests are handled through a shared helper function, and common
endpoints are wrapped for convenience.

Example:
    from src.openf1_api import OpenF1Client

    client = OpenF1Client()
    sessions = client.get_sessions(year=2024, country_name="Japan")
    laps = client.get_laps(session_key=9158, driver_number=1)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


JsonList = list[dict[str, Any]]


class OpenF1APIError(RuntimeError):
    """Raised when an OpenF1 request fails."""


@dataclass(slots=True)
class OpenF1Client:
    """Minimal client for the OpenF1 API."""

    base_url: str = "https://api.openf1.org/v1"
    timeout: float = 30.0

    def _build_url(self, endpoint: str, **params: Any) -> str:
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
    def _stringify_value(value: Any) -> Any:
        if isinstance(value, bool):
            return str(value).lower()
        return value

    def _get(self, endpoint: str, **params: Any) -> JsonList:
        url = self._build_url(endpoint, **params)
        try:
            with urlopen(url, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            raise OpenF1APIError(
                f"OpenF1 request failed with status {exc.code}: {url}"
            ) from exc
        except URLError as exc:
            raise OpenF1APIError(f"Could not reach OpenF1 API: {exc.reason}") from exc

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise OpenF1APIError("OpenF1 returned invalid JSON.") from exc

        if not isinstance(data, list):
            raise OpenF1APIError(
                f"Expected a list response from OpenF1, got {type(data).__name__}."
            )
        return data

    def get(self, endpoint: str, **params: Any) -> JsonList:
        """Fetch any OpenF1 endpoint by path."""

        return self._get(endpoint, **params)

    def get_meetings(self, **params: Any) -> JsonList:
        return self._get("meetings", **params)

    def get_sessions(
        self,
        *,
        year: int | None = None,
        meeting_key: int | None = None,
        session_key: int | None = None,
        session_name: str | None = None,
        country_name: str | None = None,
        **params: Any,
    ) -> JsonList:
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
        session_key: int | None = None,
        meeting_key: int | None = None,
        driver_number: int | None = None,
        team_name: str | None = None,
        **params: Any,
    ) -> JsonList:
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
        session_key: int,
        driver_number: int | None = None,
        lap_number: int | None = None,
        **params: Any,
    ) -> JsonList:
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
        session_key: int,
        driver_number: int | None = None,
        **params: Any,
    ) -> JsonList:
        return self._get(
            "position",
            session_key=session_key,
            driver_number=driver_number,
            **params,
        )

    def get_car_data(
        self,
        *,
        session_key: int,
        driver_number: int | None = None,
        **params: Any,
    ) -> JsonList:
        return self._get(
            "car_data",
            session_key=session_key,
            driver_number=driver_number,
            **params,
        )

    def get_pit(
        self,
        *,
        session_key: int,
        driver_number: int | None = None,
        **params: Any,
    ) -> JsonList:
        return self._get(
            "pit",
            session_key=session_key,
            driver_number=driver_number,
            **params,
        )

    def get_weather(self, *, session_key: int, **params: Any) -> JsonList:
        return self._get("weather", session_key=session_key, **params)

    def get_qualifying_sessions(
        self,
        *,
        year: int | None = None,
        meeting_key: int | None = None,
        country_name: str | None = None,
        **params: Any,
    ) -> JsonList:
        """Convenience helper for qualifying sessions only."""

        return self.get_sessions(
            year=year,
            meeting_key=meeting_key,
            country_name=country_name,
            session_name="Qualifying",
            **params,
        )
