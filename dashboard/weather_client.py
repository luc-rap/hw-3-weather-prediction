"""
Client for the National Weather Service API (api.weather.gov).
"""

import hashlib
import json
from datetime import datetime
from typing import Any

import requests


_BASE_URL = "https://api.weather.gov"
_DEFAULT_TIMEOUT = 30


class WeatherClient:
    """Client for the National Weather Service API."""

    def __init__(self, base_url: str | None = None, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        # NWS API requires a User-Agent header (no auth needed - it's public)
        self._session.headers.update(
            {
                "User-Agent": "(Databricks Weather App, student@databricks.com)",
                "Accept": "application/geo+json",
            }
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Make a GET request to the NWS API."""
        resp = self._session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def get_grid_point(self, lat: float, lon: float) -> dict:
        """
        Resolve a lat/lon to a NWS grid point.
        Returns: {
            "gridId": "OKX",
            "gridX": 32,
            "gridY": 35,
            "forecast": "https://api.weather.gov/gridpoints/OKX/32,35/forecast",
            "forecastOffice": "https://api.weather.gov/offices/OKX"
        }
        """
        data = self.get(f"/points/{lat},{lon}")
        properties = data.get("properties", {})
        return {
            "gridId": properties.get("gridId"),
            "gridX": properties.get("gridX"),
            "gridY": properties.get("gridY"),
            "forecast": properties.get("forecast"),
            "forecastOffice": properties.get("forecastOffice"),
        }

    def get_active_alerts(self, lat: float, lon: float, limit: int = 50, location_name: str | None = None) -> list[dict]:
        """
        Fetch active weather alerts for a specific point (lat/lon).
        Returns normalized alert documents.
        """
        # Note: NWS API doesn't support 'limit' parameter - we slice results instead
        data = self.get(f"/alerts/active", params={"point": f"{lat},{lon}"})
        features = data.get("features", [])
        
        # Use location_name if provided, otherwise fall back to lat/lon
        location_display = location_name or f"{lat},{lon}"
        
        alerts = []
        for feature in features[:limit]:
            props = feature.get("properties", {})
            
            # Create stable ID from alert id or hash of key fields
            alert_id = props.get("id") or self._generate_id(
                f"alert-{lat}-{lon}-{props.get('event')}-{props.get('sent')}"
            )
            
            alerts.append({
                "id": alert_id,
                "location": location_display,
                "latitude": lat,
                "longitude": lon,
                "source_type": "alert",
                "headline": props.get("headline", ""),
                "event": props.get("event", ""),
                "narrative_text": props.get("description", ""),
                "issued_at": props.get("sent"),
                "effective_at": props.get("effective"),
                "payload": json.dumps(feature),
                "synced_at": datetime.utcnow().isoformat(),
            })
        
        return alerts

    def get_state_alerts(self, state_code: str, location_name: str | None = None, limit: int = 50) -> list[dict]:
        """
        Fetch active weather alerts for an entire state (e.g., "FL", "TX").
        Returns normalized alert documents.
        
        This provides a much richer alert dataset than point-based queries.
        """
        # Note: NWS API doesn't support 'limit' parameter - we slice results instead
        data = self.get(f"/alerts/active", params={"area": state_code})
        features = data.get("features", [])
        
        # Use location_name if provided, otherwise use state code
        location_display = location_name or state_code
        
        alerts = []
        for feature in features[:limit]:
            props = feature.get("properties", {})
            
            # Extract coordinates from the alert geometry (use centroid if available)
            geometry = feature.get("geometry")
            lat, lon = None, None
            if geometry and geometry.get("type") == "Point":
                coords = geometry.get("coordinates", [])
                if len(coords) >= 2:
                    lon, lat = coords[0], coords[1]  # GeoJSON is [lon, lat]
            
            # Create stable ID from the alert's native ID
            alert_id = props.get("id") or self._generate_id(
                f"state-alert-{state_code}-{props.get('event')}-{props.get('sent')}"
            )
            
            alerts.append({
                "id": alert_id,
                "location": location_display,
                "latitude": lat,
                "longitude": lon,
                "source_type": "state_alert",
                "headline": props.get("headline", ""),
                "event": props.get("event", ""),
                "narrative_text": props.get("description", ""),
                "issued_at": props.get("sent"),
                "effective_at": props.get("effective"),
                "payload": json.dumps(feature),
                "synced_at": datetime.utcnow().isoformat(),
            })
        
        return alerts

    def get_forecast(self, lat: float, lon: float, location_name: str | None = None, limit: int = 14) -> list[dict]:
        """
        Fetch forecast for a location.
        Returns normalized forecast documents (one per period).
        
        Args:
            limit: Maximum number of forecast periods to return (default: 14 = 7 days)
        """
        # First get the grid point
        grid = self.get_grid_point(lat, lon)
        
        # Then get the forecast
        forecast_url = grid.get("forecast", "").replace(self.base_url, "")
        if not forecast_url:
            return []
        
        data = self.get(forecast_url)
        periods = data.get("properties", {}).get("periods", [])
        
        # Limit the number of forecast periods
        periods = periods[:limit]
        
        # Use location_name if provided, otherwise fall back to lat/lon
        location_display = location_name or f"{lat},{lon}"
        
        forecasts = []
        for period in periods:
            # Generate stable ID from location + period
            forecast_id = self._generate_id(
                f"forecast-{lat}-{lon}-{period.get('number')}-{period.get('startTime')}"
            )
            
            forecasts.append({
                "id": forecast_id,
                "location": location_display,
                "latitude": lat,
                "longitude": lon,
                "source_type": "forecast",
                "headline": period.get("name", ""),
                "event": f"Forecast: {period.get('name', '')}",
                "narrative_text": period.get("detailedForecast", ""),
                "issued_at": period.get("startTime"),
                "effective_at": period.get("startTime"),
                "payload": json.dumps(period),
                "synced_at": datetime.utcnow().isoformat(),
            })
        
        return forecasts

    def get_hourly_forecast(self, lat: float, lon: float, location_name: str | None = None, limit: int = 24) -> list[dict]:
        """
        Fetch hourly forecast for a location.
        Returns normalized hourly forecast documents (one per hour).
        
        Args:
            limit: Maximum number of hourly periods to return (default: 24 = 1 day)
        """
        # First get the grid point
        grid = self.get_grid_point(lat, lon)
        
        # Then get the hourly forecast
        forecast_url = grid.get("forecast", "").replace(self.base_url, "")
        if not forecast_url:
            return []
        
        # Replace /forecast with /forecast/hourly
        hourly_forecast_url = forecast_url.replace("/forecast", "/forecast/hourly")
        
        data = self.get(hourly_forecast_url)
        periods = data.get("properties", {}).get("periods", [])
        
        # Limit the number of hourly periods
        periods = periods[:limit]
        
        # Use location_name if provided, otherwise fall back to lat/lon
        location_display = location_name or f"{lat},{lon}"
        
        hourly_forecasts = []
        for period in periods:
            # Generate stable ID from location + period
            forecast_id = self._generate_id(
                f"hourly-forecast-{lat}-{lon}-{period.get('number')}-{period.get('startTime')}"
            )
            
            # Hourly forecasts often don't have detailedForecast - construct narrative from available fields
            detailed = period.get("detailedForecast", "")
            short = period.get("shortForecast", "")
            temp = period.get("temperature")
            temp_unit = period.get("temperatureUnit", "F")
            wind_speed = period.get("windSpeed", "")
            wind_dir = period.get("windDirection", "")
            
            # Build a narrative text: prefer detailed, fall back to constructed
            if detailed:
                narrative = detailed
            elif short:
                # Construct from short forecast and weather parameters
                parts = [short]
                if temp is not None:
                    parts.append(f"Temperature: {temp}°{temp_unit}")
                if wind_speed:
                    parts.append(f"Wind: {wind_speed} from {wind_dir}" if wind_dir else f"Wind: {wind_speed}")
                narrative = ". ".join(parts) + "."
            else:
                # Last resort: just temperature if available
                narrative = f"Temperature: {temp}°{temp_unit}" if temp is not None else "No forecast details available"
            
            # Extract time for headline (hourly forecasts don't have a 'name' field)
            start_time_str = period.get("startTime", "")
            headline = period.get("name", "")
            if not headline and start_time_str:
                try:
                    # Parse ISO timestamp and format as "Aug 6, 4:00 PM"
                    from dateutil import parser
                    dt = parser.parse(start_time_str)
                    headline = dt.strftime("%b %-d, %-I:%M %p")  # e.g., "Aug 6, 4:00 PM"
                except Exception:
                    headline = start_time_str  # Fallback to raw timestamp
            
            hourly_forecasts.append({
                "id": forecast_id,
                "location": location_display,
                "latitude": lat,
                "longitude": lon,
                "source_type": "hourly_forecast",
                "headline": headline,
                "event": f"Hourly Forecast: {headline}",
                "narrative_text": narrative,
                "issued_at": period.get("startTime"),
                "effective_at": period.get("startTime"),
                "payload": json.dumps(period),
                "synced_at": datetime.utcnow().isoformat(),
            })
        
        return hourly_forecasts

    def get_forecast_discussion(self, office: str, limit: int = 5) -> list[dict]:
        """
        Fetch Area Forecast Discussion (AFD) for a forecast office.
        Returns normalized discussion documents.
        """
        data = self.get(f"/products/types/AFD/locations/{office}")
        products = data.get("@graph", [])
        
        discussions = []
        for product in products[:limit]:
            product_id = product.get("id", "")
            if not product_id:
                continue
            
            # Fetch the full product text
            try:
                product_url = product_id.replace(self.base_url, "")
                product_data = self.get(product_url)
                product_text = product_data.get("productText", "")
                
                discussion_id = self._generate_id(product_id)
                
                discussions.append({
                    "id": discussion_id,
                    "location": office,
                    "source_type": "discussion",
                    "headline": f"Forecast Discussion - {office}",
                    "event": "Area Forecast Discussion",
                    "narrative_text": product_text,
                    "issued_at": product.get("issuanceTime"),
                    "effective_at": product.get("issuanceTime"),
                    "payload": json.dumps(product_data),
                    "synced_at": datetime.utcnow().isoformat(),
                })
            except Exception as e:
                print(f"Error fetching discussion {product_id}: {e}")
                continue
        
        return discussions

    def get_weather_for_location(
        self, 
        lat: float, 
        lon: float, 
        location_name: str | None = None,
        include_alerts: bool = True,
        include_forecast: bool = True,
        include_hourly: bool = True,
        include_discussion: bool = False,
        limit: int = 50,
    ) -> list[dict]:
        """
        Fetch all weather data for a location.
        Returns a list of normalized weather documents.
        
        Args:
            include_alerts: Include active weather alerts (default: True)
            include_forecast: Include multi-day forecast, typically 14 periods (default: True)
            include_hourly: Include hourly forecast, typically 24-156 hours (default: True)
            include_discussion: Include Area Forecast Discussion/AFD (default: False)
            limit: Maximum number of items to fetch per source (default: 50)
        """
        documents = []
        
        if include_alerts:
            try:
                documents.extend(self.get_active_alerts(lat, lon, limit=limit, location_name=location_name))
            except Exception as e:
                print(f"Error fetching alerts for {lat},{lon}: {e}")
        
        if include_forecast:
            try:
                documents.extend(self.get_forecast(lat, lon, location_name=location_name, limit=limit))
            except Exception as e:
                print(f"Error fetching forecast for {lat},{lon}: {e}")
        
        if include_hourly:
            try:
                documents.extend(self.get_hourly_forecast(lat, lon, location_name=location_name, limit=limit))
            except Exception as e:
                print(f"Error fetching hourly forecast for {lat},{lon}: {e}")
        
        if include_discussion:
            try:
                grid = self.get_grid_point(lat, lon)
                office = grid.get("gridId")
                if office:
                    documents.extend(self.get_forecast_discussion(office, limit=5))
            except Exception as e:
                print(f"Error fetching discussion for {lat},{lon}: {e}")
        
        return documents

    @staticmethod
    def _generate_id(text: str) -> str:
        """Generate a stable hash ID from text."""
        return hashlib.sha256(text.encode()).hexdigest()


def geocode_location(location: str) -> tuple[float, float] | None:
    """
    Geocode a location string (city, state or full address) to lat/lon coordinates.
    
    Uses OpenStreetMap's Nominatim geocoding service (free, no API key required).
    Falls back to Census API for US addresses if Nominatim fails.
    """
    if not location or not location.strip():
        return None
    
    # Try Nominatim first (works well with city names)
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": location,
                "format": "json",
                "limit": 1,
                "countrycodes": "us",  # Limit to US for weather.gov compatibility
            },
            headers={
                "User-Agent": "Databricks Weather App (student@databricks.com)"
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        
        if data and len(data) > 0:
            result = data[0]
            lat = result.get("lat")
            lon = result.get("lon")
            if lat and lon:
                return float(lat), float(lon)
    except Exception as e:
        print(f"Nominatim geocoding failed for '{location}': {e}")
    
    # Fall back to Census API for US addresses
    try:
        resp = requests.get(
            "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress",
            params={
                "address": location,
                "benchmark": "2020",
                "format": "json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        
        matches = data.get("result", {}).get("addressMatches", [])
        if matches:
            coords = matches[0].get("coordinates", {})
            lon = coords.get("x")
            lat = coords.get("y")
            if lat and lon:
                return float(lat), float(lon)
    except Exception as e:
        print(f"Census geocoding failed for '{location}': {e}")
    
    return None
