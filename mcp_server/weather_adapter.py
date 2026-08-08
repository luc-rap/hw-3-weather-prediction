"""Weather adapter module for the Weather MCP server.

This module contains all HTTP calls to the
National Weather Service (NWS) API. The MCP server calls these functions
rather than making raw requests directly.

The NWS API is public and requires no authentication - just a proper
User-Agent header. All coordinates use WGS84 (standard lat/lon).
"""

import json
from datetime import datetime, timedelta
from typing import Any

import requests


_BASE_URL = "https://api.weather.gov"
_DEFAULT_TIMEOUT = 30
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """Get or create a requests session with NWS-required headers."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(
            {
                "User-Agent": "(Databricks Weather MCP, lucia.rapanova@gmail.com)",
                "Accept": "application/geo+json",
            }
        )
    return _session


def _get(path: str, params: dict[str, Any] | None = None) -> Any:
    """Make a GET request to the NWS API."""
    resp = _get_session().get(
        f"{_BASE_URL}{path}", params=params, timeout=_DEFAULT_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def _parse_location(location: str) -> tuple[float, float]:
    """
    Parse a location string into lat/lon coordinates.
    
    Accepts:
    - Coordinates: "40.7128,-74.0060" or "40.7128, -74.0060"
    - City with state: "Chicago, IL" or "Chicago, Illinois" (RECOMMENDED)
    - Full addresses: "1600 Pennsylvania Avenue, Washington DC"
    
    Note: For city names, always include the state (e.g., "Chicago, IL" not just "Chicago")
    to avoid ambiguity, as many city names exist in multiple states.
    
    Uses OpenStreetMap Nominatim geocoding API (free, no key required).
    Falls back to Census API for US addresses.
    
    Returns:
        Tuple of (latitude, longitude)
    """
    location = location.strip()
    
    if not location:
        raise ValueError("Location cannot be empty")
    
    # Try to parse as coordinates first
    if "," in location:
        try:
            lat, lon = [float(x.strip()) for x in location.split(",")]
            return (lat, lon)
        except ValueError:
            pass  # Not coordinates, try as city name
    
    # Geocode using Nominatim (works well with city names and addresses)
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
                "User-Agent": "(Databricks Weather MCP, lucia.rapanova@gmail.com)"
            },
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
        
        if results and len(results) > 0:
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            return (lat, lon)
    except Exception as e:
        # Log but continue to Census fallback
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
                return (float(lat), float(lon))
    except Exception as e:
        # Log but don't raise yet
        print(f"Census geocoding failed for '{location}': {e}")
    
    # If all geocoding failed, raise a helpful error
    raise ValueError(
        f"Could not find coordinates for location: {location!r}. "
        f"For city names, include the state (e.g., 'Chicago, IL' not just 'Chicago'). "
        f"Or use coordinates in 'lat,lon' format (e.g., '41.8781,-87.6298')."
    )


def _get_grid_point(lat: float, lon: float) -> dict:
    """
    Resolve a lat/lon to a NWS grid point.
    Returns grid coordinates and forecast URLs.
    """
    data = _get(f"/points/{lat},{lon}")
    properties = data.get("properties", {})
    return {
        "gridId": properties.get("gridId"),
        "gridX": properties.get("gridX"),
        "gridY": properties.get("gridY"),
        "forecast": properties.get("forecast"),
        "forecastHourly": properties.get("forecastHourly"),
    }


def get_current_weather(location: str) -> dict:
    """
    Get current weather conditions for a location.
    
    Args:
        location: Location string - either coordinates "lat,lon" (e.g., "40.7128,-74.0060")
                  or a city with state (e.g., "Chicago, IL", "Miami, FL")
    
    Returns:
        A dict with location, temperature, conditions, humidity, wind, and timestamp.
    """
    # Parse location (handles both coordinates and city names)
    lat, lon = _parse_location(location)
    
    # Get grid point
    grid = _get_grid_point(lat, lon)
    
    # Get latest observation from the gridpoint's observation stations
    # NWS provides observations through nearby stations
    gridId = grid.get("gridId")
    gridX = grid.get("gridX")
    gridY = grid.get("gridY")
    
    if not all([gridId, gridX, gridY]):
        raise ValueError(f"Could not resolve location {location} to a grid point")
    
    # Get observations for this gridpoint
    # Note: observations come from nearby stations, not the gridpoint itself
    observations_url = f"/gridpoints/{gridId}/{gridX},{gridY}/stations"
    stations_data = _get(observations_url)
    
    # Get the first station and its latest observation
    stations = stations_data.get("features", [])
    if not stations:
        raise ValueError(f"No observation stations found for location {location}")
    
    station_id = stations[0].get("properties", {}).get("stationIdentifier")
    if not station_id:
        raise ValueError(f"No station ID found for location {location}")
    
    # Get latest observation from this station
    obs_data = _get(f"/stations/{station_id}/observations/latest")
    obs = obs_data.get("properties", {})
    
    # Extract weather data
    temp_c = obs.get("temperature", {}).get("value")
    temp_f = (temp_c * 9/5) + 32 if temp_c is not None else None
    
    humidity = obs.get("relativeHumidity", {}).get("value")
    wind_speed_ms = obs.get("windSpeed", {}).get("value")
    wind_speed_mph = wind_speed_ms * 2.237 if wind_speed_ms is not None else None
    wind_direction = obs.get("windDirection", {}).get("value")
    
    conditions = obs.get("textDescription", "Unknown")
    timestamp = obs.get("timestamp", datetime.utcnow().isoformat())
    
    return {
        "location": location,
        "temperature_f": round(temp_f, 1) if temp_f is not None else None,
        "temperature_c": round(temp_c, 1) if temp_c is not None else None,
        "conditions": conditions,
        "humidity_percent": round(humidity, 0) if humidity is not None else None,
        "wind_speed_mph": round(wind_speed_mph, 1) if wind_speed_mph is not None else None,
        "wind_direction_degrees": round(wind_direction, 0) if wind_direction is not None else None,
        "timestamp": timestamp,
    }


def get_forecast(location: str, days: int = 7) -> list[dict]:
    """
    Get multi-day forecast for a location.
    
    Args:
        location: Location string - either coordinates "lat,lon" (e.g., "40.7128,-74.0060")
                  or a city with state (e.g., "Chicago, IL", "Miami, FL")
        days: Number of days to forecast (default 7, max 7)
    
    Returns:
        A list of dicts, each with date, temp_high, temp_low, 
        conditions, precipitation_chance, and narrative.
    """
    # Parse location (handles both coordinates and city names)
    lat, lon = _parse_location(location)
    
    # Get grid point
    grid = _get_grid_point(lat, lon)
    forecast_url = grid.get("forecast", "").replace(_BASE_URL, "")
    
    if not forecast_url:
        raise ValueError(f"No forecast URL for location {location}")
    
    # Get forecast data
    data = _get(forecast_url)
    periods = data.get("properties", {}).get("periods", [])
    
    # NWS returns periods as day/night pairs
    # We'll group them by date and extract high/low
    forecast_days = []
    current_date = None
    day_data = {}
    
    for period in periods:
        # Parse start time to get date
        start_time = period.get("startTime", "")
        try:
            dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            continue
        
        # If we're on a new date, save previous day and start a new one
        if date_str != current_date:
            if current_date is not None and day_data:
                forecast_days.append(day_data)
            
            current_date = date_str
            day_data = {
                "date": date_str,
                "temp_high_f": None,
                "temp_low_f": None,
                "conditions": "",
                "precipitation_chance": 0,
                "narrative": "",
            }
        
        # Update day/night data
        temp = period.get("temperature")
        is_daytime = period.get("isDaytime", False)
        
        if is_daytime:
            day_data["temp_high_f"] = temp
            day_data["conditions"] = period.get("shortForecast", "")
            day_data["narrative"] = period.get("detailedForecast", "")
        else:
            day_data["temp_low_f"] = temp
        
        # Precipitation chance (use max from day/night)
        precip = period.get("probabilityOfPrecipitation", {}).get("value", 0) or 0
        day_data["precipitation_chance"] = max(
            day_data["precipitation_chance"], precip
        )
    
    # Add the last day
    if day_data and current_date is not None:
        forecast_days.append(day_data)
    
    # Limit to requested number of days
    return forecast_days[:days]


def predict_umbrella_needed(location: str, date: str) -> dict:
    """
    Predict if an umbrella is needed for a specific date at a location.
    
    This is a derived judgment based on precipitation chance from the forecast.
    
    Args:
        location: Location string - either coordinates "lat,lon" or a city with state (e.g., "Chicago, IL")
        date: Date string in format "YYYY-MM-DD"
    
    Returns:
        A dict with location, date, umbrella_needed (bool), 
        precipitation_chance, and reasoning.
    """
    # Get forecast
    forecast = get_forecast(location, days=7)
    
    # Find the matching date
    target_forecast = None
    for day in forecast:
        if day["date"] == date:
            target_forecast = day
            break
    
    if target_forecast is None:
        raise ValueError(
            f"No forecast available for date {date}. "
            f"Available dates: {[d['date'] for d in forecast]}"
        )
    
    # Decision logic: umbrella needed if precipitation chance > 40%
    precip_chance = target_forecast["precipitation_chance"]
    umbrella_needed = precip_chance > 40
    
    # Build reasoning
    if umbrella_needed:
        reasoning = (
            f"Bring an umbrella! There's a {precip_chance}% chance of precipitation. "
            f"Conditions: {target_forecast['conditions']}"
        )
    else:
        reasoning = (
            f"No umbrella needed. Only {precip_chance}% chance of precipitation. "
            f"Conditions: {target_forecast['conditions']}"
        )
    
    return {
        "location": location,
        "date": date,
        "umbrella_needed": umbrella_needed,
        "precipitation_chance": precip_chance,
        "conditions": target_forecast["conditions"],
        "reasoning": reasoning,
    }


def get_travel_recommendation(location: str, date: str) -> dict:
    """
    Get a travel recommendation for a specific date at a location.
    
    This is a derived judgment based on temperature, precipitation, and conditions.
    
    Args:
        location: Location string - either coordinates "lat,lon" or a city with state (e.g., "Chicago, IL")
        date: Date string in format "YYYY-MM-DD"
    
    Returns:
        A dict with location, date, recommendation (good/fair/poor), 
        score (0-100), and detailed_reasoning.
    """
    # Get forecast
    forecast = get_forecast(location, days=7)
    
    # Find the matching date
    target_forecast = None
    for day in forecast:
        if day["date"] == date:
            target_forecast = day
            break
    
    if target_forecast is None:
        raise ValueError(
            f"No forecast available for date {date}. "
            f"Available dates: {[d['date'] for d in forecast]}"
        )
    
    # Calculate travel score (0-100)
    score = 100
    reasons = []
    
    # Temperature scoring
    temp_high = target_forecast["temp_high_f"]
    temp_low = target_forecast["temp_low_f"]
    
    if temp_high is not None:
        if temp_high > 95 or temp_high < 20:
            score -= 30
            reasons.append(f"Extreme temperature ({temp_high}°F)")
        elif temp_high > 85 or temp_high < 32:
            score -= 15
            reasons.append(f"Uncomfortable temperature ({temp_high}°F)")
    
    # Precipitation scoring
    precip = target_forecast["precipitation_chance"]
    if precip > 70:
        score -= 30
        reasons.append(f"High chance of precipitation ({precip}%)")
    elif precip > 40:
        score -= 15
        reasons.append(f"Moderate chance of precipitation ({precip}%)")
    
    # Conditions-based scoring
    conditions = target_forecast["conditions"].lower()
    severe_conditions = [
        "storm", "tornado", "hurricane", "blizzard", 
        "flood", "severe", "warning"
    ]
    if any(cond in conditions for cond in severe_conditions):
        score -= 40
        reasons.append(f"Severe weather conditions ({target_forecast['conditions']})")
    
    # Determine recommendation category
    if score >= 80:
        recommendation = "good"
    elif score >= 50:
        recommendation = "fair"
    else:
        recommendation = "poor"
    
    # Build detailed reasoning
    if not reasons:
        detailed_reasoning = (
            f"Great day for travel! Temperature: {temp_high}°F, "
            f"{precip}% chance of precipitation. {target_forecast['conditions']}."
        )
    else:
        detailed_reasoning = (
            f"Travel conditions are {recommendation} (score: {score}/100). "
            f"Issues: {'; '.join(reasons)}. "
            f"Temperature: {temp_high}°F/{temp_low}°F, "
            f"{precip}% precipitation chance. {target_forecast['conditions']}."
        )
    
    return {
        "location": location,
        "date": date,
        "recommendation": recommendation,
        "score": score,
        "temperature_high_f": temp_high,
        "temperature_low_f": temp_low,
        "precipitation_chance": precip,
        "conditions": target_forecast["conditions"],
        "detailed_reasoning": detailed_reasoning,
    }