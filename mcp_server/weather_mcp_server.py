"""Weather MCP server.

Exposes weather tools over MCP (Model Context Protocol) so a Databricks
Agent Bricks agent can call them like any other tool:
    - get_current_weather(location) - Current conditions for a location
    - get_forecast(location, days) - Multi-day forecast
    - predict_umbrella_needed(location, date) - Umbrella recommendation
    - get_travel_recommendation(location, date) - Travel recommendation

These tools are backed by the National Weather Service API (see
weather_adapter.py), which is free and public (no API key required).

Location format: "lat,lon" (e.g., "40.7128,-74.0060") or "City, State" (e.g., "Chicago, IL")
Date format: "YYYY-MM-DD" (e.g., "2026-08-15")

Deploy this as its own Databricks App (same app.yaml + FastMCP entrypoint
pattern documented at
https://docs.databricks.com/aws/en/agents/mcp-tools/custom-mcp), separate
from the main weather dashboard app, so an Agent Bricks agent (or any MCP
client) can register its URL as an external MCP server.

Run locally:
    python weather_mcp_server.py
"""

import os
import logging
from contextvars import ContextVar

from fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

import weather_adapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-mcp-server")

# Context variable to store request headers for accessing end-user identity
_request_context: ContextVar[dict] = ContextVar('request_context', default={})


def _get_end_user_email() -> str:
    """Get the actual end user's email from request headers, or fallback to service principal."""
    # Try to get from X-Forwarded-User header (Databricks App context)
    headers = _request_context.get()
    forwarded_user = headers.get('x-forwarded-user')
    if forwarded_user:
        return forwarded_user
    
    # Fallback: use service principal (local development or non-App contexts)
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        return w.current_user.me().user_name or 'unknown@databricks.com'
    except Exception:
        return 'unknown@databricks.com'


mcp = FastMCP("weather-service")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to capture HTTP headers containing end-user identity."""
    async def dispatch(self, request: Request, call_next):
        # Capture headers that Databricks injects with user identity
        headers = {
            'x-forwarded-user': request.headers.get('x-forwarded-user'),
            'x-forwarded-email': request.headers.get('x-forwarded-email'),
        }
        _request_context.set(headers)
        response = await call_next(request)
        return response


@mcp.tool
def get_current_weather(location: str) -> dict:
    """
    Get current weather conditions for a location.
    
    Args:
        location: Location in format "lat,lon" (e.g., "40.7128,-74.0060") or
                  "City, State" (e.g., "Chicago, IL" or "Miami, FL").
                  For city names, always include the state to avoid ambiguity.
    
    Returns:
        A dict with location, temperature (F and C), conditions, humidity,
        wind speed/direction, and timestamp.
    
    Example:
        get_current_weather("40.7128,-74.0060") -> {
            "location": "40.7128,-74.0060",
            "temperature_f": 72.3,
            "temperature_c": 22.4,
            "conditions": "Partly Cloudy",
            "humidity_percent": 65,
            "wind_speed_mph": 8.5,
            "wind_direction_degrees": 180,
            "timestamp": "2026-08-08T14:30:00Z"
        }
    """
    user_email = _get_end_user_email()
    logger.info(f"get_current_weather called by {user_email} for location {location}")
    
    try:
        result = weather_adapter.get_current_weather(location)
        return result
    except Exception as e:
        logger.exception(f"Failed to get current weather for {location}")
        return {
            "status": "error",
            "message": f"Failed to get current weather: {str(e)}",
        }


@mcp.tool
def get_forecast(location: str, days: int = 7) -> list[dict]:
    """
    Get multi-day weather forecast for a location.
    
    Args:
        location: Location in format "lat,lon" (e.g., "40.7128,-74.0060") or
                  "City, State" (e.g., "Chicago, IL"). Include state for city names.
        days: Number of days to forecast (default 7, max 7).
    
    Returns:
        A list of dicts, each representing one day with date, temp_high_f,
        temp_low_f, conditions, precipitation_chance, and narrative.
    
    Example:
        get_forecast("40.7128,-74.0060", days=3) -> [
            {
                "date": "2026-08-08",
                "temp_high_f": 82,
                "temp_low_f": 68,
                "conditions": "Partly Cloudy",
                "precipitation_chance": 20,
                "narrative": "Partly cloudy with a high near 82..."
            },
            ...
        ]
    """
    user_email = _get_end_user_email()
    logger.info(
        f"get_forecast called by {user_email} for location {location}, days={days}"
    )
    
    try:
        result = weather_adapter.get_forecast(location, days)
        return result
    except Exception as e:
        logger.exception(f"Failed to get forecast for {location}")
        return [
            {
                "status": "error",
                "message": f"Failed to get forecast: {str(e)}",
            }
        ]


@mcp.tool
def predict_umbrella_needed(location: str, date: str) -> dict:
    """
    Predict if an umbrella is needed for a specific date at a location.
    
    This is a derived judgment based on precipitation chance. The rule:
    umbrella needed if precipitation chance > 40%.
    
    Args:
        location: Location in format "lat,lon" (e.g., "40.7128,-74.0060") or
                  "City, State" (e.g., "Chicago, IL"). Include state for city names.
        date: Date in format "YYYY-MM-DD" (e.g., "2026-08-15").
    
    Returns:
        A dict with location, date, umbrella_needed (bool),
        precipitation_chance, conditions, and reasoning.
    
    Example:
        predict_umbrella_needed("40.7128,-74.0060", "2026-08-15") -> {
            "location": "40.7128,-74.0060",
            "date": "2026-08-15",
            "umbrella_needed": true,
            "precipitation_chance": 60,
            "conditions": "Showers",
            "reasoning": "Bring an umbrella! There's a 60% chance of precipitation. Conditions: Showers"
        }
    """
    user_email = _get_end_user_email()
    logger.info(
        f"predict_umbrella_needed called by {user_email} for location {location}, date {date}"
    )
    
    try:
        result = weather_adapter.predict_umbrella_needed(location, date)
        return result
    except Exception as e:
        logger.exception(f"Failed to predict umbrella for {location} on {date}")
        return {
            "status": "error",
            "message": f"Failed to predict umbrella: {str(e)}",
        }


@mcp.tool
def get_travel_recommendation(location: str, date: str) -> dict:
    """
    Get a travel recommendation for a specific date at a location.
    
    This is a derived judgment based on temperature, precipitation, and
    severe weather conditions. Returns a score (0-100) and recommendation
    category (good/fair/poor).
    
    Args:
        location: Location in format "lat,lon" (e.g., "40.7128,-74.0060") or
                  "City, State" (e.g., "Chicago, IL"). Include state for city names.
        date: Date in format "YYYY-MM-DD" (e.g., "2026-08-15").
    
    Returns:
        A dict with location, date, recommendation (good/fair/poor),
        score (0-100), weather details, and detailed_reasoning.
    
    Example:
        get_travel_recommendation("40.7128,-74.0060", "2026-08-15") -> {
            "location": "40.7128,-74.0060",
            "date": "2026-08-15",
            "recommendation": "fair",
            "score": 70,
            "temperature_high_f": 88,
            "temperature_low_f": 72,
            "precipitation_chance": 30,
            "conditions": "Partly Cloudy",
            "detailed_reasoning": "Travel conditions are fair (score: 70/100). Issues: Uncomfortable temperature (88°F). Temperature: 88°F/72°F, 30% precipitation chance. Partly Cloudy."
        }
    """
    user_email = _get_end_user_email()
    logger.info(
        f"get_travel_recommendation called by {user_email} for location {location}, date {date}"
    )
    
    try:
        result = weather_adapter.get_travel_recommendation(location, date)
        return result
    except Exception as e:
        logger.exception(f"Failed to get travel recommendation for {location} on {date}")
        return {
            "status": "error",
            "message": f"Failed to get travel recommendation: {str(e)}",
        }


if __name__ == "__main__":
    # Add middleware to capture request headers for end-user identity
    # This must be done before mcp.run() is called
    if hasattr(mcp, 'app') and mcp.app is not None:
        mcp.app.add_middleware(RequestContextMiddleware)
    
    # Databricks Apps route external HTTP traffic to this port via app.yaml;
    # streamable-http is the transport Databricks' MCP client/gateway expects
    # (see the "Host your own MCP" doc linked in the module docstring above).
    port = int(os.getenv("DATABRICKS_APP_PORT", os.getenv("PORT", 8000)))
    mcp.run(transport="http", host="0.0.0.0", port=port)