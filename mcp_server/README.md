# Weather MCP Server

This directory contains the Weather MCP server implementation.

## Overview

The Weather MCP server exposes weather data tools over MCP (Model Context Protocol) so that Agent Bricks agents can call them. It uses the free National Weather Service API (no API key required).

## Files

### weather_adapter.py

The adapter module that contains all HTTP calls to the National Weather Service API. This module:

* Makes all external API requests
* Handles data parsing and transformation
* Provides clean Python functions that the MCP server can call
* Uses the requests library with proper NWS API headers

**Key Functions:**

* `get_current_weather(location)` - Returns current conditions (temp, humidity, wind, conditions)
* `get_forecast(location, days)` - Returns multi-day forecast (high/low temps, precipitation, conditions)
* `predict_umbrella_needed(location, date)` - Returns umbrella recommendation (> 40% precipitation = bring umbrella)
* `get_travel_recommendation(location, date)` - Returns travel score (0-100) and recommendation (good/fair/poor)

### weather_mcp_server.py

The FastMCP server that exposes the weather tools over MCP. This module:

* Uses FastMCP to create an MCP-compliant server
* Defines @mcp.tool decorated functions for each tool
* Captures end-user identity from Databricks App headers
* Handles errors gracefully and logs all requests
* NO raw requests calls - all HTTP calls are in weather_adapter.py

**Exposed Tools:**

* `get_current_weather(location)` - Current weather conditions
* `get_forecast(location, days=7)` - Multi-day forecast
* `predict_umbrella_needed(location, date)` - Umbrella recommendation
* `get_travel_recommendation(location, date)` - Travel recommendation

## Location Format

All tools accept location as `"lat,lon"` format (e.g., `"40.7128,-74.0060"` for New York City) or city name/full addresses (e.g. Miami, FL)


## Architecture

```
Agent Bricks Agent
    |
    v
weather_mcp_server.py (FastMCP tools)
    |
    v
weather_adapter.py (HTTP calls to NWS API)
    |
    v
National Weather Service API
```

## Example Usage

Once deployed and registered with Agent Bricks, you can ask natural language questions like:

* "What's the current weather in New York?" 
* "Should I bring an umbrella to Chicago on August 15th?"
* "Give me a 3-day forecast for San Francisco"
* "Is August 20th a good day for travel to Miami?"

The Agent Bricks agent will call the appropriate MCP tools and format the responses naturally.