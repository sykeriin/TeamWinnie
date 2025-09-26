
"""
Aviation Weather Analysis Service - Integrated Version
=====================================================
This code integrates with an existing main.py file that handles weather data fetching.
It provides AI-powered analysis of aviation weather data using Google Gemini API.

Key Features:
- Integrates with existing weather data fetching system
- AI-powered weather analysis using Google Gemini
- Route-based weather analysis
- Multiple analysis types (comprehensive, severity, forecast, route)
- RESTful API endpoints
- Safety-critical logging and error handling

Dependencies:
- main.py (your existing weather data fetching module)
- Google Gemini API
- Flask for API endpoints
- Various Python libraries for data processing
"""

import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
from geopy.distance import geodesic
import re


# Import your existing weather data fetching functionality
# Assuming your main.py has functions like get_metar_data, get_taf_data, etc.
try:
    from main import (
        metar,      # Replace with your actual function names
        taf,        # Replace with your actual function names  
        pirep,      # Replace with your actual function names
        isigmet,     # Replace with your actual function names
        airsigmet      # Replace with your actual function names
    )
    print("Successfully imported weather data functions from main.py")
except ImportError as e:
    print(f"Warning: Could not import from main.py: {e}")
    print("Please ensure your main.py file exists and has the required functions")
    # You'll need to replace these with your actual function names

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
"""
Configure comprehensive logging for this safety-critical aviation application.
All weather analysis operations and errors are logged for audit and debugging.
"""
logging.basicConfig(
    level=logging.INFO,  # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s',  # Timestamp and message format
    handlers=[
        logging.FileHandler('aviation_weather_analysis.log'),  # File logging
        logging.StreamHandler()  # Console logging
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# FLASK APPLICATION SETUP
# ============================================================================
"""
Initialize Flask application with CORS support for cross-origin requests.
This allows the API to be called from web applications running on different domains.
"""
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,application/json')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers['Content-Type'] = 'application/json'
    return response

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class WeatherSeverity(Enum):
    """
    Defines weather severity levels for aviation operations.
    These levels help pilots make go/no-go decisions.
    """
    CLEAR = "clear"          # Perfect VFR conditions
    LIGHT = "light"          # Minor weather, VFR with caution
    MODERATE = "moderate"    # IFR conditions or moderate impacts
    SEVERE = "severe"        # Dangerous conditions, avoid if possible
    EXTREME = "extreme"      # Very dangerous, strongly advise against flight
    CRITICAL = "critical"    # Extreme danger, cease flight operations

class ReportType(Enum):
    """
    Aviation weather report types as defined by ICAO standards.
    Each type provides different information crucial for flight planning.
    """
    METARs = "metar"    # Meteorological Aerodrome Report (current conditions)
    TAFs = "taf"        # Terminal Aerodrome Forecast (forecast conditions)
    PIREPs = "pirep"    # Pilot Report (real pilot observations)
    SIGMETs = "sigmet"  # Significant Meteorological Information (hazardous weather)
    AIRMETs = "airmet"  # Airmen's Meteorological Information (moderate weather)

@dataclass
class WeatherData:
    """
    Standardized data structure for all weather information.
    This ensures consistent data handling across different report types.
    """
    report_type: str                           # Type of weather report (METAR, TAF, etc.)
    station_id: str                           # ICAO airport code (e.g., KJFK, EGLL)
    raw_data: str                            # Original weather report text
    timestamp: datetime                       # When the report was issued
    location: Optional[Dict[str, float]] = None  # Latitude/longitude if available

@dataclass
class RoutePoint:
    """
    Represents a waypoint along a flight route.
    Used for route-based weather analysis and flight planning.
    """
    icao_code: str    # ICAO airport identifier
    name: str         # Human-readable airport name
    latitude: float   # Latitude in decimal degrees
    longitude: float  # Longitude in decimal degrees

# ============================================================================
# WEATHER DATA INTEGRATION CLASS
# ============================================================================

class WeatherDataIntegrator:
    """
    This class integrates with your existing main.py weather data fetching functions.
    It standardizes the data format and handles errors from your existing system.

    IMPORTANT: You need to modify the function calls in this class to match
    the actual function names and signatures in your main.py file.
    """

    def __init__(self):
        """
        Initialize the integrator.
        Add any configuration needed for your existing weather data functions.
        """
        self.session = requests.Session()  # Reuse HTTP connections for efficiency
        logger.info("WeatherDataIntegrator initialized")

    def fetch_metar(self, station_ids: List[str], hours_before: int = 2) -> List[WeatherData]:
        """
        Fetch METAR data using your existing main.py functions.

        Args:
            station_ids: List of ICAO airport codes (e.g., ['KJFK', 'KLAX'])
            hours_before: How many hours of historical data to fetch

        Returns:
            List of WeatherData objects containing METAR information
        """
        try:
            logger.info(f"Fetching METAR data for stations: {station_ids}")

            # MODIFY THIS: Call your actual METAR fetching function from main.py
            # Replace 'get_metar_data' with your actual function name
            raw_metar_data = metar(station_ids, hours_before)

            # Convert your data format to our standardized WeatherData format
            weather_data_list = []

            # MODIFY THIS: Adapt this loop to match your data format
            for station_id in station_ids:
                if station_id in raw_metar_data:  # Adjust based on your data structure
                    weather_data = WeatherData(
                        report_type=ReportType.METAR.value,
                        station_id=station_id,
                        raw_data=raw_metar_data[station_id],  # Adjust field names
                        timestamp=datetime.utcnow(),  # Use actual timestamp from your data
                        location=None  # Add coordinates if available in your data
                    )
                    weather_data_list.append(weather_data)

            logger.info(f"Successfully fetched {len(weather_data_list)} METAR reports")
            return weather_data_list

        except Exception as e:
            logger.error(f"Error fetching METAR data: {e}")
            return []  # Return empty list on error to prevent crashes

    def fetch_taf(self, station_ids: List[str]) -> List[WeatherData]:
        """
        Fetch TAF (Terminal Aerodrome Forecast) data using your existing functions.

        Args:
            station_ids: List of ICAO airport codes

        Returns:
            List of WeatherData objects containing TAF information
        """
        try:
            logger.info(f"Fetching TAF data for stations: {station_ids}")

            # MODIFY THIS: Call your actual TAF fetching function from main.py
            raw_taf_data = taf(station_ids)

            weather_data_list = []

            # MODIFY THIS: Adapt this loop to match your TAF data format
            for station_id in station_ids:
                if station_id in raw_taf_data:
                    weather_data = WeatherData(
                        report_type=ReportType.TAF.value,
                        station_id=station_id,
                        raw_data=raw_taf_data[station_id],
                        timestamp=datetime.utcnow(),
                        location=None
                    )
                    weather_data_list.append(weather_data)

            logger.info(f"Successfully fetched {len(weather_data_list)} TAF reports")
            return weather_data_list

        except Exception as e:
            logger.error(f"Error fetching TAF data: {e}")
            return []

    def fetch_pirep(self, bounds: Dict[str, float], hours_before: int = 6) -> List[WeatherData]:
        """
        Fetch PIREP (Pilot Report) data using your existing functions.

        Args:
            bounds: Geographic boundaries (min_lat, max_lat, min_lon, max_lon)
            hours_before: How many hours of historical PIREP data to fetch

        Returns:
            List of WeatherData objects containing PIREP information
        """
        try:
            logger.info(f"Fetching PIREP data for bounds: {bounds}")

            # MODIFY THIS: Call your actual PIREP fetching function from main.py
            raw_pirep_data = pirep(bounds, hours_before)

            weather_data_list = []

            # MODIFY THIS: Adapt this to match your PIREP data format
            # PIREPs might not have specific station IDs, so handle accordingly
            if isinstance(raw_pirep_data, list):
                for i, pirep in enumerate(raw_pirep_data):
                    weather_data = WeatherData(
                        report_type=ReportType.PIREP.value,
                        station_id=f"PIREP_{i}",  # Generate ID if not available
                        raw_data=str(pirep),  # Convert to string if needed
                        timestamp=datetime.utcnow(),  # Use actual timestamp from data
                        location=None  # Add lat/lon if available
                    )
                    weather_data_list.append(weather_data)

            logger.info(f"Successfully fetched {len(weather_data_list)} PIREP reports")
            return weather_data_list

        except Exception as e:
            logger.error(f"Error fetching PIREP data: {e}")
            return []
    

    def fetch_sigmet_airmet(self, bounds: Dict[str, float]) -> Dict[str, list]:
        """
        Fetch SIGMET and AIRMET data using your existing functions.
        Returns a dictionary with 'sigmet' and 'airmet' keys.
        """
        try:
            sigmet_list = []
            airmet_list = []

            raw_sigmet_data = isigmet(bounds)
            raw_airmet_data = airsigmet(bounds)

            if isinstance(raw_sigmet_data, list):
                for i, sigmet in enumerate(raw_sigmet_data):
                    sigmet_list.append(
                        WeatherData(
                            report_type=ReportType.SIGMET.value,
                            station_id=f"SIGMET_{i}",
                            raw_data=str(sigmet),
                            timestamp=datetime.utcnow(),
                            location=None
                        )
                    )

            if isinstance(raw_airmet_data, list):
                for i, airmet in enumerate(raw_airmet_data):
                    airmet_list.append(
                        WeatherData(
                            report_type=ReportType.AIRMET.value,
                            station_id=f"AIRMET_{i}",
                            raw_data=str(airmet),
                            timestamp=datetime.utcnow(),
                            location=None
                        )
                    )

            return {"sigmet": sigmet_list, "airmet": airmet_list}
        except Exception as e:
            logger.error(f"Error fetching SIGMET/AIRMET data: {e}")
            return {"sigmet": [], "airmet": []}

    def airsigmet(self, bounds: Dict[str, float]) -> Dict[str, List[WeatherData]]:
        """
        Fetch SIGMET and AIRMET data using your existing functions.

        Args:
            bounds: Geographic boundaries for the search area

        Returns:
            Dictionary with 'sigmet' and 'airmet' keys containing lists of WeatherData
        """
        try:
            logger.info(f"Fetching SIGMET/AIRMET data for bounds: {bounds}")

            # MODIFY THIS: Call your actual SIGMET/AIRMET functions from main.py
            raw_sigmet_data = isigmet(bounds)
            raw_airmet_data = airsigmet(bounds)

            # Process SIGMET data
            sigmet_list = []
            if isinstance(raw_sigmet_data, list):
                for i, sigmet in enumerate(raw_sigmet_data):
                    weather_data = WeatherData(
                        report_type=ReportType.SIGMET.value,
                        station_id=f"SIGMET_{i}",
                        raw_data=str(sigmet),
                        timestamp=datetime.utcnow(),
                        location=None
                    )
                    sigmet_list.append(weather_data)

            # Process AIRMET data
            airmet_list = []
            if isinstance(raw_airmet_data, list):
                for i, airmet in enumerate(raw_airmet_data):
                    weather_data = WeatherData(
                        report_type=ReportType.AIRMET.value,
                        station_id=f"AIRMET_{i}",
                        raw_data=str(airmet),
                        timestamp=datetime.utcnow(),
                        location=None
                    )
                    airmet_list.append(weather_data)

            result = {
                'sigmete': sigmet_list,
                'airmete': airmet_list
            }

            logger.info(f"Successfully fetched {len(sigmet_list)} SIGMET and {len(airmet_list)} AIRMET reports")
            return result

        except Exception as e:
            logger.error(f"Error fetching SIGMET/AIRMET data: {e}")
            return {'sigmete': [], 'airmete': []}

# ============================================================================
# AI WEATHER ANALYSIS CLASS
# ============================================================================

class GeminiWeatherAnalyzer:
    """
    Handles AI-powered weather analysis using Google's Gemini API.
    This class takes raw weather data and provides intelligent analysis
    that helps pilots make informed decisions about flight safety.
    """

    def __init__(self, api_key: str):
        """
        Initialize the Gemini AI analyzer.

        Args:
            api_key: Your Google Gemini API key
        """
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            logger.info("Gemini AI analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini AI: {e}")
            raise

    def analyze_weather_data(self, weather_data: List[WeatherData], 
                           analysis_type: str = "comprehensive",
                           filters: List[str] = None) -> Dict:
        """
        Main analysis function that processes weather data using AI.

        Args:
            weather_data: List of WeatherData objects to analyze
            analysis_type: Type of analysis ('comprehensive', 'severity', 'forecast', 'route')
            filters: List of report types to include in analysis

        Returns:
            Dictionary containing AI analysis results
        """
        try:
            logger.info(f"Starting {analysis_type} analysis of {len(weather_data)} weather reports")

            # Filter data if specific types are requested
            if filters:
                weather_data = [wd for wd in weather_data if wd.report_type in filters]
                logger.info(f"Filtered to {len(weather_data)} reports matching: {filters}")

            # Prepare context for AI analysis
            context = self._prepare_analysis_context(weather_data)

            # Route to appropriate analysis method
            if analysis_type == "comprehensive":
                return self._comprehensive_analysis(context)
            elif analysis_type == "severity":
                return self._severity_analysis(context)
            elif analysis_type == "forecast":
                return self._forecast_analysis(context)
            elif analysis_type == "route":
                return self._route_analysis(context)
            else:
                logger.warning(f"Unknown analysis type: {analysis_type}, defaulting to comprehensive")
                return self._comprehensive_analysis(context)

        except Exception as e:
            logger.error(f"Error in AI weather analysis: {e}")
            return {
                "error": "Analysis failed", 
                "severity": "unknown",
                "timestamp": datetime.utcnow().isoformat()
            }

    def _prepare_analysis_context(self, weather_data: List[WeatherData]) -> str:
        """
        Prepare weather data context for AI analysis.
        This creates a structured text format that the AI can understand.

        Args:
            weather_data: List of weather reports to format

        Returns:
            Formatted string containing all weather data for AI analysis
        """
        context_parts = []

        # Add critical safety header
        context_parts.append("=== CRITICAL AVIATION WEATHER ANALYSIS ===")
        context_parts.append("SAFETY FIRST: Lives depend on accurate interpretation.")
        context_parts.append("Do not miss any significant weather phenomena.")
        context_parts.append("Provide precise, actionable aviation weather guidance.")
        context_parts.append("")

        # Add timestamp and data summary
        context_parts.append(f"Analysis Time: {datetime.utcnow().isoformat()} UTC")
        context_parts.append(f"Total Reports: {len(weather_data)}")
        context_parts.append("")

        # Format each weather report
        for i, data in enumerate(weather_data, 1):
            context_parts.append(f"=== REPORT {i}: {data.report_type.upper()} - {data.station_id} ===")
            context_parts.append(f"Report Time: {data.timestamp.isoformat()}")
            context_parts.append(f"Raw Data: {data.raw_data}")

            # Add location if available
            if data.location:
                context_parts.append(f"Location: {data.location}")

            context_parts.append("")  # Empty line between reports

        formatted_context = "\n".join(context_parts)
        logger.debug(f"Prepared analysis context with {len(formatted_context)} characters")

        return formatted_context

    def _comprehensive_analysis(self, context: str) -> Dict:
        """
        Perform comprehensive weather analysis covering all aspects important for aviation.

        Args:
            context: Formatted weather data context

        Returns:
            Dictionary with comprehensive analysis results
        """
        prompt = f"""
{context}

As an expert aviation meteorologist, provide a comprehensive weather analysis for flight operations.
Your analysis must be precise, safety-focused, and actionable for pilots.

Analyze and provide:
1. CURRENT CONDITIONS: Summarize present weather at all locations
2. VISIBILITY: Visibility conditions, restrictions, and impacts on VFR/IFR operations
3. WINDS: Wind speed, direction, gusts, and turbulence indicators
4. PRECIPITATION: Type, intensity, and flight impacts
5. CLOUDS/CEILING: Cloud layers, heights, and IFR implications  
6. TEMPERATURE: Current temps, trends, and icing potential
7. PRESSURE: Altimeter settings and trends
8. HAZARDS: All weather hazards that could affect flight safety
9. RECOMMENDATIONS: Specific flight planning recommendations
10. SEVERITY: Overall severity rating

Format response as valid JSON with these keys:
- summary: Brief overall conditions summary (2-3 sentences)
- visibility: Visibility analysis and VFR/IFR status
- winds: Wind conditions and turbulence assessment
- precipitation: Precipitation type, intensity, impacts
- clouds: Cloud coverage, heights, ceiling information
- temperature: Temperature analysis and icing risk
- pressure: Pressure readings and trends
- hazards: List of specific weather hazards
- recommendations: Actionable flight planning advice
- severity_rating: One of: clear, light, moderate, severe, extreme, critical

Be concise but complete. Focus on flight safety implications.
"""

        try:
            logger.info("Generating comprehensive weather analysis")
            response = self.model.generate_content(prompt)
            analysis_result = self._parse_ai_response(response.text)

            # Validate severity rating
            if 'severity_rating' in analysis_result:
                valid_severities = [s.value for s in WeatherSeverity]
                if analysis_result['severity_rating'] not in valid_severities:
                    logger.warning(f"Invalid severity rating: {analysis_result['severity_rating']}")
                    analysis_result['severity_rating'] = 'unknown'

            logger.info("Comprehensive analysis completed successfully")
            return analysis_result

        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            return {
                "error": "Comprehensive analysis failed", 
                "severity_rating": "unknown",
                "timestamp": datetime.utcnow().isoformat()
            }

    def _severity_analysis(self, context: str) -> Dict:
        """
        Perform quick severity assessment for immediate go/no-go decisions.

        Args:
            context: Formatted weather data context

        Returns:
            Dictionary with severity assessment results
        """
        prompt = f"""
{context}

As an expert aviation meteorologist, provide a rapid severity assessment for immediate flight decision-making.

Severity Levels:
- clear: Perfect VFR conditions, no significant weather
- light: Minor weather impacts, VFR operations possible with normal caution
- moderate: IFR conditions or moderate weather impacts, increased caution required
- severe: Dangerous conditions, flight operations should be avoided if possible
- extreme: Very dangerous conditions, strongly advise against any flight operations
- critical: Extreme danger to life and aircraft, all flight operations should cease

Analyze the weather data and provide:
1. Overall severity assessment (one word from above levels)
2. Brief explanation of the assessment (2-3 sentences)
3. Key factors that influenced the severity rating
4. Immediate recommendations for flight operations

Format as valid JSON with keys:
- severity: One of the severity levels above
- explanation: Brief explanation of conditions (2-3 sentences)
- key_factors: List of main weather factors affecting severity
- immediate_action: Immediate recommendation for flight operations
- confidence: Your confidence in this assessment (high/medium/low)

Focus on the most critical weather factors that affect flight safety.
"""

        try:
            logger.info("Generating severity assessment")
            response = self.model.generate_content(prompt)
            severity_result = self._parse_ai_response(response.text)

            # Validate severity
            if 'severity' in severity_result:
                valid_severities = [s.value for s in WeatherSeverity]
                if severity_result['severity'] not in valid_severities:
                    logger.warning(f"Invalid severity: {severity_result['severity']}")
                    severity_result['severity'] = 'unknown'

            logger.info("Severity assessment completed successfully")
            return severity_result

        except Exception as e:
            logger.error(f"Severity analysis failed: {e}")
            return {
                "severity": "unknown", 
                "explanation": "Severity analysis unavailable due to processing error",
                "error": str(e)
            }

    def _forecast_analysis(self, context: str) -> Dict:
        """
        Analyze forecast trends and timing for flight planning.

        Args:
            context: Formatted weather data context

        Returns:
            Dictionary with forecast analysis results
        """
        prompt = f"""
{context}

As an expert aviation meteorologist, analyze weather trends and forecasts for flight planning.

Based on TAF data, current conditions, and weather patterns, provide:

1. FORECAST PERIODS: Break down conditions by time periods (next 6, 12, 24 hours)
2. SIGNIFICANT CHANGES: Identify when major weather changes will occur
3. TIME-SPECIFIC CONDITIONS: Detail conditions for different flight time windows
4. TREND ANALYSIS: Overall weather trend (improving/deteriorating/stable)
5. TIMING RECOMMENDATIONS: Best and worst times for flight operations

Format as valid JSON with keys:
- forecast_periods: Object with time periods and their conditions
- significant_changes: List of major weather changes with timing
- trend_direction: Overall trend (improving/deteriorating/stable)
- best_flight_times: Recommended time windows for flight operations
- worst_flight_times: Time periods to avoid for flight operations
- planning_notes: Additional flight planning considerations
- confidence: Forecast confidence level (high/medium/low)

Focus on actionable timing information for flight planning decisions.
"""

        try:
            logger.info("Generating forecast analysis")
            response = self.model.generate_content(prompt)
            forecast_result = self._parse_ai_response(response.text)
            logger.info("Forecast analysis completed successfully")
            return forecast_result

        except Exception as e:
            logger.error(f"Forecast analysis failed: {e}")
            return {
                "error": "Forecast analysis failed",
                "trend_direction": "unknown",
                "confidence": "low"
            }

    def _route_analysis(self, context: str) -> Dict:
        """
        Analyze weather conditions along a specific flight route.

        Args:
            context: Formatted weather data context

        Returns:
            Dictionary with route-specific analysis results
        """
        prompt = f"""
{context}

As an expert aviation meteorologist, analyze weather conditions along this flight route.

Provide route-specific analysis including:

1. ROUTE SEGMENTS: Break down weather by route segments/waypoints
2. HAZARD MAPPING: Identify weather hazards along the route
3. ALTITUDE RECOMMENDATIONS: Suggest optimal altitudes for best conditions
4. ALTERNATE ROUTING: Identify if route deviations are needed
5. CRITICAL AREAS: Highlight route segments with most challenging conditions
6. FUEL PLANNING: Weather impacts on fuel requirements

Format as valid JSON with keys:
- route_segments: Object mapping waypoints to weather conditions
- hazards_by_location: Weather hazards mapped to specific route locations
- altitude_recommendations: Suggested altitudes with reasoning
- route_deviations: Recommended route changes if needed
- critical_segments: Most challenging parts of the route
- fuel_considerations: Weather impacts on fuel planning
- overall_route_assessment: Overall route weather assessment

Focus on practical route planning and safety considerations.
"""

        try:
            logger.info("Generating route-specific analysis")
            response = self.model.generate_content(prompt)
            route_result = self._parse_ai_response(response.text)
            logger.info("Route analysis completed successfully")
            return route_result

        except Exception as e:
            logger.error(f"Route analysis failed: {e}")
            return {
                "error": "Route analysis failed",
                "overall_route_assessment": "Analysis unavailable"
            }

    def _parse_ai_response(self, response_text: str) -> Dict:
        """
        Parse AI response text into JSON format, handling potential formatting issues.

        Args:
            response_text: Raw text response from AI

        Returns:
            Parsed dictionary or error information
        """
        try:
            # Remove markdown code blocks if present
            cleaned_response = re.sub(r'```json\n?|```\n?', '', response_text)
            cleaned_response = cleaned_response.strip()

            # Attempt to parse as JSON
            parsed_response = json.loads(cleaned_response)
            logger.debug("Successfully parsed AI response as JSON")
            return parsed_response

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI response as JSON: {e}")
            logger.debug(f"Raw response text: {response_text[:500]}...")  # Log first 500 chars

            # Return raw text with parse error flag
            return {
                "raw_response": response_text,
                "parse_error": True,
                "error_message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# ============================================================================
# ROUTE MANAGEMENT CLASS
# ============================================================================

class RouteManager:
    """
    Manages flight route calculations and waypoint determination.
    This class helps identify waypoints along a route for weather analysis.
    """

    def __init__(self):
        """
        Initialize with a database of major airports.
        In production, this would use a comprehensive airport database.
        """
        # Major airports database - MODIFY THIS to include airports relevant to your area
        self.airports = {
            # Indian airports (modify based on your region)
            'VOBL': RoutePoint('VOBL', 'Bangalore (Kempegowda)', 12.9716, 77.5946),
            'VOBZ': RoutePoint('VOBZ', 'Vijayawada', 16.5311, 80.7983),
            'VECC': RoutePoint('VECC', 'Kolkata (Netaji Subhas)', 22.6547, 88.4467),
            'VEGT': RoutePoint('VEGT', 'Bagdogra', 26.6815, 88.3285),
            'VIDP': RoutePoint('VIDP', 'Delhi (Indira Gandhi)', 28.5562, 77.1000),
            'VOMM': RoutePoint('VOMM', 'Chennai', 12.9941, 80.1805),
            'VABB': RoutePoint('VABB', 'Mumbai (Chhatrapati Shivaji)', 19.0896, 72.8656),
            'VOHS': RoutePoint('VOHS', 'Hyderabad', 17.2403, 78.4294),
            'VOCI': RoutePoint('VOCI', 'Coimbatore', 11.0297, 77.0436),
            'VOTR': RoutePoint('VOTR', 'Tiruchirappalli', 10.7654, 78.7097),

            # Add more airports as needed for your region
            # Format: 'ICAO': RoutePoint('ICAO', 'Name', latitude, longitude)
        }

        logger.info(f"RouteManager initialized with {len(self.airports)} airports")

    def calculate_route_points(self, origin: str, destination: str) -> List[RoutePoint]:
        """
        Calculate waypoints along a flight route.

        Args:
            origin: ICAO code of departure airport
            destination: ICAO code of arrival airport

        Returns:
            List of RoutePoint objects representing the flight path
        """
        try:
            # Validate airport codes
            if origin not in self.airports:
                logger.error(f"Origin airport not found: {origin}")
                return []

            if destination not in self.airports:
                logger.error(f"Destination airport not found: {destination}")
                return []

            origin_point = self.airports[origin]
            dest_point = self.airports[destination]

            logger.info(f"Calculating route from {origin} to {destination}")

            # Start with origin airport
            route_points = [origin_point]

            # Calculate great circle distance
            total_distance = geodesic(
                (origin_point.latitude, origin_point.longitude),
                (dest_point.latitude, dest_point.longitude)
            ).kilometers

            logger.info(f"Total route distance: {total_distance:.1f} km")

            # Add intermediate waypoints for longer routes
            if total_distance > 500:  # Add waypoints for routes longer than 500km
                intermediate_points = self._find_intermediate_waypoints(
                    origin_point, dest_point, total_distance
                )
                route_points.extend(intermediate_points)
                logger.info(f"Added {len(intermediate_points)} intermediate waypoints")

            # Add destination airport
            route_points.append(dest_point)

            logger.info(f"Route calculation complete: {len(route_points)} total waypoints")
            return route_points

        except Exception as e:
            logger.error(f"Route calculation failed: {e}")
            return []

    def _find_intermediate_waypoints(self, origin: RoutePoint, destination: RoutePoint, 
                                   total_distance: float) -> List[RoutePoint]:
        """
        Find airports that lie roughly along the route between origin and destination.

        Args:
            origin: Starting airport
            destination: Ending airport
            total_distance: Total route distance in kilometers

        Returns:
            List of intermediate RoutePoint objects
        """
        intermediate_points = []
        tolerance_km = min(150, total_distance * 0.15)  # 15% of route distance or 150km

        logger.debug(f"Searching for waypoints within {tolerance_km:.1f}km of route")

        # Check each airport to see if it's roughly on the route
        for icao, airport in self.airports.items():
            # Skip origin and destination
            if icao in [origin.icao_code, destination.icao_code]:
                continue

            if self._is_on_route(origin, destination, airport, tolerance_km):
                intermediate_points.append(airport)
                logger.debug(f"Added intermediate waypoint: {icao} ({airport.name})")

        # Sort waypoints by distance from origin
        intermediate_points.sort(
            key=lambda wp: geodesic(
                (origin.latitude, origin.longitude),
                (wp.latitude, wp.longitude)
            ).kilometers
        )

        return intermediate_points

    def _is_on_route(self, origin: RoutePoint, destination: RoutePoint, 
                     candidate: RoutePoint, tolerance_km: float) -> bool:
        """
        Check if an airport is roughly on the great circle route between two points.

        Args:
            origin: Starting point
            destination: Ending point
            candidate: Airport to check
            tolerance_km: Maximum deviation from route in kilometers

        Returns:
            True if the airport is on the route within tolerance
        """
        # Calculate direct distance from origin to destination
        direct_distance = geodesic(
            (origin.latitude, origin.longitude),
            (destination.latitude, destination.longitude)
        ).kilometers

        # Calculate distance via the candidate airport
        dist_origin_to_candidate = geodesic(
            (origin.latitude, origin.longitude),
            (candidate.latitude, candidate.longitude)
        ).kilometers

        dist_candidate_to_destination = geodesic(
            (candidate.latitude, candidate.longitude),
            (destination.latitude, destination.longitude)
        ).kilometers

        total_via_candidate = dist_origin_to_candidate + dist_candidate_to_destination

        # Check if the deviation is within tolerance
        deviation = abs(total_via_candidate - direct_distance)

        return deviation <= tolerance_km

# ============================================================================
# MAIN WEATHER SERVICE CLASS
# ============================================================================

class WeatherService:
    """
    Main orchestrating class that coordinates weather data fetching and AI analysis.
    This is the primary interface for all weather analysis operations.
    """

    def __init__(self, gemini_api_key: str):
        """
        Initialize the complete weather analysis service.

        Args:
            gemini_api_key: Google Gemini API key for AI analysis
        """
        try:
            self.weather_integrator = WeatherDataIntegrator()  # Handles your main.py integration
            self.ai_analyzer = GeminiWeatherAnalyzer(gemini_api_key)  # AI analysis
            self.route_manager = RouteManager()  # Route calculations

            logger.info("WeatherService initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize WeatherService: {e}")
            raise

    def get_weather_analysis(self, station_ids: List[str], 
                           report_types: List[str] = None,
                           analysis_type: str = "comprehensive",
                           include_forecast: bool = False,
                           raw_data: bool = False) -> Dict:
        """
        Main method to get comprehensive weather analysis.

        Args:
            station_ids: List of ICAO airport codes to analyze
            report_types: Types of reports to include (metar, taf, pirep, etc.)
            analysis_type: Type of analysis (comprehensive, severity, forecast, route)
            include_forecast: Whether to include forecast analysis
            raw_data: Whether to include raw weather data in response

        Returns:
            Dictionary containing complete weather analysis
        """
        try:
            logger.info(f"Starting weather analysis for stations: {station_ids}")
            logger.info(f"Analysis type: {analysis_type}, Report types: {report_types}")

            all_weather_data = []

            # Calculate geographic bounds for area-based reports (PIREP, SIGMET, AIRMET)
            bounds = self._calculate_bounds_from_stations(station_ids)

            # Fetch different types of weather data based on request
            if not report_types or 'metar' in report_types:
                logger.info("Fetching METAR data...")
                metar_data = self.weather_integrator.fetch_metar(station_ids)
                all_weather_data.extend(metar_data)
                logger.info(f"Retrieved {len(metar_data)} METAR reports")

            if not report_types or 'taf' in report_types:
                logger.info("Fetching TAF data...")
                taf_data = self.weather_integrator.fetch_taf(station_ids)
                all_weather_data.extend(taf_data)
                logger.info(f"Retrieved {len(taf_data)} TAF reports")

            if not report_types or 'pirep' in report_types:
                logger.info("Fetching PIREP data...")
                pirep_data = self.weather_integrator.fetch_pirep(bounds)
                all_weather_data.extend(pirep_data)
                logger.info(f"Retrieved {len(pirep_data)} PIREP reports")

            if not report_types or any(rt in ['sigmet', 'airmet'] for rt in report_types):
                logger.info("Fetching SIGMET/AIRMET data...")
                sigmet_airmet = self.weather_integrator.fetch_sigmet_airmet(bounds)

                if 'sigmet' in report_types or not report_types:
                    all_weather_data.extend(sigmet_airmet['sigmet'])
                    logger.info(f"Retrieved {len(sigmet_airmet['sigmet'])} SIGMET reports")

                if 'airmet' in report_types or not report_types:
                    all_weather_data.extend(sigmet_airmet['airmet'])
                    logger.info(f"Retrieved {len(sigmet_airmet['airmet'])} AIRMET reports")

            logger.info(f"Total weather reports collected: {len(all_weather_data)}")

            # Perform AI analysis on collected data
            logger.info("Starting AI weather analysis...")
            analysis = self.ai_analyzer.analyze_weather_data(
                all_weather_data, analysis_type, report_types
            )

            # Prepare comprehensive response
            response = {
                "timestamp": datetime.utcnow().isoformat(),
                "stations_analyzed": station_ids,
                "analysis_type": analysis_type,
                "analysis": analysis,
                "data_sources": list(set([wd.report_type for wd in all_weather_data])),
                "total_reports": len(all_weather_data)
            }

            # Add raw data if requested
            if raw_data:
                logger.info("Including raw weather data in response")
                response["raw_data"] = [
                    {
                        "type": wd.report_type,
                        "station": wd.station_id,
                        "data": wd.raw_data,
                        "timestamp": wd.timestamp.isoformat(),
                        "location": wd.location
                    } for wd in all_weather_data
                ]

            # Add forecast analysis if requested
            if include_forecast:
                logger.info("Adding forecast analysis...")
                forecast_analysis = self.ai_analyzer.analyze_weather_data(
                    all_weather_data, "forecast", report_types
                )
                response["forecast"] = forecast_analysis

            logger.info("Weather analysis completed successfully")
            return response

        except Exception as e:
            logger.error(f"Weather analysis failed: {e}")
            return {
                "error": str(e), 
                "timestamp": datetime.utcnow().isoformat(),
                "stations_requested": station_ids
            }

    def get_route_weather(self, origin: str, destination: str,
                         alternate_routes: bool = False) -> Dict:
        """
        Get comprehensive weather analysis for a specific flight route.

        Args:
            origin: ICAO code of departure airport
            destination: ICAO code of arrival airport
            alternate_routes: Whether to suggest alternate routes if weather is severe

        Returns:
            Dictionary containing route-specific weather analysis
        """
        try:
            logger.info(f"Starting route weather analysis: {origin} → {destination}")

            # Calculate route waypoints
            route_points = self.route_manager.calculate_route_points(origin, destination)
            if not route_points:
                return {
                    "error": "Invalid origin or destination airport code",
                    "origin": origin,
                    "destination": destination
                }

            # Extract ICAO codes for weather analysis
            station_ids = [rp.icao_code for rp in route_points]
            logger.info(f"Route waypoints: {station_ids}")

            # Get comprehensive weather analysis for all route points
            weather_analysis = self.get_weather_analysis(
                station_ids, 
                analysis_type="route", 
                include_forecast=True
            )

            # Prepare route-specific response
            response = {
                "route_info": {
                    "origin": {
                        "icao": origin,
                        "name": route_points[0].name if route_points else "Unknown",
                        "coordinates": [route_points[0].latitude, route_points[0].longitude] if route_points else None
                    },
                    "destination": {
                        "icao": destination,
                        "name": route_points[-1].name if len(route_points) > 1 else "Unknown",
                        "coordinates": [route_points[-1].latitude, route_points[-1].longitude] if len(route_points) > 1 else None
                    },
                    "waypoints": [
                        {
                            "icao": rp.icao_code,
                            "name": rp.name,
                            "coordinates": [rp.latitude, rp.longitude]
                        } for rp in route_points
                    ],
                    "total_distance_km": self._calculate_total_route_distance(route_points)
                },
                "weather_analysis": weather_analysis,
                "timestamp": datetime.utcnow().isoformat()
            }

            # Generate alternate routes if weather is severe and requested
            if alternate_routes:
                severity = weather_analysis.get("analysis", {}).get("severity_rating", "unknown")
                if severity in ["severe", "extreme", "critical"]:
                    logger.info("Generating alternate routes due to severe weather")
                    response["alternate_routes"] = self._generate_alternate_routes(
                        origin, destination, route_points, severity
                    )

            logger.info("Route weather analysis completed successfully")
            return response

        except Exception as e:
            logger.error(f"Route weather analysis failed: {e}")
            return {
                "error": str(e),
                "origin": origin,
                "destination": destination,
                "timestamp": datetime.utcnow().isoformat()
            }

    def _calculate_bounds_from_stations(self, station_ids: List[str]) -> Dict[str, float]:
        """
        Calculate geographic bounding box for the given stations.
        Used for fetching area-based weather reports (PIREP, SIGMET, AIRMET).

        Args:
            station_ids: List of ICAO airport codes

        Returns:
            Dictionary with min/max latitude and longitude bounds
        """
        try:
            valid_airports = [
                self.route_manager.airports[icao] 
                for icao in station_ids 
                if icao in self.route_manager.airports
            ]

            if not valid_airports:
                # Default to India bounds if no valid airports found
                logger.warning("No valid airports found, using default India bounds")
                return {
                    'min_lat': 6.0,
                    'max_lat': 37.0,
                    'min_lon': 68.0,
                    'max_lon': 97.0
                }

            # Calculate actual bounds from airport locations
            latitudes = [airport.latitude for airport in valid_airports]
            longitudes = [airport.longitude for airport in valid_airports]

            # Add some padding around the bounds (about 100km in degrees)
            padding = 1.0  # Roughly 100km at equator

            bounds = {
                'min_lat': min(latitudes) - padding,
                'max_lat': max(latitudes) + padding,
                'min_lon': min(longitudes) - padding,
                'max_lon': max(longitudes) + padding
            }

            logger.debug(f"Calculated bounds for {len(valid_airports)} airports: {bounds}")
            return bounds

        except Exception as e:
            logger.error(f"Error calculating bounds: {e}")
            # Return default bounds on error
            return {
                'min_lat': 6.0,
                'max_lat': 37.0,
                'min_lon': 68.0,
                'max_lon': 97.0
            }

    def _calculate_total_route_distance(self, route_points: List[RoutePoint]) -> float:
        """
        Calculate total distance along the route.

        Args:
            route_points: List of waypoints along the route

        Returns:
            Total distance in kilometers
        """
        if len(route_points) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(route_points) - 1):
            segment_distance = geodesic(
                (route_points[i].latitude, route_points[i].longitude),
                (route_points[i + 1].latitude, route_points[i + 1].longitude)
            ).kilometers
            total_distance += segment_distance

        return round(total_distance, 1)

    def _generate_alternate_routes(self, origin: str, destination: str,
                                 current_route: List[RoutePoint], severity: str) -> List[Dict]:
        """
        Generate alternate routes when weather conditions are severe.

        Args:
            origin: Origin airport ICAO code
            destination: Destination airport ICAO code
            current_route: Current route waypoints
            severity: Weather severity level

        Returns:
            List of alternate route suggestions
        """
        alternates = []

        try:
            # This is a simplified alternate route generator
            # In production, you would use sophisticated flight planning algorithms

            # Example alternate routes for common Indian routes
            route_key = f"{origin}-{destination}"

            if route_key == "VOBL-VEGT":  # Bangalore to Bagdogra
                alternates.extend([
                    {
                        "name": "Via Delhi Route",
                        "waypoints": ['VOBL', 'VIDP', 'VEGT'],
                        "description": "Northern route via Delhi with fuel stop opportunity",
                        "estimated_distance_km": 2100,
                        "advantages": ["Major airport facilities", "Alternative if direct route has weather"]
                    },
                    {
                        "name": "Via Kolkata Route", 
                        "waypoints": ['VOBL', 'VECC', 'VEGT'],
                        "description": "Eastern route via Kolkata",
                        "estimated_distance_km": 1800,
                        "advantages": ["Shorter than Delhi route", "Good airport facilities"]
                    }
                ])

            elif route_key == "VIDP-VOMM":  # Delhi to Chennai
                alternates.extend([
                    {
                        "name": "Via Mumbai Route",
                        "waypoints": ['VIDP', 'VABB', 'VOMM'],
                        "description": "Western coastal route via Mumbai",
                        "estimated_distance_km": 2000,
                        "advantages": ["Coastal route may avoid inland weather", "Major hub facilities"]
                    },
                    {
                        "name": "Via Hyderabad Route",
                        "waypoints": ['VIDP', 'VOHS', 'VOMM'],
                        "description": "Central route via Hyderabad",
                        "estimated_distance_km": 1900,
                        "advantages": ["More direct than Mumbai route", "Good alternate airport"]
                    }
                ])

            # Add generic alternates if no specific routes defined
            if not alternates:
                # Find potential hub airports that could serve as alternates
                potential_hubs = ['VIDP', 'VABB', 'VECC', 'VOMM', 'VOBL']

                for hub in potential_hubs:
                    if hub not in [origin, destination] and hub in self.route_manager.airports:
                        hub_airport = self.route_manager.airports[hub]
                        alternates.append({
                            "name": f"Via {hub_airport.name}",
                            "waypoints": [origin, hub, destination],
                            "description": f"Alternate routing via {hub_airport.name}",
                            "estimated_distance_km": 0,  # Would calculate in production
                            "advantages": ["Major airport with full services", "Weather avoidance option"]
                        })

                        # Limit to 2-3 alternates to avoid overwhelming
                        if len(alternates) >= 3:
                            break

            # Add severity-specific recommendations
            for alternate in alternates:
                if severity == "critical":
                    alternate["recommendation"] = "Consider delaying flight until conditions improve"
                elif severity == "extreme":
                    alternate["recommendation"] = "Strongly recommend using this alternate route"
                elif severity == "severe":
                    alternate["recommendation"] = "Consider this route if direct route conditions are unacceptable"

            logger.info(f"Generated {len(alternates)} alternate routes for {origin}-{destination}")
            return alternates

        except Exception as e:
            logger.error(f"Error generating alternate routes: {e}")
            return [{
                "name": "Alternate Route Generation Failed",
                "description": "Unable to generate alternate routes due to system error",
                "recommendation": "Consult flight planning specialist for manual route planning"
            }]

# ============================================================================
# FLASK API ENDPOINTS
# ============================================================================

# Global weather service instance
weather_service = None

@app.before_request
def initialize_service():
    """
    Initialize the weather service before handling any requests.
    This ensures the service is ready when the first API call comes in.
    """
    global weather_service

    try:
        # Get Gemini API key from environment variable
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY environment variable not set")
            logger.error("Please set your Google Gemini API key: export GEMINI_API_KEY='your-key-here'")
            return

        # Initialize the weather service
        weather_service = WeatherService(gemini_api_key)
        logger.info("Weather service initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize weather service: {e}")
        return False

initialize_service()

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the service is running.

    Returns:
        JSON response with service status and timestamp
    """
    return jsonify({
        "status": "healthy" if weather_service else "initializing", 
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Aviation Weather Analysis API"
    })

@app.route('/weather/analysis', methods=['POST'])
def get_weather_analysis_endpoint():
    """
    Main weather analysis endpoint.

    Expected JSON payload:
    {
        "stations": ["VOBL", "VOMM"],           // Required: List of ICAO codes
        "report_types": ["metar", "taf"],       // Optional: Specific report types
        "analysis_type": "comprehensive",       // Optional: Analysis type
        "include_forecast": false,              // Optional: Include forecast
        "raw_data": false                       // Optional: Include raw data
    }

    Returns:
        JSON response with comprehensive weather analysis
    """
    try:
        if not weather_service:
            return jsonify({"error": "Weather service not initialized"}), 503

        # Parse request data
        try:
            data = request.get_json(force=True)
        except Exception as e:
            return jsonify({
                "error": "Invalid JSON in request body",
                "message": str(e),
                "expected_format": {
                    "stations": ["VOBL","VOMM"],
                    "analysis_type":"comprehensive"
                }
            }), 400
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "expected_format": {
                    "stations": ["VOBL","VOMM"],
                    "analysis_type":"comprehensive"
                }
                }), 400

        station_ids = data.get('stations', [])
        if not isinstance(station_ids, list) or not station_ids:
            return jsonify({
                "error": "stations must be a non-empty list of ICAO codes",
                "example": ["VOBL", "VOMM"]
            }), 400
        
        report_types = data.get('report_types', None)
        analysis_type = data.get('analysis_type', 'comprehensive')
        include_forecast = data.get('include_forecast', False)
        raw_data = data.get('raw_data', False)
        
        logger.info(f"Weather analysis request: {station_ids}, type: {analysis_type}")
        
        # Perform analysis
        result = weather_service.get_weather_analysis(
            station_ids, report_types, analysis_type, include_forecast, raw_data
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Weather analysis endpoint failed: {e}")
        return jsonify({
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "help": "Check that your request has proper JSON format"
        }), 500

@app.route('/weather/severity', methods=['POST'])
def get_weather_severity_endpoint():
    """
    Quick severity assessment endpoint.
    """
    try:
        if not weather_service:
            return jsonify({"error": "Weather service not initialized"}), 503
        
        # ✅ FIXED JSON parsing
        try:
            data = request.get_json(force=True)
        except:
            return jsonify({"error": "Invalid JSON format"}), 400
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        station_ids = data.get('stations', [])
        if not isinstance(station_ids, list) or not station_ids:
            return jsonify({"error": "stations must be a non-empty list"}), 400
        
        logger.info(f"Severity assessment request: {station_ids}")
        
        result = weather_service.get_weather_analysis(
            station_ids, analysis_type="severity"
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Severity assessment endpoint failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/weather/route', methods=['POST'])
def get_route_weather_endpoint():
    """
    Route-specific weather analysis endpoint.
    """
    try:
        if not weather_service:
            return jsonify({"error": "Weather service not initialized"}), 503
        
        # ✅ FIXED JSON parsing
        try:
            data = request.get_json(force=True)
        except:
            return jsonify({"error": "Invalid JSON format"}), 400
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        origin = data.get('origin', '').strip()
        destination = data.get('destination', '').strip()
        alternate_routes = data.get('alternate_routes', False)
        
        if not origin or not destination:
            return jsonify({
                "error": "Both origin and destination required",
                "example": {"origin": "VOBL", "destination": "VEGT"}
            }), 400
        
        logger.info(f"Route weather analysis request: {origin} → {destination}")
        
        result = weather_service.get_route_weather(
            origin, destination, alternate_routes
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Route weather analysis endpoint failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/weather/raw', methods=['POST'])
def get_raw_weather_data_endpoint():
    """
    Raw weather data endpoint without AI analysis.
    """
    try:
        if not weather_service:
            return jsonify({"error": "Weather service not initialized"}), 503
        
        # ✅ FIXED JSON parsing
        try:
            data = request.get_json(force=True)
        except:
            return jsonify({"error": "Invalid JSON format"}), 400
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        station_ids = data.get('stations', [])
        if not isinstance(station_ids, list) or not station_ids:
            return jsonify({"error": "stations must be a non-empty list"}), 400
        
        report_types = data.get('report_types', None)
        
        logger.info(f"Raw weather data request: {station_ids}")
        
        result = weather_service.get_weather_analysis(
            station_ids, report_types, analysis_type="comprehensive", raw_data=True
        )
        
        # Remove AI analysis from response, keep only raw data
        if "analysis" in result:
            del result["analysis"]
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Raw weather data endpoint failed: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    """
    Main entry point for the application.

    IMPORTANT SETUP STEPS:
    1. Set your Gemini API key: export GEMINI_API_KEY='your-api-key-here'
    2. Ensure your main.py file exists with weather data functions
    3. Modify the function imports at the top of this file to match your main.py
    4. Update the airport database in RouteManager if needed
    5. Test with a simple request to verify everything works
    """

    # Verify environment setup
    if not os.getenv('GEMINI_API_KEY'):
        print("ERROR: GEMINI_API_KEY environment variable must be set")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        print("Then run: export GEMINI_API_KEY='your-api-key-here'")
        exit(1)

    print("=" * 60)
    print("Aviation Weather Analysis Service Starting...")
    print("=" * 60)
    print("⚠️  IMPORTANT: This is a safety-critical aviation application")
    print("🔧 SETUP REQUIRED:")
    print("   1. Ensure your main.py file exists with weather data functions")
    print("   2. Update function imports at the top of this file")
    print("   3. Set GEMINI_API_KEY environment variable")
    print("   4. Test thoroughly before using for actual flight planning")
    print("=" * 60)

    # Run in production mode for safety-critical application
    # Never use debug=True in production for safety-critical systems
    app.run(
        host='0.0.0.0',    # Listen on all interfaces
        port=5000,         # Standard port
        debug=False,       # NEVER True for safety-critical applications
        threaded=True      # Handle multiple requests
    )
