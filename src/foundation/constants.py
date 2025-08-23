"""
System constants that are NOT configuration
These are protocol standards and mathematical constants only
NO configuration values should be here - those go in environment/config
"""

# HTTP Protocol Standards (not configuration)
HTTP_HEADER_CORRELATION_ID = "X-Correlation-ID"
HTTP_HEADER_REQUEST_ID = "X-Request-ID"
HTTP_HEADER_TRACE_ID = "X-Trace-ID"

# Standard HTTP Status Codes (protocol constants)
HTTP_OK = 200
HTTP_CREATED = 201
HTTP_ACCEPTED = 202
HTTP_NO_CONTENT = 204
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_METHOD_NOT_ALLOWED = 405
HTTP_CONFLICT = 409
HTTP_UNPROCESSABLE_ENTITY = 422
HTTP_TOO_MANY_REQUESTS = 429
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_BAD_GATEWAY = 502
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_GATEWAY_TIMEOUT = 504

# Time Constants (mathematical, not configuration)
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR
SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY
MILLISECONDS_PER_SECOND = 1000
MICROSECONDS_PER_SECOND = 1000000
NANOSECONDS_PER_SECOND = 1000000000

# Mathematical Constants
PI = 3.141592653589793
E = 2.718281828459045
GOLDEN_RATIO = 1.618033988749895

# Market Hours (standard market times, not configuration)
# These are NYSE standard hours, not configurable
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
PRE_MARKET_OPEN_HOUR = 4
PRE_MARKET_OPEN_MINUTE = 0
AFTER_MARKET_CLOSE_HOUR = 20
AFTER_MARKET_CLOSE_MINUTE = 0

# Trading Days (standard, not configuration)
TRADING_DAYS_PER_YEAR = 252

# IMPORTANT: DO NOT ADD CONFIGURATION HERE
# Configuration values such as:
# - Timeouts
# - Limits
# - Thresholds
# - Ports
# - Hosts
# - API keys
# - Database settings
# - File paths
# Should ALL come from environment variables or config files