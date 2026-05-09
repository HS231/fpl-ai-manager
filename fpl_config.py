import os
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FPL_TEAM_ID = int(os.environ.get("FPL_TEAM_ID", "7757121"))
FPL_EMAIL = os.environ.get("FPL_EMAIL", "")
FPL_PASSWORD = os.environ.get("FPL_PASSWORD", "")
FPL_COOKIE = os.environ.get("FPL_COOKIE", "")
GW_HISTORY_DEPTH = 10
FORECAST_GWS = 3
PORT = int(os.environ.get("PORT", 5000))