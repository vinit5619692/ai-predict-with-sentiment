#create virtual env
#py -m venv venv  -- creat folder venv
#venv\Scripts\activate -- this will activate venv (venv) C:\AI models git\...
#deactivate -- come out of virtual env
#pip install requests pandas yfinance torch scikit-learn
#pip install -r requirements.txt
#python featureStore.py --full
#python Model.py


# config.py — API keys and settings

NEWSDATA_API_KEY    = "pub_a1dd29b516a54ec59a971de6d4167070"
GUARDIAN_API_KEY    = "06cd6145-067b-40a5-90a8-0117e9931859"
EIA_API_KEY         = "8sfwtgigsRDypxWUY0nefoPg9siOeUQqlz8wZCR8"
NEWSAPI_KEY         = ""   # optional: newsapi.org (geo fallback)
ALPHA_VANTAGE_KEY   = "XR6ZTKRB7SGH1VKS"
FRED_STLOUISFED_KEY = "43e1a1732849eb9b37d37e6219cda329"

# Path to one-time CSV seed for crude oil
CRUDE_CSV_PATH = r"Crude Oil WTI Futures Historical Data.csv"


STOCK = "TATASTEEL.NS"

# How many days of history to seed
HISTORY_DAYS = 300

# SQLite DB path
DB_PATH = r"C:\AI models git\sqllite\tatasteel.db"

# News keywords for geopolitics/policy relevance
GEO_KEYWORDS   = ["russia ukraine", "china steel", "middle east", "iran sanctions", "war", "conflict"]
POLICY_KEYWORDS = ["government policy", "india budget", "PLI scheme", "steel import", "GST", "infrastructure"]
TATA_KEYWORDS   = ["tata steel", "TATASTEEL", "steel india", "steel prices"]