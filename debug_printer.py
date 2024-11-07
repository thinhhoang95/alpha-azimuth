# Load environment variables
import os 
script_dir = os.path.dirname(os.path.abspath(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(script_dir, "azimuth.env"))

# Load the debug environment variable
DEBUG = os.getenv("DEBUG") == "True"

def debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)