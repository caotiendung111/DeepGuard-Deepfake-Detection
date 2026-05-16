
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Importing app from api.main...")
from api.main import app
print("App imported!")
