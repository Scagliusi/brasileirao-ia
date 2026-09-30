from pathlib import Path
from dotenv import load_dotenv
RAIZ = Path(__file__).resolve().parents[1]
load_dotenv(RAIZ / '.env')
ARTEFATOS = RAIZ / 'artifacts'
