"""Asegura que la raíz del proyecto esté en sys.path al correr pytest,
para que los tests puedan hacer `from src...` sin instalar el paquete."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
