import traceback
import sys

sys.path.insert(0, '.')

try:
    from src.bayesian import networks
    print("✓ Módulo importado correctamente")
except Exception as e:
    print("✗ Error al importar:")
    traceback.print_exc()
