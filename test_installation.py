#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para verificar la instalación de dependencias del proyecto.
"""

import sys

def test_imports():
    """Verifica que todas las dependencias principales estén instaladas."""
    
    print("="*70)
    print("VERIFICACIÓN DE INSTALACIÓN - Deep Learning y Redes Bayesianas")
    print("="*70)
    
    print(f"\n✓ Python: {sys.version}")
    
    # Lista de paquetes a verificar
    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('pgmpy', 'PGMpy'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('networkx', 'NetworkX'),
        ('sklearn', 'Scikit-learn'),
        ('torch', 'PyTorch'),
        ('tqdm', 'tqdm'),
        ('yaml', 'PyYAML'),
    ]
    
    print("\nDependencias instaladas:")
    print("-" * 70)
    
    all_ok = True
    for module_name, display_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {display_name:20s} : {version}")
        except ImportError as e:
            print(f"✗ {display_name:20s} : NO INSTALADO")
            all_ok = False
    
    print("-" * 70)
    
    if all_ok:
        print("\n✓✓✓ ¡Todas las dependencias están instaladas correctamente! ✓✓✓")
    else:
        print("\n✗ Algunas dependencias faltan. Ejecute:")
        print("  pip install -r requirements.txt")
        return False
    
    # Verificar módulos del proyecto
    print("\nMódulos del proyecto:")
    print("-" * 70)
    
    project_modules = [
        'data.loaders',
        'src.bayesian.networks',
        'src.bayesian.validation',
        'src.causal.dag',
        'src.causal.interventions',
        'src.deep_learning.models',
        'src.deep_learning.training',
        'src.visualization.plots',
    ]
    
    project_ok = True
    for module_name in project_modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name}: {str(e)[:50]}")
            project_ok = False
    
    print("-" * 70)
    
    if project_ok:
        print("\n✓✓✓ ¡Todos los módulos del proyecto se importan correctamente! ✓✓✓")
    else:
        print("\n⚠ Algunos módulos del proyecto tienen problemas.")
    
    print("\n" + "="*70)
    print("Verificación completada.")
    print("="*70)
    
    return all_ok and project_ok


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
