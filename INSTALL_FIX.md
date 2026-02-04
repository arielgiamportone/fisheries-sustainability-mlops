# Solución al Problema de Compatibilidad con Python 3.13

## Problema Detectado

Durante la instalación con Python 3.13.3, se detectaron errores de compatibilidad con SciPy:
```
DLL load failed while importing _univariate_diffus
```

Este error impide que los módulos del proyecto (bayesian, causal, deep_learning) funcionen correctamente.

## ⚠️ Recomendación: Usar Python 3.10, 3.11 o 3.12

Python 3.13 es muy reciente (lanzado en octubre 2024) y algunas librerías científicas aún no tienen soporte completo.

### Opción 1: Instalar Python 3.11 (Recomendado)

#### 1. Descargar Python 3.11

Descarga Python 3.11 desde: https://www.python.org/downloads/

- **Windows**: https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe
- Durante la instalación, marca "Add Python to PATH"

#### 2. Verificar instalación

```powershell
py -3.11 --version
# Debería mostrar: Python 3.11.x
```

#### 3. Recrear entorno virtual

```powershell
# En el directorio del proyecto
cd "c:\Users\Ariel\Desktop\Proyectos Pesquerias ML DataScience IA\Machine_and_Deep_Learning_for_Fishing_Engineers\Deep_Learning_Causalidad_RedesBayesianas"

# Desactivar entorno actual (si está activo)
deactivate

# Eliminar entorno Python 3.13
Remove-Item -Recurse -Force .venv

# Crear nuevo entorno con Python 3.11
py -3.11 -m venv .venv

# Activar entorno
.\.venv\Scripts\Activate.ps1

# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
python test_installation.py
```

### Opción 2: Usar Conda (Alternativa)

Si prefieres usar Conda:

```bash
# Crear entorno con Python 3.11
conda create -n fisheries_ml python=3.11

# Activar
conda activate fisheries_ml

# Instalar dependencias
pip install -r requirements.txt
```

### Opción 3: Continuar con Python 3.13 (No Recomendado)

Si decides continuar con Python 3.13, algunas funcionalidades pueden no estar disponibles hasta que las librerías actualicen su compatibilidad.

**Limitaciones conocidas:**
- Módulos de bayesian, causal y deep_learning pueden no funcionar
- Algunos notebooks pueden tener errores
- Tests pueden fallar

**Workarounds temporales:**
1. Instalar versiones de desarrollo:
   ```bash
   pip install --pre scipy
   ```

2. Instalar desde source (avanzado):
   ```bash
   pip install --no-binary :all: scipy
   ```

## ✅ Verificación Final

Una vez recreado el entorno con Python 3.11:

```bash
python test_installation.py
```

Deberías ver:
```
✓✓✓ ¡Todas las dependencias están instaladas correctamente! ✓✓✓
✓✓✓ ¡Todos los módulos del proyecto se importan correctamente! ✓✓✓
```

## 📋 Checklist de Instalación

- [ ] Python 3.11 instalado
- [ ] Entorno virtual creado con Python 3.11
- [ ] Entorno activado (ver `(.venv)` en el prompt)
- [ ] pip actualizado: `python -m pip install --upgrade pip`
- [ ] Dependencias instaladas: `pip install -r requirements.txt`
- [ ] Test de instalación exitoso: `python test_installation.py`
- [ ] Jupyter funciona: `jupyter notebook`

## 🆘 Ayuda Adicional

Si sigues teniendo problemas:

1. **Verificar versión de Python en uso:**
   ```bash
   python --version
   ```

2. **Listar entornos de Python disponibles:**
   ```bash
   py -0  # Windows
   ```

3. **Verificar que el entorno virtual está activo:**
   - El prompt debería mostrar `(.venv)` al inicio
   - `Get-Command python` debería apuntar a `.venv\Scripts\python.exe`

4. **Limpiar caché de pip:**
   ```bash
   pip cache purge
   pip install --no-cache-dir -r requirements.txt
   ```

---

*Si necesitas más ayuda, revisa AGENTS.md para documentación completa.*
