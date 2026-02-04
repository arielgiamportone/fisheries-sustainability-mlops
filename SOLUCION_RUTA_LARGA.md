# SOLUCIÓN: Ruta demasiado larga en Windows

## Problema Detectado

El error real es: **"El nombre del archivo o la extensión es demasiado largo"**

La ruta actual del proyecto tiene 150 caracteres:
```
C:\Users\Ariel\Desktop\Proyectos Pesquerias ML DataScience IA\Machine_and_Deep_Learning_for_Fishing_Engineers\Deep_Learning_Causalidad_RedesBayesianas\
```

Cuando se añaden los subdirectorios del entorno virtual (.venv\Lib\site-packages\statsmodels\...) 
se supera el límite de 260 caracteres de Windows, causando que las DLLs no se carguen.

## ✅ Solución Recomendada: Mover el Proyecto

### Paso 1: Elegir una ruta más corta

Sugerencias:
- `C:\Dev\Fisheries_ML\`
- `C:\Projects\Fisheries\`
- `C:\Proyectos\Pesquerias\`

### Paso 2: Mover el proyecto

```powershell
# Opción A: Mover a C:\Proyectos\Pesquerias\
New-Item -ItemType Directory -Path "C:\Proyectos\Pesquerias" -Force
Move-Item -Path "C:\Users\Ariel\Desktop\Proyectos Pesquerias ML DataScience IA\Machine_and_Deep_Learning_for_Fishing_Engineers\Deep_Learning_Causalidad_RedesBayesianas" -Destination "C:\Proyectos\Pesquerias\DL_Bayesian"

# Opción B: Mover a C:\Dev\Fisheries_ML\
New-Item -ItemType Directory -Path "C:\Dev" -Force
Move-Item -Path "C:\Users\Ariel\Desktop\Proyectos Pesquerias ML DataScience IA\Machine_and_Deep_Learning_for_Fishing_Engineers\Deep_Learning_Causalidad_RedesBayesianas" -Destination "C:\Dev\Fisheries_ML"
```

### Paso 3: Recrear el entorno virtual en la nueva ubicación

```powershell
# Ir a la nueva ubicación
cd C:\Proyectos\Pesquerias\DL_Bayesian
# o
cd C:\Dev\Fisheries_ML

# Eliminar el entorno virtual antiguo
Remove-Item -Recurse -Force .venv

# Configurar Python 3.11
pyenv local 3.11.7

# Crear nuevo entorno
& "C:\Users\Ariel\.pyenv\pyenv-win\versions\3.11.7\python.exe" -m venv .venv

# Activar
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt

# Verificar
python test_installation.py
```

## Alternativa: Habilitar Rutas Largas en Windows (Requiere Admin)

Si prefieres mantener la ubicación actual, puedes habilitar soporte para rutas largas:

### Opción 1: Via Registro de Windows (Requiere reinicio)

```powershell
# Ejecutar PowerShell como Administrador
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Luego **reiniciar Windows**.

### Opción 2: Via Editor de Políticas de Grupo (Windows Pro)

1. Presionar Win+R
2. Escribir `gpedit.msc`
3. Navegar a: Computer Configuration > Administrative Templates > System > Filesystem
4. Habilitar "Enable Win32 long paths"
5. Reiniciar Windows

Después de habilitar rutas largas y reiniciar:
```powershell
cd "C:\Users\Ariel\Desktop\Proyectos Pesquerias ML DataScience IA\Machine_and_Deep_Learning_for_Fishing_Engineers\Deep_Learning_Causalidad_RedesBayesianas"
Remove-Item -Recurse -Force .venv
pyenv local 3.11.7
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python test_installation.py
```

## ⚡ Solución Más Rápida

**Mover el proyecto a C:\Proyectos\Pesquerias\DL_Bayesian** (toma 2-3 minutos)

```powershell
New-Item -ItemType Directory -Path "C:\Proyectos\Pesquerias" -Force
Move-Item "C:\Users\Ariel\Desktop\Proyectos Pesquerias ML DataScience IA\Machine_and_Deep_Learning_for_Fishing_Engineers\Deep_Learning_Causalidad_RedesBayesianas" "C:\Proyectos\Pesquerias\DL_Bayesian"
cd C:\Proyectos\Pesquerias\DL_Bayesian
Remove-Item -Recurse -Force .venv
pyenv local 3.11.7
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python test_installation.py
```

¡Esto debería resolver el problema completamente!
