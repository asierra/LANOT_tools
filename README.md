# LANOT_tools — mapdrawer

Herramientas y utilidades comunes para LANOT (Laboratorio Nacional de Observación de la Tierra)

## Instalación

### Instalación en servidor (accesible para todos los usuarios)

Esta es la forma recomendada para instalar en un servidor Linux donde múltiples usuarios necesiten acceso:

```bash
# 1. Crear directorio para la instalación
sudo mkdir -p /opt/lanot-tools

# 2. Copiar el código fuente
sudo cp -r /ruta/al/LANOT_tools /opt/lanot-tools/src

# 3. Crear virtualenv
sudo python3 -m venv /opt/lanot-tools/venv

# 4. Instalar el paquete en el virtualenv
sudo /opt/lanot-tools/venv/bin/pip install --upgrade pip
sudo /opt/lanot-tools/venv/bin/pip install /opt/lanot-tools/src

# 5. Crear el wrapper script en /usr/local/bin
sudo tee /usr/local/bin/mapdrawer << 'EOF'
#!/bin/bash
# Wrapper para mapdrawer - ejecuta desde virtualenv
VENV_DIR="/opt/lanot-tools/venv"
exec "${VENV_DIR}/bin/mapdrawer" "$@"
EOF

# 6. Dar permisos de ejecución
sudo chmod +x /usr/local/bin/mapdrawer

# 7. Verificar instalación
mapdrawer --help
```

**Ventajas:**
- ✅ Accesible para todos los usuarios del sistema
- ✅ Dependencias aisladas en virtualenv
- ✅ No contamina el Python del sistema
- ✅ Fácil de actualizar y desinstalar

**Para actualizar después:**
```bash
sudo cp -r /ruta/al/LANOT_tools /opt/lanot-tools/src
sudo /opt/lanot-tools/venv/bin/pip install --upgrade --force-reinstall /opt/lanot-tools/src
```

### Instalación en modo desarrollo (para desarrollo activo)
```bash
cd /ruta/al/LANOT_tools
pip install -e .
```

### Instalación para usuario individual
```bash
cd /ruta/al/LANOT_tools
pip install --user .
```
Asegúrate de que `~/.local/bin` esté en tu PATH.

### Instalación desde GitHub (si el repositorio es público)
```bash
pip install git+https://github.com/asierra/LANOT_tools.git
```

## Uso

### Como comando de terminal
Después de la instalación, puedes usar `mapdrawer` directamente desde la terminal:

```bash
# Ejemplo básico con bounds
mapdrawer imagen.png --bounds -120 35 -80 15 --layer COASTLINE:blue:0.5

# Usando región predefinida
mapdrawer imagen.png --recorte conus --layer COASTLINE:yellow:0.5 --logo-pos 3

# Con proyección GOES y leyenda
mapdrawer imagen.png --recorte fulldisk --crs goes16 \
    --layer COASTLINE:white:1.0 \
    --layer COUNTRIES:gray:0.5 \
    --logo-pos 3 --cpt leyenda.cpt \
    --output salida.png
```

### Como módulo Python
```python
from lanot_tools import MapDrawer
from PIL import Image

# Abrir imagen
img = Image.open("mi_imagen.png")

# Crear drawer con proyección GOES-16
mapper = MapDrawer(target_crs='goes16')
mapper.set_image(img)

# Establecer límites (o usar región predefinida)
mapper.load_bounds_from_csv('fulldisk')

# Dibujar capas
mapper.draw_layer('COASTLINE', color='white', width=1.0)
mapper.draw_layer('COUNTRIES', color='gray', width=0.5)

# Agregar logo y fecha
mapper.draw_logo(logosize=128, position=3)

from datetime import datetime
mapper.draw_fecha(datetime.utcnow(), position=2, fontsize=20)

# Guardar
img.save("salida.png")
```

## Características

- **Proyecciones**: Soporta proyecciones GOES (goes16, goes17, goes18, goes19) y cualquier proyección EPSG vía pyproj
- **Regiones predefinidas**: CONUS, Full Disk, y regiones personalizadas vía CSV
- **Capas vectoriales**: Líneas costeras, países, estados de México
- **Decoraciones**: Logo LANOT, fecha/hora, leyendas desde archivos CPT
- **Optimizado**: Caché de shapefiles, clipping inteligente para rendimiento

## Opciones de línea de comandos

```
mapdrawer <imagen_entrada> [opciones]

Límites geográficos:
  --bounds ULX ULY LRX LRY    Límites manualmente (lon/lat)
  --recorte NOMBRE            Región predefinida (conus, fulldisk, etc.)

Proyección:
  --crs SISTEMA               Sistema de coordenadas (goes16, epsg:4326, etc.)

Capas:
  --layer NOMBRE:COLOR:GROSOR Dibujar capa (ej: COASTLINE:blue:0.5)
                              Puede usarse múltiples veces

Logo:
  --logo-pos {0,1,2,3}        Posición del logo (0=UL, 1=UR, 2=LL, 3=LR)
  --logo-size PIXELES         Tamaño del logo

Fecha:
  --timestamp TEXTO           Texto de fecha/hora
  --timestamp-pos {0,1,2,3}   Posición de la fecha
  --font-size TAMAÑO          Tamaño de fuente
  --font-color COLOR          Color de fuente

Leyenda:
  --cpt ARCHIVO               Archivo CPT para generar leyenda

Salida:
  --output ARCHIVO            Archivo de salida (default: sobreescribe entrada)
```

## Requisitos

- Python >= 3.8
- Pillow
- aggdraw
- pyshp
- pyproj
- numpy

## Estructura del Proyecto

```
LANOT_tools/
├── __init__.py              # Exporta MapDrawer
├── mapdrawer.py            # Clase principal
├── setup.py                # Configuración de instalación
├── requirements.txt        # Dependencias
└── README.md              # Este archivo
```

## Licencia

GNU General Public License v3.0

## Autor

Abraham Sierra