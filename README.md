# LANOT_tools

Suite de procesamiento y visualización de imágenes satelitales GeoTIFF para LANOT (Laboratorio Nacional de Observación de la Tierra).

## Descripción

LANOT_tools proporciona cuatro módulos integrados para el procesamiento de datos satelitales:

- **geotiff2view.py** - Convierte GeoTIFF a imágenes visualizables (PNG/JPEG) con paletas de color (CPT), composiciones RGB y transparencia NoData
- **mapdrawer.py** - Sistema de superposición de capas vectoriales, logos, leyendas y timestamps sobre imágenes con soporte para proyecciones GOES/EPSG
- **colorpalettetable.py** - Manejo de paletas GMT-style CPT con gradientes continuos y discretos
- **metadata.py** - Contenedor dict-like para gestión de metadatos GeoTIFF con helpers de transformación

## Instalación

### Instalación en servidor (recomendada)

```bash
sudo ./install.sh
```

Instala en `/opt/lanot-tools/` con virtualenv aislado y crea comandos globales en `/usr/local/bin/`:
- `geotiff2view` - Conversión GeoTIFF a imagen
- `mapdrawer` - Post-procesamiento de imágenes

**Desinstalar:**
```bash
sudo ./uninstall.sh
```

### Modo desarrollo

```bash
pip install -e .
```

Los cambios en el código se reflejan inmediatamente sin reinstalar

## Uso rápido

### Procesamiento completo: GeoTIFF → Imagen con mapa

```bash
# Imagen con paleta de temperatura y overlay de costa
geotiff2view datos.tif --cpt sst.cpt --alpha \
  --layer COASTLINE:white:1.0 \
  --logo-pos 3 --timestamp-pos 0 \
  -o salida.png

# Composición RGB con capas vectoriales
geotiff2view banda_roja.tif banda_verde.tif banda_azul.tif \
  --layer COASTLINE:yellow:1.0 --layer COUNTRIES:gray:0.5 \
  --bounds conus --crs goes16 -o rgb_composite.png

# Agregar grilla lat/lon con etiquetas (intervalos de 5°, 10°, 15°...)
mapdrawer imagen.png --bounds conus \
  --layer grid5:white:1.0:labels \
  --layer COASTLINE:yellow:1.0 -o con_grilla.png
```

### Post-procesamiento de imágenes existentes

```bash
# Agregar overlays a imagen sin metadata GeoTIFF
mapdrawer imagen.png --metadata metadata.json \
  --layer COASTLINE:blue:1.0 \
  --logo-pos 3 --timestamp "2026-01-30 12:00 UTC" \
  -o output.png
```

### Como biblioteca Python

```python
from lanot_tools import MapDrawer, Metadata
from PIL import Image
import rasterio

# Cargar GeoTIFF y extraer metadata
with rasterio.open("datos.tif") as src:
    metadata = Metadata.from_rasterio(src)
    img = Image.open("datos.tif")

# Configurar mapa con proyección
mapper = MapDrawer(target_crs='goes16')
mapper.set_image(img)

# Usar bounds del metadata
ulx, uly, lrx, lry = metadata.get_mapdrawer_bounds()
mapper.set_bounds(ulx, uly, lrx, lry)

# Dibujar capas y decoraciones
mapper.draw_layer('COASTLINE', 'white', 1.0)
mapper.draw_grid(interval=10, color='gray', width=0.5, labels=True)
mapper.draw_logo(position=3)
mapper.draw_fecha(metadata.get('timestamp'), position=0)

img.save("output.png")
```

## Características principales

- **Proyecciones satelitales**: GOES-16/17/18/19, EPSG y Proj4 strings via pyproj
- **Paletas de color**: CPT continuos y discretos con soporte para valores especiales (NoData, background, foreground)
- **Manejo de NoData**: Transparencia automática y máscaras en composiciones RGB
- **Capas vectoriales**: GeoPackage/Shapefile (costa, países, estados) con clipping inteligente
- **Grillas lat/lon**: Gratículas con intervalos configurables y etiquetas direccionales N/S/E/W
- **Metadata flexible**: Extracción automática de GeoTIFF o JSON sidecar
- **Regiones predefinidas**: `conus`, `fulldisk` para recortes rápidos

## Paletas CPT incluidas

- `sst.cpt` - Temperatura superficial del mar (268-310K, continuo)
- `cld_temp_acha.cpt` - Temperatura cloud-top (180-295K, 10 clases discretas)
- `phase.cpt` - Fase de nubes (0-5: clear/water/supercooled/mixed/ice/unknown)
- `rainbow.cpt` - Gradiente normalizado (0.0-1.0)

## Requisitos

- Python >= 3.8
- **Requeridos**: Pillow, fiona, pyproj, numpy
- **Opcionales**: rasterio (lectura GeoTIFF con metadata, altamente recomendado)

## Recursos del sistema

Después de la instalación se crean:
- Ejecutables: `/opt/lanot-tools/venv/`
- Comandos globales: `/usr/local/bin/{geotiff2view,mapdrawer}`
- Paletas CPT: `/usr/local/share/lanot/colortables/`
- Capas vectoriales: `/usr/local/share/lanot/gpkg/`
- Logos: `/usr/local/share/lanot/logos/`

## Estructura del proyecto

```
LANOT_tools/
├── geotiff2view.py           # CLI: GeoTIFF → imagen
├── mapdrawer.py             # CLI: overlays sobre imágenes
├── colorpalettetable.py     # Manejo de paletas CPT
├── metadata.py              # Contenedor de metadata
├── *.cpt                    # Paletas de color incluidas
├── setup.py                 # Configuración pip
├── install.sh / uninstall.sh
└── README.md
```

## Licencia

GNU General Public License v3.0

## Autor

Alejandro Aguilar Sierra - LANOT