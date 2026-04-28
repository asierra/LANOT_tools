# LANOT_tools

Suite de procesamiento y visualización de imágenes satelitales GeoTIFF para LANOT (Laboratorio Nacional de Observación de la Tierra).

## Descripción

LANOT_tools proporciona módulos integrados para el procesamiento de datos satelitales:

- **geotiff2view.py** - Convierte GeoTIFF a imágenes visualizables (PNG/JPEG) con paletas de color (CPT), composiciones RGB y transparencia NoData. Usa MapDrawer internamente para capas, logos y leyendas cuando se solicitan
- **mapdrawer.py** - Sistema de superposición de capas vectoriales, logos, leyendas, timestamps y datos GLM sobre imágenes con soporte para proyecciones GOES/EPSG
- **glm_renderer.py** - Renderizador de datos del GLM (Geostationary Lightning Mapper). Genera capas RGBA de densidad de rayos a partir de archivos NetCDF GLM, usable como módulo desde `mapdrawer` o de forma standalone
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
# Imagen con paleta de temperatura, barra de color y overlay de costa
geotiff2view datos.tif --cpt sst.cpt --alpha \
  --layer COASTLINE:white:1.0 \
  --logo-pos 3 --timestamp-pos 0 \
  --legend-pos 2 \
  -o salida.png

# Composición RGB (tres archivos separados por coma: R,G,B)
geotiff2view banda_roja.tif,banda_verde.tif,banda_azul.tif \
  --layer COASTLINE:yellow:1.0 --layer COUNTRIES:gray:0.5 \
  -o rgb_composite.png

# Agregar grilla lat/lon con etiquetas (intervalos de 5°, 10°, 15°...)
mapdrawer imagen.png \
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

### Opciones útiles de mapdrawer

| Argumento | Descripción |
|---|---|
| `--bounds ULX,ULY,LRX,LRY` | Límites geográficos de la imagen (si no vienen del GeoTIFF) |
| `--crs CRS` | Sistema de coordenadas: `goes16`, `goes18`, `epsg:4326`, Proj4... |
| `--metadata FILE` | JSON sidecar con CRS, bounds y timestamp para imágenes sin georref. |
| `--clip REGION` | Recortar a región: `ULX,ULY,LRX,LRY` o nombre predefinido (`conus`, `fulldisk`) |
| `--layer NOMBRE:COLOR:GROSOR[:labels]` | Capa vectorial (`COASTLINE`, `COUNTRIES`, `MEXSTATES`) o grilla (`grid10`) |
| `--logo-pos N` | Posición del logo (0=UL, 1=UR, 2=LL, 3=LR) |
| `--logo-size S` | Tamaño del logo (px, float ≤1.0 o porcentaje) |
| `--timestamp TEXTO` | Texto de fecha/hora a mostrar |
| `--timestamp-pos N` | Posición del timestamp (0-3); sin `--timestamp` usa fecha actual |
| `--font-size S` | Tamaño de fuente (px o porcentaje) |
| `--font-color C` | Color del texto |
| `--cpt FILE` | CPT para generar barra de color |
| `--legend-pos N` | Posición de la barra de color (0-3) |
| `--scale S` | Redimensionar imagen antes de dibujar |
| `--glm FILE [FILE ...]` | Archivos NetCDF GLM a sobreponer (densidad de rayos) |
| `--glm-color COLOR` | Color base de rayos GLM: `yellow` (default), `magenta`, `white` |
| `--verbose` | Mensajes de depuración |

### Opciones útiles de geotiff2view

| Argumento | Descripción |
|---|---|
| `--cpt FILE` | Paleta de color CPT |
| `--alpha` | Transparencia en valores NoData |
| `--backcolor` | Usar color Background del CPT para NoData |
| `--invert` | Invertir colores de la imagen o paleta |
| `--autoscale` | Escalar datos normalizados (0-1) al rango del CPT |
| `--scale S` | Redimensionar imagen (ej. `0.5` = mitad) |
| `--clip REGION` | Recortar a región: `ULX,ULY,LRX,LRY` o nombre predefinido |
| `--legend-pos N` | Posición de la barra de color (0=UL, 1=UR, 2=LL, 3=LR) |
| `--timestamp-pos N` | Posición del timestamp (0-3) |
| `--font-size S` | Tamaño de fuente (px o porcentaje, ej. `0.025`) |
| `--font-color C` | Color del texto (ej. `white`, `yellow`) |
| `--lat-south LAT` | Reserva espacio sur a partir de esa latitud para la barra de color; solo actúa si hay CPT y el borde sur de los datos está al norte de LAT |
| `--compress` | Con `--lat-south`: comprime los datos en lugar de desplazarlos (preserva el norte exacto) |
| `--save-metadata FILE` | Exporta CRS, bounds y timestamp a JSON |
| `--verbose` | Mensajes de depuración |

### Como biblioteca Python

```python
from lanot_tools import MapDrawer
from metadata import Metadata
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

```python
# Overlay de rayos GLM sobre imagen ABI
from glm_renderer import render_glm_layer
from PIL import Image

# metadata ya tiene 'crs', 'bounds' e 'image_size'
metadata['image_size'] = (img.width, img.height)
glm_layer = render_glm_layer(glm_files, metadata, base_color=(255, 255, 0))
if glm_layer:
    img = Image.alpha_composite(img.convert('RGBA'), glm_layer)
    # metadata ahora contiene 'glm_time_start' y 'glm_time_end'
    print(metadata.format_timestamp_glm())
```

## Características principales

- **Proyecciones satelitales**: GOES-16/17/18/19, EPSG y Proj4 strings via pyproj
- **Paletas de color**: CPT continuos y discretos con soporte para valores especiales (NoData, background, foreground)
- **Manejo de NoData**: Transparencia automática y máscaras en composiciones RGB
- **Capas vectoriales**: GeoPackage/Shapefile (costa, países, estados) con clipping inteligente
- **Grillas lat/lon**: Gratículas con intervalos configurables y etiquetas direccionales N/S/E/W
- **Overlay GLM**: Densidad de rayos del GLM (histograma 2D vectorizado) con color y opacidad configurables
- **Timestamp unificado ABI/GLM**: Cadena automática con rango temporal del GLM (`format_timestamp_glm()`)
- **Metadata flexible**: Extracción automática de GeoTIFF o JSON sidecar
- **Regiones predefinidas**: `conus`, `fulldisk` y cualquier región definida en `docs/recortes_coordenadas.csv` de la instalación

## Paletas CPT incluidas

| Archivo | Producto | Rango / Clases |
|---|---|---|
| `sst.cpt` | Temperatura superficial del mar | 268–310 K, continuo |
| `cld_temp_acha.cpt` | Temperatura cloud-top (AWG ACHA) | 180–295 K, 10 clases discretas |
| `cld_height_acha.cpt` | Altura cloud-top (AWG ACHA) | Discreto |
| `cld_emiss.cpt` | Emisividad cloud-top (AWG ACHA) | Continuo |
| `cloud_type.cpt` | Tipo de nube | Clases discretas |
| `phase.cpt` | Fase de nubes | 0–5: clear/water/supercooled/mixed/ice/unknown |
| `viirs_confidence_cat.cpt` | Confianza de detección de fuego (VIIRS) | Categorías discretas |
| `rainbow.cpt` | Gradiente de propósito general | 0.0–1.0, normalizado |

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

## Scripts de operación

- **`crea_vistas_viirs.sh`** — Procesamiento automático de productos VIIRS recientes (CLAVRX, ACSPO, Fire). Busca archivos `.tif` modificados en la última hora, aplica la paleta correcta para cada producto y genera imágenes JPEG con overlays de capas y logo.

- **`GLMconus_png.sh`** — Renderiza los 15 archivos GLM más recientes sobre el CONUS ABI C13. Pipeline:
  1. Genera imagen base con `hpsv gray` (GeoTIFF georreferenciado).
  2. Superpone la densidad de rayos GLM en amarillo vía `mapdrawer --glm`.
  3. Añade costas, estados, logo LANOT y fecha unificada ABI/GLM automáticamente.

  Reemplaza el flujo anterior basado en `l1bIR2tifsp.py` + `gdal_rasterize` + `tifglm2png.py`.

## Estructura del proyecto

```
LANOT_tools/
├── geotiff2view.py           # CLI: GeoTIFF → imagen
├── mapdrawer.py             # CLI: overlays sobre imágenes (incluye GLM)
├── glm_renderer.py          # Módulo/CLI: renderizado de datos GLM
├── colorpalettetable.py     # Manejo de paletas CPT
├── metadata.py              # Contenedor de metadata
├── GLMconus_png.sh          # Pipeline GLM CONUS (ABI C13 + rayos)
├── crea_vistas_viirs.sh     # Procesamiento por lote de productos VIIRS
├── *.cpt                    # Paletas de color incluidas
├── setup.py                 # Configuración pip
├── install.sh / uninstall.sh
└── README.md
```

## Licencia

GNU General Public License v3.0

## Autor

Alejandro Aguilar Sierra - LANOT