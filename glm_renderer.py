#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glm_renderer - Renderizador de datos GLM (Geostationary Lightning Mapper).

Genera una capa RGBA con la densidad de eventos de rayo a partir de archivos
NetCDF GLM del GOES, para sobreposición sobre imágenes ABI.

Uso como módulo:
    from glm_renderer import render_glm_layer
    glm_layer = render_glm_layer(glm_files, metadata)
    base_img = Image.alpha_composite(base_img.convert('RGBA'), glm_layer)

Uso standalone:
    glm_renderer.py base.tif archivo1.nc archivo2.nc ... -o salida.png

Autor: Alejandro Aguilar Sierra
LANOT - Laboratorio Nacional de Observación de la Tierra
"""

import sys
import argparse
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from netCDF4 import Dataset

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# Proyecciones GOES predefinidas (mismas que mapdrawer)
GOES_PROJECTIONS = {
    'goes16': '+proj=geos +h=35786023.0 +lon_0=-75.0 +sweep=x +a=6378137.0 +b=6356752.31414 +units=m +no_defs',
    'goes17': '+proj=geos +h=35786023.0 +lon_0=-137.0 +sweep=x +a=6378137.0 +b=6356752.31414 +units=m +no_defs',
    'goes18': '+proj=geos +h=35786023.0 +lon_0=-137.0 +sweep=x +a=6378137.0 +b=6356752.31414 +units=m +no_defs',
    'goes19': '+proj=geos +h=35786023.0 +lon_0=-75.0 +sweep=x +a=6378137.0 +b=6356752.31414 +units=m +no_defs',
}


def _resolve_crs(crs_name):
    if crs_name is None:
        return None
    return GOES_PROJECTIONS.get(crs_name.lower(), crs_name)


def render_glm_layer(glm_files, metadata, base_color=(255, 255, 0)):
    """
    Genera una capa RGBA con la densidad de eventos GLM lista para composición.

    Usa el CRS y los bounds del objeto Metadata para proyectar los eventos de
    rayo al espacio de imagen. Almacena el rango temporal de los archivos GLM
    como 'glm_time_start' y 'glm_time_end' en el objeto metadata recibido.

    Args:
        glm_files (list[str]): Lista de rutas a archivos NetCDF GLM.
        metadata: Instancia de Metadata con 'crs' y 'bounds' ya presentes.
        base_color (tuple): Color RGB base de los rayos. Default amarillo (255,255,0).

    Returns:
        PIL.Image or None: Imagen RGBA con la capa de rayos, o None si no hay datos.
    """
    if not HAS_PYPROJ:
        print("Error: pyproj es necesario para render_glm_layer.", file=sys.stderr)
        return None

    # Resolver CRS desde metadata
    crs_str = _resolve_crs(metadata.get('crs'))
    if crs_str is None:
        print("Error: metadata no contiene 'crs'.", file=sys.stderr)
        return None

    bounds = metadata.get('bounds')
    if bounds is None:
        print("Error: metadata no contiene 'bounds'.", file=sys.stderr)
        return None
    # bounds en formato rasterio: (left, bottom, right, top)
    xmin, ymin, xmax, ymax = bounds[0], bounds[1], bounds[2], bounds[3]

    # Tamaño de imagen desde metadata si está disponible, o desde 'image_size'
    img_size = metadata.get('image_size')
    if img_size:
        img_width, img_height = img_size
    else:
        img_width, img_height = 2500, 1500

    # 1. Recolectar eventos y tiempos de todos los archivos GLM
    lons_list, lats_list = [], []
    time_starts, time_ends = [], []

    for f in glm_files:
        try:
            with Dataset(f, 'r') as nc:
                event_lon = nc.variables['event_lon'][:]
                event_lat = nc.variables['event_lat'][:]
                lons_list.append(np.ma.filled(event_lon, np.nan))
                lats_list.append(np.ma.filled(event_lat, np.nan))

                # Extraer rango temporal del archivo
                for time_var in ('product_time', 'time_coverage_start',
                                 'event_time_offset'):
                    if time_var in nc.variables or hasattr(nc, time_var):
                        try:
                            if hasattr(nc, time_var):
                                t_str = getattr(nc, time_var)
                                dt = datetime.strptime(
                                    t_str[:19], "%Y-%m-%dT%H:%M:%S"
                                ).replace(tzinfo=timezone.utc)
                            else:
                                from netCDF4 import num2date
                                t_var = nc.variables[time_var]
                                dt_nc = num2date(t_var[0], units=t_var.units)
                                dt = datetime(dt_nc.year, dt_nc.month, dt_nc.day,
                                              dt_nc.hour, dt_nc.minute, dt_nc.second,
                                              tzinfo=timezone.utc)
                            time_starts.append(dt)
                            time_ends.append(dt)
                            break
                        except Exception:
                            pass
        except Exception as e:
            print(f"Advertencia: no se pudo leer {f}: {e}", file=sys.stderr)

    if not lons_list:
        print("Advertencia: no hay datos GLM en los archivos proporcionados.",
              file=sys.stderr)
        return None

    # Almacenar rango temporal en metadata
    if time_starts:
        t0 = min(time_starts)
        t1 = max(time_ends)
        metadata['glm_time_start'] = t0.strftime("%Y:%m:%d %H:%M:%S")
        metadata['glm_time_end'] = t1.strftime("%Y:%m:%d %H:%M:%S")

    all_lons = np.concatenate(lons_list)
    all_lats = np.concatenate(lats_list)

    # Filtrar NaN
    valid = np.isfinite(all_lons) & np.isfinite(all_lats)
    all_lons = all_lons[valid]
    all_lats = all_lats[valid]

    if all_lons.size == 0:
        return None

    # 2. Proyectar al CRS de la imagen
    transformer = Transformer.from_crs("epsg:4326", crs_str, always_xy=True)
    x_proj, y_proj = transformer.transform(all_lons, all_lats)

    # Filtrar puntos fuera de los límites proyectados
    in_bounds = (
        (x_proj >= xmin) & (x_proj <= xmax) &
        (y_proj >= ymin) & (y_proj <= ymax)
    )
    x_proj = x_proj[in_bounds]
    y_proj = y_proj[in_bounds]

    if x_proj.size == 0:
        return None

    # 3. Histograma 2D: densidad de eventos por píxel
    # numpy.histogram2d requiere bins monotónicamente crecientes
    x_bins = np.linspace(xmin, xmax, img_width + 1)
    y_bins = np.linspace(ymin, ymax, img_height + 1)

    density, _, _ = np.histogram2d(x_proj, y_proj, bins=[x_bins, y_bins])
    density = density.T      # Transponer: filas=Y, columnas=X
    density = np.flipud(density)  # Invertir Y: ymax queda en fila 0 (top de imagen)

    # 4. Construir capa RGBA
    rgba_array = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    mask = density > 0

    r, g, b = base_color
    rgba_array[mask, 0] = r
    rgba_array[mask, 1] = g
    rgba_array[mask, 2] = b

    # Alpha proporcional a la densidad; núcleos con alta densidad → más opaco
    alpha = np.clip(density[mask] * 40, 30, 250).astype(np.uint8)
    rgba_array[mask, 3] = alpha

    return Image.fromarray(rgba_array, 'RGBA')


# ---------------------------------------------------------------------------
# Uso standalone
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import rasterio
    from metadata import Metadata

    parser = argparse.ArgumentParser(
        description="Sobrepone datos GLM sobre una imagen base ABI.")
    parser.add_argument("base", help="Imagen base GeoTIFF o PNG.")
    parser.add_argument("glm_files", nargs='+', help="Archivos NetCDF GLM.")
    parser.add_argument("-o", "--output", default="glm_out.png",
                        help="Archivo de salida PNG (default: glm_out.png).")
    parser.add_argument("--color", default="yellow",
                        choices=["yellow", "magenta", "white"],
                        help="Color base de los rayos (default: yellow).")
    args = parser.parse_args()

    COLOR_MAP = {
        'yellow':  (255, 255, 0),
        'magenta': (255, 0, 255),
        'white':   (255, 255, 255),
    }

    # Cargar metadata desde GeoTIFF base
    try:
        with rasterio.open(args.base) as src:
            metadata = Metadata.from_rasterio(src)
            img_w, img_h = src.width, src.height
        metadata['image_size'] = (img_w, img_h)
        base_img = Image.open(args.base).convert('RGBA')
    except Exception as e:
        print(f"Error abriendo imagen base: {e}", file=sys.stderr)
        sys.exit(1)

    glm_layer = render_glm_layer(args.glm_files, metadata,
                                 base_color=COLOR_MAP[args.color])
    if glm_layer is None:
        print("No se generó capa GLM. Guardando imagen base sin cambios.")
        base_img.save(args.output)
    else:
        result = Image.alpha_composite(base_img, glm_layer)
        result.save(args.output)
        print(f"Guardado: {args.output}")
        if 'glm_time_start' in metadata:
            print(f"Rango GLM: {metadata['glm_time_start']} – {metadata['glm_time_end']}")
