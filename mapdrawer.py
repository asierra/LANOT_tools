#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mapdrawer - Herramienta para dibujar mapas y decoraciones en imágenes satelitales.

Permite sobreponer capas vectoriales (costas, fronteras), logos, timestamps y leyendas
sobre imágenes. Soporta proyecciones GOES y otras proyecciones vía pyproj.

Autor: Alejandro Aguilar Sierra
LANOT - Laboratorio Nacional de Observación de la Tierra
"""

import os
import csv
import sys
import argparse
import json
from PIL import Image, ImageDraw, ImageFont
import fiona
import math
import numpy as np

from metadata import Metadata

# Desactivar límite de píxeles para imágenes satelitales grandes
Image.MAX_IMAGE_PIXELS = None

# Intentamos importar rasterio para lectura de metadatos GeoTIFF
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Intentamos importar pyproj. Si no existe, el programa sigue funcionando en modo lineal.
try:
    from pyproj import Transformer
    from pyproj.enums import TransformDirection
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

VERBOSE = False


def debug_msg(msg):
    if VERBOSE:
        print(f"[DEBUG] {msg}", file=sys.stderr)


# Proyecciones GOES predefinidas
GOES_PROJECTIONS = {
    'goes16': '+proj=geos +h=35786023.0 +lon_0=-75.0 +sweep=x +a=6378137.0 +b=6356752.31414 +units=m +no_defs',
    'goes17': '+proj=geos +h=35786023.0 +lon_0=-137.0 +sweep=x +a=6378137.0 +b=6356752.31414 +units=m +no_defs',
    'goes18': '+proj=geos +h=35786023.0 +lon_0=-137.0 +sweep=x +a=6378137.0 +b=6356752.31414 +units=m +no_defs',
    'goes19': '+proj=geos +h=35786023.0 +lon_0=-75.0 +sweep=x +a=6378137.0 +b=6356752.31414 +units=m +no_defs',
}

# Regiones predefinidas (ulx, uly, lrx, lry)
# Valores típicos para GOES-16 (East)
PREDEFINED_REGIONS = {
    'conus': (-152.1093, 56.76145, -52.94688, 14.57134),
    # Para Full Disk, definimos lat/lon aproximados, pero MapDrawer
    # usará límites en metros hardcoded si detecta esta clave y proyección GOES.
    'fulldisk': (-156.2995, 81.3282, 6.2995, -81.3282),
    'fd': (-156.2995, 81.3282, 6.2995, -81.3282),
}

# Límites en metros para Full Disk GOES-R (ABI)
# h * 0.151872 radians
GOES_FD_EXTENT_METERS = {
    'min_x': -5434894.885056,
    'max_y': 5434894.885056,
    'width': 10869789.770112,
    'height': -10869789.770112  # Negativo porque y_min - y_max
}

# Límites en metros para CONUS GOES-16 (East)
# x_rad: [-0.101360, 0.038640], y_rad: [0.128240, 0.044240]
# h = 35786023.0
GOES_CONUS_EXTENT_METERS = {
    'min_x': -3627271.29,
    'max_y': 4589200.59,
    'width': 5010043.22,
    'height': -3006025.93
}

# Directorio global de recursos (instalación estándar)
GLOBAL_LANOT_DIR = "/usr/local/share/lanot"


class MapDrawer:
    def __init__(self, lanot_dir=GLOBAL_LANOT_DIR, target_crs=None):
        """
        Inicializa el dibujante de mapas.

        Args:
            lanot_dir (str): Ruta base de los recursos (shapefiles/logos).
            target_crs (str, opcional): Código EPSG (ej. 'epsg:3857'), string Proj4, 
                                        o clave corta GOES ('goes16', 'goes17', 'goes18').
                                        Si es None, usa proyección lineal (Plate Carrée).
        """
        self.lanot_dir = lanot_dir
        self.image = None
        self._shp_cache = {}  # Caché para no re-leer shapefiles del disco
        # Mapeo interno de capas simbólicas -> rutas relativas de archivos vectoriales
        # Se puede extender con add_layer(). Las claves se manejan en mayúsculas.
        # Soporta GeoPackage (.gpkg) y Shapefiles (.shp)
        self._layers = {
            'COASTLINE': 'gpkg/costas_mundo_10m.gpkg',
            'COUNTRIES': 'gpkg/paises_fronteras_10m.gpkg',
            'MEXSTATES': 'gpkg/mexico_estados.gpkg'
            # Alternativa con más detalle: 'shapefiles/dest_2015gwLines.shp'
        }

        # Configuración de proyección
        self.use_proj = False
        self.transformer = None

        if target_crs:
            debug_msg(f"MapDrawer init con target_crs='{target_crs}'")
            # Resolver claves cortas de GOES
            crs_lower = target_crs.lower()
            if crs_lower in GOES_PROJECTIONS:
                resolved_crs = GOES_PROJECTIONS[crs_lower]
                print(f"Info: Resolviendo '{target_crs}' a proyección GOES.")
            else:
                resolved_crs = target_crs

            if HAS_PYPROJ:
                # 'always_xy=True' asegura el orden (lon, lat)
                self.transformer = Transformer.from_crs(
                    "epsg:4326", resolved_crs, always_xy=True)
                self.use_proj = True
                print(f"Info: Usando proyección {target_crs} vía pyproj.")
            else:
                print(
                    "Advertencia: pyproj no está instalado. Se usará proyección lineal simple.")

        # Coordenadas (se inicializan en 0)
        self.bounds = {'ulx': 0., 'uly': 0., 'lrx': 0., 'lry': 0.}

    def _load_font(self, size):
        """Carga una fuente TrueType del sistema con el tamaño especificado.
        
        Args:
            size (int): Tamaño de la fuente en puntos.
            
        Returns:
            ImageFont: Fuente cargada o fuente por defecto si falla.
        """
        font_paths = [
            '/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf',
            '/usr/share/fonts/jetbrains-mono-fonts/JetBrainsMono-Regular.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
            '/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf',
        ]
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, size)
            except (OSError, IOError, RuntimeError):
                continue
        
        # Fallback
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except IOError:
            return ImageFont.load_default()

    def set_image(self, pil_image):
        self.image = pil_image

    def set_bounds(self, ulx, uly, lrx, lry):
        """
        Define los límites geográficos (Lon/Lat WGS84) de la imagen.
        Si se usa pyproj, calcula también los límites en el plano proyectado.
        """
        debug_msg(f"set_bounds: ulx={ulx}, uly={uly}, lrx={lrx}, lry={lry}")
        self.bounds['ulx'] = ulx
        self.bounds['uly'] = uly
        self.bounds['lrx'] = lrx
        self.bounds['lry'] = lry

        if self.use_proj:
            # Para proyecciones curvas (como GOES), muestrear el perímetro
            # para obtener los límites correctos en el espacio proyectado
            n_samples = 50
            edge_lon = []
            edge_lat = []

            # Borde superior e inferior
            lon_samples = np.linspace(ulx, lrx, n_samples)
            edge_lon.extend(lon_samples)
            edge_lat.extend([uly] * n_samples)
            edge_lon.extend(lon_samples)
            edge_lat.extend([lry] * n_samples)

            # Borde izquierdo y derecho
            lat_samples = np.linspace(lry, uly, n_samples)
            edge_lon.extend([ulx] * n_samples)
            edge_lat.extend(lat_samples)
            edge_lon.extend([lrx] * n_samples)
            edge_lat.extend(lat_samples)

            # Transformar todos los puntos del perímetro
            x_vals, y_vals = self.transformer.transform(edge_lon, edge_lat)

            # Filtrar valores inválidos
            x_vals = np.array(x_vals)
            y_vals = np.array(y_vals)
            valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)

            if np.any(valid_mask):
                x_valid = x_vals[valid_mask]
                y_valid = y_vals[valid_mask]

                x_min = np.min(x_valid)
                x_max = np.max(x_valid)
                y_min = np.min(y_valid)
                y_max = np.max(y_valid)
            else:
                # Fallback: usar solo las esquinas
                x_min, y_max = self.transformer.transform(ulx, uly)
                x_max, y_min = self.transformer.transform(lrx, lry)

            self.proj_bounds = {
                'min_x': x_min, 'max_y': y_max,
                'width': x_max - x_min,
                'height': y_min - y_max  # Note: y_min suele ser menor que y_max
            }
            debug_msg(f"Límites proyectados calculados: {self.proj_bounds}")

    def set_projected_bounds(self, min_x, min_y, max_x, max_y):
        """
        Define los límites en coordenadas proyectadas (metros, etc).
        Calcula inversamente los límites Lat/Lon para optimización de clipping.
        """
        self.proj_bounds = {
            'min_x': min_x,
            'max_y': max_y,
            'width': max_x - min_x,
            'height': min_y - max_y  # Negativo si max_y > min_y
        }
        debug_msg(
            f"Límites proyectados establecidos directamente: {self.proj_bounds}")

        if self.use_proj and self.transformer:
            try:
                # Calcular bounds lat/lon aproximados para el clipping
                # Usar esquinas y centro
                xs = [min_x, max_x, max_x, min_x, (min_x+max_x)/2]
                ys = [min_y, min_y, max_y, max_y, (min_y+max_y)/2]

                lons, lats = self.transformer.transform(
                    xs, ys, direction=TransformDirection.INVERSE)

                # Filtrar valores finitos
                valid_lons = [l for l in lons if math.isfinite(l)]
                valid_lats = [l for l in lats if math.isfinite(l)]

                if valid_lons and valid_lats:
                    ulx, lrx = min(valid_lons), max(valid_lons)
                    lry, uly = min(valid_lats), max(valid_lats)

                    # Si los límites colapsan (ej. solo el centro es válido en Full Disk),
                    # usar fallback global.
                    if (lrx - ulx) < 1e-5 and (uly - lry) < 1e-5:
                        debug_msg(
                            "Límites estimados colapsados (posible Full Disk). Usando fallback global.")
                        self.bounds = {'ulx': -180,
                                       'uly': 90, 'lrx': 180, 'lry': -90}
                    else:
                        self.bounds['ulx'] = ulx
                        self.bounds['lrx'] = lrx
                        self.bounds['lry'] = lry
                        self.bounds['uly'] = uly
                    debug_msg(f"Límites Lat/Lon estimados: {self.bounds}")
                else:
                    # Fallback si no hay puntos válidos
                    self.bounds = {'ulx': -180,
                                   'uly': 90, 'lrx': 180, 'lry': -90}
            except Exception as e:
                debug_msg(
                    f"Advertencia: No se pudieron calcular límites Lat/Lon inversos: {e}")
                # Fallback a todo el mundo
                self.bounds = {'ulx': -180, 'uly': 90, 'lrx': 180, 'lry': -90}

    def get_region_bounds(self, recorte_name, csv_path=None):
        """Obtiene los límites (ulx, uly, lrx, lry) de una región por nombre."""
        name_lower = recorte_name.lower()

        if name_lower in PREDEFINED_REGIONS:
            return PREDEFINED_REGIONS[name_lower]

        if csv_path is None:
            csv_path = os.path.join(
                self.lanot_dir, "docs/recortes_coordenadas.csv")

        try:
            if os.path.exists(csv_path):
                with open(csv_path, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if row[0] == recorte_name:
                            return [float(i) for i in row[2:]]
        except Exception:
            pass
        return None

    def load_bounds_from_csv(self, recorte_name, csv_path=None):
        vals = self.get_region_bounds(recorte_name, csv_path)

        if not vals:
            print(f"Advertencia: Recorte '{recorte_name}' no encontrado.")
            return False

        name_lower = recorte_name.lower()

        # Verificar casos especiales de proyección (GOES FD/CONUS)
        if self.use_proj:
            if name_lower in ('fd', 'fulldisk'):
                self.proj_bounds = GOES_FD_EXTENT_METERS.copy()
                self.bounds['ulx'], self.bounds['uly'], self.bounds['lrx'], self.bounds['lry'] = vals
                print(
                    f"Info: Usando límites Full Disk GOES en metros: {self.proj_bounds}")
                return True

            if name_lower == 'conus':
                self.proj_bounds = GOES_CONUS_EXTENT_METERS.copy()
                self.bounds['ulx'], self.bounds['uly'], self.bounds['lrx'], self.bounds['lry'] = vals
                print(
                    f"Info: Usando límites CONUS GOES en metros: {self.proj_bounds}")
                return True

        self.set_bounds(*vals)
        print(f"Info: Usando región '{recorte_name}': {vals}")
        return True

    def _geo2pixel(self, lon, lat):
        """Convierte lon/lat a u/v (píxeles) usando la estrategia activa.
           Devuelve None si la proyección falla (infinito/nan).
        """
        w = self.image.width
        h = self.image.height

        if self.use_proj:
            # 1. Proyectar punto (Lat/Lon -> Metros)
            x_p, y_p = self.transformer.transform(lon, lat)

            # Verificar si el resultado es finito
            if not (math.isfinite(x_p) and math.isfinite(y_p)):
                return None

            # 2. Interpolar en el plano proyectado
            pb = self.proj_bounds
            if pb['width'] == 0 or pb['height'] == 0:
                return 0, 0

            u = int(w * (x_p - pb['min_x']) / pb['width'])
            # Coordenada Y de imagen crece hacia abajo, coordenadas geográficas/proyectadas crecen hacia arriba
            v = int(h * (y_p - pb['max_y']) / pb['height'])
            return u, v

        else:
            # Estrategia Original (Lineal / Plate Carrée)
            b = self.bounds
            width_span = b['lrx'] - b['ulx']
            height_span = b['uly'] - b['lry']  # uly suele ser mayor que lry

            if width_span == 0 or height_span == 0:
                return 0, 0

            u = int(w * (lon - b['ulx']) / width_span)
            v = int(h * (b['uly'] - lat) / height_span)
            return u, v

    def crop(self, ulx, uly, lrx, lry):
        """Recorta la imagen a los límites geográficos especificados."""
        if self.image is None:
            return

        debug_msg(f"Calculando recorte para: {ulx}, {uly}, {lrx}, {lry}")

        # Convertir coordenadas geográficas a píxeles
        res1 = self._geo2pixel(ulx, uly)
        res2 = self._geo2pixel(lrx, lry)

        if res1 is None or res2 is None:
            print(
                "Error: Coordenadas de recorte fuera de proyección o inválidas.", file=sys.stderr)
            return

        u1, v1 = res1
        u2, v2 = res2

        # Asegurar orden correcto (min, max)
        min_u, max_u = sorted([u1, u2])
        min_v, max_v = sorted([v1, v2])

        # Validar límites de imagen y recortar
        min_u = max(0, min_u)
        min_v = max(0, min_v)
        max_u = min(self.image.width, max_u)
        max_v = min(self.image.height, max_v)

        if max_u > min_u and max_v > min_v:
            debug_msg(
                f"Recortando imagen a píxeles: {min_u}, {min_v}, {max_u}, {max_v}")
            self.image = self.image.crop((min_u, min_v, max_u, max_v))
            # Actualizar bounds a los nuevos límites
            self.set_bounds(ulx, uly, lrx, lry)
        else:
            print("Error: Recorte resultante vacío o fuera de la imagen.",
                  file=sys.stderr)

    def draw_shapefile(self, vector_rel_path, color='yellow', width=0.5, layer=None):
        """Dibuja un archivo vectorial (shapefile o geopackage) sobre la imagen.

        Args:
            vector_rel_path (str): Ruta relativa al archivo vectorial desde lanot_dir.
            color (str): Color de las líneas.
            width (float): Grosor de las líneas.
            layer (str, opcional): Nombre de la capa (solo para GeoPackage multi-capa).
        """
        debug_msg(f"draw_shapefile: {vector_rel_path} (layer={layer})")
        if self.image is None:
            return

        full_path = os.path.join(self.lanot_dir, vector_rel_path)

        # Cache: guardar como tupla (path, layer) para GeoPackage
        cache_key = (full_path, layer) if layer else full_path

        if cache_key not in self._shp_cache:
            try:
                debug_msg(f"Cargando vectorial: {full_path}")
                # Fiona abre con context manager, pero guardaremos features en cache
                with fiona.open(full_path, layer=layer) as src:
                    # Guardar todas las geometrías en memoria
                    self._shp_cache[cache_key] = [feature for feature in src]
            except Exception as e:
                print(f"Error leyendo archivo vectorial {full_path}: {e}")
                return

        features = self._shp_cache[cache_key]
        draw = ImageDraw.Draw(self.image)
        debug_msg(f"Procesando {len(features)} geometrías...")

        b = self.bounds
        margin = 5.0

        for feature in features:
            geom = feature['geometry']
            if not geom:
                continue

            # Obtener bbox de la geometría si existe
            # Fiona usa bounds en feature, pero no siempre está disponible
            # Mejor procesar las coordenadas directamente

            geom_type = geom['type']
            coords = geom['coordinates']

            # Manejar diferentes tipos de geometría
            if geom_type in ['LineString', 'MultiLineString']:
                # Convertir a lista de LineStrings
                if geom_type == 'LineString':
                    linestrings = [coords]
                else:  # MultiLineString
                    linestrings = coords

                for linestring in linestrings:
                    if not linestring:
                        continue

                    pixel_coords = []

                    for lon, lat in linestring:
                        # Clipping suave
                        if (b['ulx'] - margin < lon < b['lrx'] + margin and
                                b['lry'] - margin < lat < b['uly'] + margin):

                            res = self._geo2pixel(lon, lat)
                            if res is None:
                                if len(pixel_coords) >= 4:
                                    draw.line(pixel_coords, fill=color,
                                              width=max(1, int(width)))
                                pixel_coords = []
                                continue

                            u, v = res
                            pixel_coords.extend((u, v))
                        else:
                            if len(pixel_coords) >= 4:
                                draw.line(pixel_coords, fill=color,
                                          width=max(1, int(width)))
                            pixel_coords = []

                    # Dibujar remanente
                    if len(pixel_coords) >= 4:
                        draw.line(pixel_coords, fill=color,
                                  width=max(1, int(width)))

            elif geom_type in ['Polygon', 'MultiPolygon']:
                # Para polígonos, dibujar solo los bordes (anillos exteriores)
                if geom_type == 'Polygon':
                    polygons = [coords]
                else:  # MultiPolygon
                    polygons = coords

                for polygon in polygons:
                    # polygon[0] es el anillo exterior, polygon[1:] son huecos
                    for ring in polygon:
                        pixel_coords = []

                        for lon, lat in ring:
                            if (b['ulx'] - margin < lon < b['lrx'] + margin and
                                    b['lry'] - margin < lat < b['uly'] + margin):

                                res = self._geo2pixel(lon, lat)
                                if res is None:
                                    if len(pixel_coords) >= 4:
                                        draw.line(pixel_coords, fill=color,
                                                  width=max(1, int(width)))
                                    pixel_coords = []
                                    continue

                                u, v = res
                                pixel_coords.extend((u, v))
                            else:
                                if len(pixel_coords) >= 4:
                                    draw.line(pixel_coords, fill=color,
                                              width=max(1, int(width)))
                                pixel_coords = []

                        if len(pixel_coords) >= 4:
                            draw.line(pixel_coords, fill=color,
                                      width=max(1, int(width)))

    # --- Nueva API basada en nombres de capa ---
    def add_layer(self, key, rel_path):
        """Agrega o actualiza una capa simbólica.

        Args:
            key (str): Nombre simbólico (ej: 'RIVERS'). Se normaliza a mayúsculas.
            rel_path (str): Ruta relativa al directorio lanot_dir.
        """
        self._layers[key.upper()] = rel_path

    def list_layers(self):
        """Devuelve lista de claves de capas disponibles."""
        return sorted(self._layers.keys())

    def draw_layer(self, key, color='yellow', width=0.5):
        """Dibuja una capa referenciada por nombre simbólico.

        Args:
            key (str): Clave de la capa (ej: 'COASTLINE'). No sensible a mayúsculas.
            color (str): Color de la línea.
            width (float): Grosor de línea.
        """
        if self.image is None:
            return
        layer_key = key.upper()
        if layer_key not in self._layers:
            print(
                f"Advertencia: capa '{key}' no registrada. Capas disponibles: {self.list_layers()}")
            return
        rel_path = self._layers[layer_key]
        self.draw_shapefile(rel_path, color=color, width=width)

    def draw_grid(self, interval=15, color='gray', width=0.5, labels=False, label_size=None):
        """
        Dibuja una malla de latitud y longitud.
        
        Args:
            interval (int): Espacio en grados entre líneas (ej. 15).
            color (str): Color de las líneas.
            width (float): Grosor de las líneas.
            labels (bool): Si True, dibuja etiquetas con los valores de lat/lon.
            label_size (int, opcional): Tamaño de fuente para las etiquetas. 
                                        Si es None, se calcula automáticamente (0.8% del ancho).
        """
        if self.image is None:
            print("Advertencia: No hay imagen cargada para dibujar grilla.")
            return

        draw = ImageDraw.Draw(self.image)
        b = self.bounds
        
        # Verificar que los bounds estén inicializados
        if b['ulx'] == 0 and b['uly'] == 0 and b['lrx'] == 0 and b['lry'] == 0:
            print("Advertencia: Los límites geográficos (bounds) no están definidos. Use --bounds para especificarlos.")
            return
        
        # Calcular tamaño de etiqueta dinámico si no se especificó
        if label_size is None:
            # 0.8% del ancho de la imagen, mínimo 15px
            label_size = max(15, int(self.image.width * 0.008))
        
        # Pequeño margen para asegurar que las líneas cubran el borde
        margin = 2.0
        
        # Calcular número de muestras dinámicamente
        # Más muestras para proyecciones curvas (GOES), menos para lineales
        if self.use_proj:
            # Para proyecciones curvas, necesitamos más puntos
            n_samples = max(100, int(self.image.width / 10))
        else:
            # Para proyecciones lineales, bastan más puntos para suavidad
            n_samples = 50
        
        debug_msg(f"draw_grid: interval={interval}°, n_samples={n_samples}, "
                  f"image_size=({self.image.width}×{self.image.height}), "
                  f"labels={labels} (size={label_size}), use_proj={self.use_proj}")
        
        # Contador de líneas dibujadas
        lines_drawn = 0
        
        def draw_gridline(coords_pairs):
            """Dibuja una línea de grilla muestreando puntos y manejando discontinuidades."""
            nonlocal lines_drawn
            pixel_coords = []
            line_width = max(1, int(width))
            
            for lon, lat in coords_pairs:
                res = self._geo2pixel(lon, lat)
                
                # Verificar validez y rango
                if res and -50 <= res[0] <= self.image.width + 50 and -50 <= res[1] <= self.image.height + 50:
                    pixel_coords.extend(res)
                elif len(pixel_coords) >= 4:
                    # Discontinuidad: dibujar segmento acumulado
                    draw.line(pixel_coords, fill=color, width=line_width)
                    lines_drawn += 1
                    pixel_coords = []
            
            # Dibujar segmento final si existe
            if len(pixel_coords) >= 4:
                draw.line(pixel_coords, fill=color, width=line_width)
                lines_drawn += 1
        
        # --- Dibujar Meridianos (Longitud) ---
        start_lon = math.floor(b['ulx'] / interval) * interval
        end_lon = math.ceil(b['lrx'] / interval) * interval
        
        for lon in np.arange(start_lon, end_lon + interval, interval):
            lats = np.linspace(b['lry'] - margin, b['uly'] + margin, n_samples)
            draw_gridline([(lon, lat) for lat in lats])
        
        # --- Dibujar Paralelos (Latitud) ---
        start_lat = math.floor(b['lry'] / interval) * interval
        end_lat = math.ceil(b['uly'] / interval) * interval
        
        for lat in np.arange(start_lat, end_lat + interval, interval):
            lons = np.linspace(b['ulx'] - margin, b['lrx'] + margin, n_samples)
            draw_gridline([(lon, lat) for lon in lons])
        
        # Reportar resultado
        if lines_drawn > 0:
            print(f"Info: Grilla dibujada ({interval}°): {lines_drawn} segmentos de línea.")
        else:
            print(f"Advertencia: No se dibujó ninguna línea de grilla (interval={interval}°). Verifique que los bounds correspondan a la imagen.")
        
        # --- Etiquetas opcionales ---
        if labels:
            try:
                font = self._load_font(label_size)
                
                # Etiquetar meridianos (borde superior)
                for lon in np.arange(start_lon, end_lon + interval, interval):
                    res = self._geo2pixel(lon, b['uly'])
                    if res and 0 <= res[0] < self.image.width:
                        lon_val, direction = (abs(int(lon)), 'E' if lon >= 0 else 'W')
                        draw.text((res[0] + 2, 5), f"{lon_val}°{direction}", fill=color, font=font)
                
                # Etiquetar paralelos (borde izquierdo)
                for lat in np.arange(start_lat, end_lat + interval, interval):
                    res = self._geo2pixel(b['ulx'], lat)
                    if res and 0 <= res[1] < self.image.height:
                        lat_val, direction = (abs(int(lat)), 'N' if lat >= 0 else 'S')
                        draw.text((5, res[1] + 2), f"{lat_val}°{direction}", fill=color, font=font)
            except Exception as e:
                debug_msg(f"Error dibujando etiquetas: {e}")

    def draw_logo(self, logosize=128, position=3):
        """
        position: bitmask (0=Left, 1=Right) | (0=Top, 2=Bottom) 
        Ej: 0=UL, 1=UR, 2=LL, 3=LR
        """
        try:
            logo_path = os.path.join(
                self.lanot_dir, 'logos/lanot_negro_sn.png')
            logo = Image.open(logo_path)
        except FileNotFoundError:
            print("Logo no encontrado.")
            return

        # Mantener aspecto
        aspect = logo.height / logo.width
        new_h = int(logosize * aspect)
        logo = logo.resize((logosize, new_h), Image.Resampling.LANCZOS)

        pos_x = position & 1
        pos_y = position >> 1

        x = self.image.width - logosize - 10 if pos_x else 10
        y = self.image.height - new_h - 10 if pos_y else 10

        self.image.paste(logo, (x, y), logo)

    def draw_fecha(self, timestamp, position=2, fontsize=15, format="%Y/%m/%d %H:%MZ", color='white'):
        """
        Dibuja la fecha/hora en la imagen.

        Args:
            timestamp (datetime): Objeto datetime con la fecha/hora a dibujar
            position (int): Posición en la imagen (0=UL, 1=UR, 2=LL, 3=LR)
            fontsize (int): Tamaño de la fuente
            format (str): Formato de fecha usando códigos strftime (por defecto: "%Y/%m/%d %H:%MZ")
            color (str): Color del texto
        """
        if self.image is None:
            return

        try:
            from datetime import datetime

            # Convertir timestamp a string usando el formato especificado
            if isinstance(timestamp, datetime):
                fecha_str = timestamp.strftime(format)
            else:
                # Si es un string, usarlo directamente
                fecha_str = str(timestamp)

            # Usar ImageDraw para dibujar texto
            draw = ImageDraw.Draw(self.image)
            font = self._load_font(fontsize)

            # Calcular el tamaño real del texto renderizado
            try:
                # draw.textbbox devuelve (left, top, right, bottom)
                left, top, right, bottom = draw.textbbox(
                    (0, 0), fecha_str, font=font)
                text_width = right - left
                text_height = bottom - top
            except (AttributeError, TypeError):
                # Fallback: aproximación si textsize no está disponible
                text_width = len(fecha_str) * int(fontsize * 0.65)
                text_height = int(fontsize * 1.2)

            pos_x = position & 1
            pos_y = position >> 1

            margin = 10

            if pos_x:  # Right
                x = self.image.width - text_width - margin
            else:  # Left
                x = margin

            if pos_y:  # Bottom
                y = self.image.height - text_height - margin
            else:  # Top
                y = margin

            # Dibujar el texto
            draw.text((x, y), fecha_str, font=font, fill=color)

        except Exception as e:
            print(f"Error dibujando fecha: {e}")

    def draw_legend(self, items, position=2, fontsize=15, box_size=None,
                    padding=10, gap=6, margin=10, vertical_offset=0,
                    bg_color='white', text_color='black', border_color=None, border_width=1):
        """Dibuja una leyenda con recuadros de color y etiquetas.

        Args:
            items (list[tuple[str, tuple|str]]): Lista de (etiqueta, color).
            position (int): 0=UL, 1=UR, 2=LL, 3=LR.
            fontsize (int): Tamaño de fuente.
            box_size (int, opcional): Tamaño del cuadro de color. Por defecto = fontsize.
            padding (int): Relleno interno del fondo.
            gap (int): Espacio entre cuadro de color y texto.
            margin (int): Margen desde el borde de la imagen.
            vertical_offset (int): Desplazamiento vertical en píxeles desde el borde
                (positivo aleja del borde: hacia arriba si Bottom, hacia abajo si Top).
            bg_color (str|tuple): Color de fondo de la leyenda.
            text_color (str|tuple): Color del texto.
            border_color (str|tuple, opcional): Color del borde. None para sin borde.
            border_width (int): Grosor del borde si border_color no es None.
        """
        if self.image is None or not items:
            return

        box_size = box_size or fontsize
        draw = ImageDraw.Draw(self.image)
        font = self._load_font(fontsize)

        # Calcular dimensiones
        def text_width(s): return int(len(str(s)) * fontsize * 0.6)
        line_heights = [max(fontsize + 4, box_size) for _, _ in items]
        line_widths = [padding + box_size + gap +
                       text_width(label) + padding for label, _ in items]

        legend_width = max(line_widths)
        legend_height = padding + sum(line_heights) + padding

        # Calcular posición
        pos_x = position & 1
        pos_y = position >> 1

        x0 = self.image.width - legend_width - margin if pos_x else margin
        y0 = (self.image.height - legend_height - margin -
              vertical_offset) if pos_y else (margin + vertical_offset)
        x1 = x0 + legend_width
        y1 = y0 + legend_height

        # Dibujar fondo
        if border_color:
            draw.rectangle((x0, y0, x1, y1), fill=bg_color,
                           outline=border_color, width=border_width)
        else:
            draw.rectangle((x0, y0, x1, y1), fill=bg_color, outline=None)

        # Dibujar filas (cuadro de color + etiqueta)
        cy = y0 + padding
        for label, color in items:
            lh = max(fontsize + 4, box_size)

            # Cuadro de color
            bx0 = x0 + padding
            by0 = cy + (lh - box_size) // 2
            bx1 = bx0 + box_size
            by1 = by0 + box_size
            draw.rectangle((bx0, by0, bx1, by1), fill=color, outline=color)

            # Texto de etiqueta
            tx = bx1 + gap
            ty = cy + (lh - fontsize) // 2
            draw.text((tx, ty), str(label), font=font, fill=text_color)

            cy += lh

    def parse_cpt(self, cpt_path):
        """
        Parsea un archivo CPT y devuelve una lista de items para la leyenda.
        Soporta formato discreto simple: valor r g b [etiqueta]
        """
        items = []
        if not os.path.exists(cpt_path):
            print(f"Advertencia: CPT {cpt_path} no encontrado.")
            return items

        try:
            with open(cpt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(('#', 'B', 'F', 'N')):
                        continue

                    parts = line.split()
                    # Formato esperado discreto: val r g b [label]
                    if len(parts) >= 4:
                        try:
                            # Intentamos parsear r,g,b
                            r, g, b = int(parts[1]), int(
                                parts[2]), int(parts[3])
                            color = (r, g, b)

                            # Etiqueta
                            if len(parts) > 4:
                                # Si hay más partes, unimos el resto como etiqueta
                                label = " ".join(parts[4:])
                            else:
                                label = parts[0]

                            items.append((label, color))
                        except ValueError:
                            # Si falla conversión a int, saltamos
                            continue
        except Exception as e:
            print(f"Error leyendo CPT: {e}")

        return items


def calculate_size(value, ref_size, default=0):
    """Calcula tamaño en píxeles. Soporta enteros (px), floats <= 1.0 (escala) y porcentajes (%)."""
    if value is None:
        return default

    s_val = str(value).strip()
    if s_val.endswith('%'):
        try:
            pct = float(s_val[:-1])
            return int(ref_size * pct / 100.0)
        except ValueError:
            return default

    try:
        val = float(s_val)
        # Si es <= 1.0, asumimos que es un factor de escala (0.1 = 10%)
        if 0 < val <= 1.0:
            return int(ref_size * val)
        return int(val)
    except ValueError:
        return default


def get_timestamp_from_filename(filename):
    """
    Intenta extraer una marca de tiempo del nombre del archivo.
    Soporta formato Juliano (YYYYjjjHHMM).
    """
    import re
    from datetime import datetime

    basename = os.path.basename(filename)

    # Patrón: YYYYjjjHHMM (Julian)
    match = re.search(r"(\d{4})(\d{3})(\d{4})", basename)
    if match:
        yyyy, jjj, hhmm = match.groups()
        try:
            dt = datetime.strptime(f"{yyyy}{jjj}{hhmm}", "%Y%j%H%M")
            return dt.strftime("%Y/%m/%d %H:%MZ")
        except ValueError:
            pass
    return None

# --- Bloque Principal para pruebas ---


def main():
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="Herramienta de línea de comandos para dibujar mapas y decoraciones en imágenes.")

    parser.add_argument(
        "input_image", help="Ruta de la imagen de entrada (PNG, JPG, etc.)")
    parser.add_argument(
        "--output", "-o", help="Ruta de la imagen de salida. Por defecto sobreescribe la entrada.")

    # Límites
    parser.add_argument(
        "--bounds", help="Límites geográficos: 'ULX,ULY,LRX,LRY' (separados por coma). Use formato --bounds=... si inicia con negativo.")
    parser.add_argument(
        "--clip", help="Recortar imagen a límites: ULX,ULY,LRX,LRY (separados por coma) o nombre de región")

    # Capas
    parser.add_argument("--layer", action="append",
                        help="Capa a dibujar: NOMBRE:COLOR:GROSOR[:LABELS] "
                             "(ej: COASTLINE:blue:0.5, grid15:white:1.0:labels)")

    # Proyección
    parser.add_argument(
        "--crs", help="Sistema de coordenadas (ej: 'goes16', 'epsg:4326').")

    # Logo
    parser.add_argument("--logo-pos", type=int,
                        choices=[0, 1, 2, 3], help="Posición del logo (0-3)")
    parser.add_argument(
        "--logo-size", help="Tamaño del logo (píxeles, float <= 1.0 o porcentaje)")

    # Fecha
    parser.add_argument(
        "--timestamp", help="Texto de la fecha/hora a mostrar.")
    parser.add_argument("--timestamp-pos", type=int, choices=[
                        0, 1, 2, 3], help="Posición de la fecha (0-3). Si se especifica sin --timestamp, usa fecha actual.")
    parser.add_argument(
        "--font-size", help="Tamaño de fuente (píxeles, float <= 1.0 o porcentaje)")
    parser.add_argument("--font-color", default="yellow",
                        help="Color de fuente")

    # Leyenda
    parser.add_argument("--cpt", help="Archivo CPT para generar leyenda")
    parser.add_argument(
        "--metadata", help="Archivo JSON con metadatos (CRS, bounds, timestamp) para imágenes sin georreferencia.")
    parser.add_argument("--legend-pos", type=int,
                        choices=[0, 1, 2, 3], help="Posición de la leyenda (0-3)")
    parser.add_argument("--scale", "-s", type=float,
                        help="Factor de escala para redimensionar la imagen (ej. 0.5)")

    parser.add_argument("--jpeg", "-j", action="store_true",
                        help="Guardar salida en formato JPEG (por defecto PNG)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Mostrar mensajes de depuración")

    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    # 1. Cargar imagen
    if not os.path.exists(args.input_image):
        print(
            f"Error: Imagen '{args.input_image}' no encontrada.", file=sys.stderr)
        sys.exit(1)

    # Intentar extraer metadatos si es GeoTIFF y rasterio está disponible
    metadata = Metadata()

    if HAS_RASTERIO:
        debug_msg(
            f"Intentando leer metadatos con rasterio de {args.input_image}")
        try:
            with rasterio.open(args.input_image) as src:
                debug_msg(
                    f"Rasterio info - CRS: {src.crs}, Bounds: {src.bounds}, Tags: {list(src.tags().keys())}")
                metadata = Metadata.from_rasterio(src)
        except Exception as e:
            debug_msg(f"Excepción leyendo rasterio: {e}")
            pass

    # Cargar metadatos externos si se proporcionan (tienen prioridad o llenan vacíos)
    if args.metadata and os.path.exists(args.metadata):
        try:
            external_meta = Metadata.from_json_file(args.metadata)
            debug_msg(
                f"Cargando metadatos externos: {external_meta.to_dict()}")

            # Sobrescribir o llenar campos del metadata
            for key in ['crs', 'timestamp', 'satellite']:
                if key in external_meta:
                    metadata[key] = external_meta[key]

            # Bounds necesitan conversión de formato JSON [minx, miny, maxx, maxy] a rasterio (left, bottom, right, top)
            if 'bounds' in external_meta:
                b = external_meta['bounds']
                if len(b) == 4:
                    # JSON: [minx, miny, maxx, maxy] -> rasterio: (left, bottom, right, top)
                    metadata['bounds'] = (b[0], b[1], b[2], b[3])
        except Exception as e:
            print(f"Error leyendo metadatos externos: {e}", file=sys.stderr)

    try:
        img = Image.open(args.input_image).convert(
            "RGB")  # Asegurar RGB para dibujar
    except Exception as e:
        print(f"Error abriendo imagen: {e}", file=sys.stderr)
        sys.exit(1)

    if args.scale:
        if args.scale <= 0:
            print("Error: El factor de escala debe ser mayor que 0.", file=sys.stderr)
            sys.exit(1)
        new_w = max(1, int(img.width * args.scale))
        new_h = max(1, int(img.height * args.scale))
        debug_msg(
            f"Escalando imagen de {img.size} a {(new_w, new_h)} (Factor: {args.scale})")
        img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    # Calcular tamaños dinámicos si no se especifican
    img_width = img.width

    # Logo: 10% del ancho, mínimo 64px
    default_logo = max(64, int(img_width * 0.10))
    logo_size = calculate_size(args.logo_size, img_width, default_logo)

    # Fuente: 1.5% del ancho, mínimo 15px
    default_font = max(15, int(img_width * 0.015))
    font_size = calculate_size(args.font_size, img_width, default_font)

    # 2. Inicializar MapDrawer
    target_crs = args.crs if args.crs else metadata.get('crs')
    if target_crs and target_crs == metadata.get('crs'):
        print(f"Info: Usando CRS detectado: {target_crs}")
    debug_msg(f"Inicializando MapDrawer con CRS: {target_crs}")
    mapper = MapDrawer(target_crs=target_crs)
    mapper.set_image(img)

    # 3. Establecer límites
    bounds_set = False
    if args.bounds:
        if ',' in args.bounds:
            try:
                b_vals = [float(x) for x in args.bounds.split(',')]
                if len(b_vals) == 4:
                    mapper.set_bounds(*b_vals)
                    debug_msg(f"Usando bounds manuales: {b_vals}")
                    bounds_set = True
                else:
                    print(
                        "Error: --bounds requiere 4 valores: ULX,ULY,LRX,LRY", file=sys.stderr)
            except ValueError:
                print(
                    "Error: Formato de --bounds inválido. Use números separados por coma.", file=sys.stderr)
        else:
            if mapper.load_bounds_from_csv(args.bounds):
                debug_msg(f"Usando bounds por nombre: {args.bounds}")
                bounds_set = True
    elif 'bounds' in metadata:
        bounds = metadata.get_mapdrawer_bounds()
        if bounds:
            if mapper.use_proj and target_crs == metadata.get('crs'):
                # Usar bounds proyectados desde metadata
                b = metadata['bounds']
                mapper.set_projected_bounds(min_x=b[0], min_y=b[1],
                                            max_x=b[2], max_y=b[3])
                print(f"Info: Usando límites proyectados detectados.")
            else:
                mapper.set_bounds(*bounds)
                print(f"Info: Usando límites detectados: {bounds}")
            bounds_set = True
    else:
        debug_msg("No se establecieron límites (bounds).")

    # 4. Recorte (Clip)
    if args.clip:
        if ',' in args.clip:
            try:
                c_vals = [float(x) for x in args.clip.split(',')]
                if len(c_vals) == 4:
                    mapper.crop(*c_vals)
                else:
                    print(
                        "Error: --clip requiere 4 valores: ULX,ULY,LRX,LRY", file=sys.stderr)
            except ValueError:
                print(
                    "Error: Formato de --clip inválido. Use números separados por coma.", file=sys.stderr)
        else:
            # Intentar buscar por nombre de región
            bounds = mapper.get_region_bounds(args.clip)
            if bounds:
                debug_msg(f"Recortando a región '{args.clip}': {bounds}")
                mapper.crop(*bounds)
            else:
                print(
                    f"Error: Región de recorte '{args.clip}' no encontrada.", file=sys.stderr)

    # 5. Dibujar capas
    if args.layer and bounds_set:
        for layer_def in args.layer:
            parts = layer_def.split(':')
            name = parts[0]
            color = parts[1] if len(parts) > 1 else 'yellow'
            width = float(parts[2]) if len(parts) > 2 else 0.5
            
            # Detectar si la capa solicitada es una grilla
            if name.startswith('grid'):
                try:
                    # Extraer el número (ej: grid15 -> 15)
                    interval = int(name.replace('grid', ''))
                except ValueError:
                    interval = 10  # Valor por defecto si solo ponen 'grid'
                
                # Verificar si se pidieron etiquetas (cuarto parámetro)
                labels = False
                if len(parts) > 3 and parts[3].lower() in ('labels', 'label', 'l'):
                    labels = True
                
                mapper.draw_grid(interval=interval, color=color, width=width, labels=labels)
            else:
                mapper.draw_layer(name, color=color, width=width)
    elif args.layer and not bounds_set:
        print("Advertencia: Se pidieron capas pero no se definieron límites (--bounds).")

    # 6. Logo
    if args.logo_pos is not None:
        mapper.draw_logo(logosize=logo_size, position=args.logo_pos)

    # 7. Fecha
    # Solo mostrar fecha si se especificó --timestamp, --timestamp-pos,
    # o se detecta patrón de fecha en el nombre del archivo
    ts = None
    pos = None

    # Función auxiliar para formatear timestamp
    def format_ts(t_str, sat=None):
        # Intentar normalizar formato TIFF estándar (YYYY:MM:DD HH:MM:SS)
        try:
            dt = datetime.strptime(t_str, "%Y:%m:%d %H:%M:%S")
            t_str = dt.strftime("%Y/%m/%d %H:%MZ")
        except ValueError:
            pass
        if sat:
            t_str = f"{sat} {t_str}"
        return t_str

    if args.timestamp:
        # Usuario especificó texto explícito
        ts = args.timestamp
        pos = args.timestamp_pos if args.timestamp_pos is not None else 2
    elif args.timestamp_pos is not None:
        # Solo se dio --timestamp-pos
        if 'timestamp' in metadata:
            ts = format_ts(metadata['timestamp'], metadata.get('satellite'))
        else:
            # Usar fecha actual si no hay metadatos
            ts = datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%MZ")
        pos = args.timestamp_pos
    else:
        # No se especificó posición: solo informar si se detecta fecha, pero NO dibujar por defecto
        # para mantener consistencia con geotiff2view.
        if 'timestamp' in metadata:
            print(
                f"Info: Fecha detectada en metadatos: {format_ts(metadata['timestamp'], metadata.get('satellite'))}")
        else:
            # Intentar extraer del nombre (YYYYjjjHHMM)
            detected_ts = get_timestamp_from_filename(args.input_image)
            if detected_ts:
                print(f"Info: Fecha detectada en nombre: {detected_ts}")

    if ts is not None and pos is not None:
        mapper.draw_fecha(ts, position=pos, fontsize=font_size,
                          color=args.font_color)

    # 8. Leyenda
    if args.cpt and args.legend_pos is not None:
        items = mapper.parse_cpt(args.cpt)
        if items:
            # Si hay fecha en la misma posición, desplazar leyenda
            # Ajustar offset vertical basado en el tamaño de fuente
            v_offset = 0
            if pos is not None and pos == args.legend_pos:
                v_offset = int(font_size * 2.5)
            mapper.draw_legend(items, position=args.legend_pos,
                               fontsize=font_size, vertical_offset=v_offset)

    # 9. Guardar
    # Recuperar la imagen del mapper por si hubo recorte (crop genera nueva instancia)
    if mapper.image:
        img = mapper.image

    default_ext = ".jpg" if args.jpeg else ".png"
    base_name = os.path.splitext(os.path.basename(args.input_image))[0]

    if args.output:
        # Si es un directorio (existente o termina en /), construir ruta con nombre de entrada
        if os.path.isdir(args.output) or args.output.endswith(os.sep):
            output_path = os.path.join(
                args.output, f"{base_name}{default_ext}")
        else:
            output_path = args.output
    else:
        # Si no se especifica salida:
        # - Si es PNG y no se pide JPEG, sobreescribir (comportamiento original para PNG)
        # - Si es TIF u otro, o se pide JPEG, guardar con nueva extensión en directorio actual
        input_ext = os.path.splitext(args.input_image)[1].lower()
        if input_ext == '.png' and not args.jpeg:
            output_path = args.input_image
        else:
            output_path = f"{base_name}{default_ext}"

    try:
        img.save(output_path)
        print(f"Imagen guardada en {output_path}")
    except Exception as e:
        print(f"Error guardando imagen: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
