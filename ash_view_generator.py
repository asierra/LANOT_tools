#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ash_view_generator - Visualizador de detección de ceniza volcánica sobre ABI.

Genera una imagen compuesta superponiendo el producto de ceniza volcánica
(GeoTIFF RGBA de 4 bandas generado por detect_ash.py) sobre una imagen base
ABI (GeoTIFF de banda única), con capas vectoriales opcionales via MapDrawer.

Uso como módulo:
    from ash_view_generator import render_ash_layer
    ash_layer = render_ash_layer(ash_tif, metadata)
    result = Image.alpha_composite(base_img.convert('RGBA'), ash_layer)

Uso standalone:
    ash_view_generator.py base.tif ceniza.tif -o salida.png \\
        --layer MEXSTATES:white:1.0 --legend-pos 2 --logo-pos 3

El GeoTIFF de ceniza debe estar en la misma proyección que la imagen base.
Si los CRS difieren, el programa imprime un error y termina.

Autor: Alejandro Aguilar Sierra
LANOT - Laboratorio Nacional de Observación de la Tierra
"""

import sys
import argparse

import numpy as np
from PIL import Image

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from pyproj import CRS
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# Colores de la leyenda (coinciden con ash.cpt de LANOT_ceniza)
ASH_LEGEND_ITEMS = [
    ('ash',           (255,   0,   0)),
    ('probable',      (255, 165,   0)),
    ('possible', (255, 255,   0)),
]


def _crs_equal(crs_a, crs_b):
    """
    Compara dos CRS (rasterio, string o pyproj) de forma semántica.
    Devuelve True si son equivalentes, False si difieren, None si no se puede
    determinar (pyproj no disponible).
    """
    if not HAS_PYPROJ:
        return None
    try:
        return CRS.from_user_input(crs_a).equals(CRS.from_user_input(crs_b))
    except Exception:
        return None


def render_ash_layer(ash_tif, metadata):
    """
    Carga un GeoTIFF de ceniza RGBA y lo mapea al espacio de la imagen base.

    El GeoTIFF de ceniza debe ser de 4 bandas (RGBA) en la misma proyección
    que la imagen base descrita por `metadata`. Si los CRS difieren, devuelve
    None con un mensaje de error.

    Args:
        ash_tif (str): Ruta al GeoTIFF de ceniza (4 bandas RGBA).
        metadata: Instancia de Metadata con 'crs', 'bounds' e 'image_size'.

    Returns:
        PIL.Image (RGBA) o None: Canvas del tamaño de la imagen base con la
        ceniza superpuesta, listo para Image.alpha_composite(). None en error.
    """
    if not HAS_RASTERIO:
        print("Error: rasterio es necesario para render_ash_layer.", file=sys.stderr)
        return None

    base_bounds = metadata.get('bounds')
    if base_bounds is None:
        print("Error: metadata no contiene 'bounds'.", file=sys.stderr)
        return None

    img_size = metadata.get('image_size')
    if img_size is None:
        print("Error: metadata no contiene 'image_size'.", file=sys.stderr)
        return None

    base_w, base_h = img_size
    base_left, base_bottom, base_right, base_top = base_bounds

    try:
        with rasterio.open(ash_tif) as src:
            if src.count < 4:
                print(f"Error: {ash_tif} debe tener 4 bandas RGBA (tiene {src.count}).",
                      file=sys.stderr)
                return None

            # --- Verificar compatibilidad de CRS ---
            base_crs_str = metadata.get('crs')
            ash_crs = src.crs

            if base_crs_str is not None:
                equal = _crs_equal(base_crs_str, ash_crs)
                if equal is False:
                    print("Error: los CRS de la imagen base y la ceniza no coinciden.",
                          file=sys.stderr)
                    print(f"  Base: {base_crs_str}", file=sys.stderr)
                    print(f"  Ash:  {ash_crs}", file=sys.stderr)
                    return None
                if equal is None:
                    print("Advertencia: pyproj no disponible; no se pudo verificar CRS.",
                          file=sys.stderr)

            # --- Leer las 4 bandas RGBA ---
            r_band, g_band, b_band, a_band = src.read([1, 2, 3, 4])
            ash_bounds = src.bounds

    except Exception as e:
        print(f"Error leyendo {ash_tif}: {e}", file=sys.stderr)
        return None

    ash_left   = ash_bounds.left
    ash_bottom = ash_bounds.bottom
    ash_right  = ash_bounds.right
    ash_top    = ash_bounds.top

    base_span_x = base_right - base_left
    base_span_y = base_top   - base_bottom

    if base_span_x == 0 or base_span_y == 0:
        print("Error: bounds de la imagen base tienen extensión cero.", file=sys.stderr)
        return None

    # --- Mapeo lineal: bounds del ash → píxeles en la base ---
    x_offset = int((ash_left   - base_left)   / base_span_x * base_w)
    y_offset = int((base_top   - ash_top)     / base_span_y * base_h)
    ash_w_px = int((ash_right  - ash_left)    / base_span_x * base_w)
    ash_h_px = int((ash_top    - ash_bottom)  / base_span_y * base_h)

    if ash_w_px <= 0 or ash_h_px <= 0:
        print("Advertencia: la ceniza no tiene superposición visible con la imagen base.",
              file=sys.stderr)
        return None

    # --- Construir imagen RGBA del ash y escalar al espacio de la base ---
    ash_rgba = np.stack([r_band, g_band, b_band, a_band], axis=-1).astype(np.uint8)
    ash_img  = Image.fromarray(ash_rgba, 'RGBA')
    ash_img  = ash_img.resize((ash_w_px, ash_h_px), Image.NEAREST)

    # --- Componer sobre canvas del tamaño de la base ---
    canvas = Image.new('RGBA', (base_w, base_h), (0, 0, 0, 0))
    canvas.paste(ash_img, (x_offset, y_offset))

    return canvas


# ---------------------------------------------------------------------------
# Uso standalone
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if not HAS_RASTERIO:
        print("Error: rasterio es requerido.", file=sys.stderr)
        sys.exit(1)

    from metadata import Metadata
    from mapdrawer import MapDrawer

    parser = argparse.ArgumentParser(
        description="Superpone detección de ceniza volcánica sobre imagen ABI.")
    parser.add_argument("base_tif",
                        help="GeoTIFF base (ej. C13-C15_*.tif).")
    parser.add_argument("ash_tif",
                        help="GeoTIFF de ceniza RGBA (ej. ceniza_*.tif).")
    parser.add_argument("-o", "--output", default="ash_output.png",
                        help="Archivo PNG de salida (default: ash_output.png).")
    parser.add_argument("--layer", action="append", metavar="NOMBRE:COLOR:GROSOR",
                        help="Capa vectorial a dibujar. Puede repetirse. "
                             "Ej: MEXSTATES:white:1.0")
    parser.add_argument("--logo-pos", type=int, choices=[0, 1, 2, 3],
                        metavar="POS",
                        help="Posición del logo LANOT (0=UL 1=UR 2=LL 3=LR).")
    parser.add_argument("--legend-pos", type=int, choices=[0, 1, 2, 3],
                        metavar="POS",
                        help="Posición de la leyenda de ceniza (0=UL 1=UR 2=LL 3=LR).")
    parser.add_argument("--scale", type=float, default=1.0,
                        metavar="FACTOR",
                        help="Factor de escala para la imagen de salida (default: 1.0).")
    parser.add_argument("--crs",
                        help="Override del CRS (ej: goes16, epsg:4326). "
                             "Por defecto se toma del GeoTIFF base.")
    args = parser.parse_args()

    # 1. Cargar imagen base y metadatos
    try:
        with rasterio.open(args.base_tif) as src:
            metadata = Metadata.from_rasterio(src)
            raw = src.read(1)
            base_w, base_h = src.width, src.height
    except Exception as e:
        print(f"Error abriendo imagen base: {e}", file=sys.stderr)
        sys.exit(1)

    if args.crs:
        metadata['crs'] = args.crs

    # Normalizar la banda base al rango completo 0-255.
    # Los visores como QGIS aplican este mismo stretch automáticamente a TIFs
    # de banda única; al producir una imagen RGB hay que hacerlo explícitamente
    # para que la apariencia sea equivalente.
    rmin, rmax = float(raw.min()), float(raw.max())
    if rmax > rmin:
        gray = ((raw.astype(np.float32) - rmin) / (rmax - rmin) * 255).astype(np.uint8)
    else:
        gray = raw.astype(np.uint8)
    base_img = Image.fromarray(gray, 'L').convert('RGB')

    if args.scale != 1.0:
        new_w = int(base_w * args.scale)
        new_h = int(base_h * args.scale)
        base_img = base_img.resize((new_w, new_h), Image.LANCZOS)
        base_w, base_h = new_w, new_h

    metadata['image_size'] = (base_w, base_h)

    # 2. Render de la capa de ceniza
    ash_layer = render_ash_layer(args.ash_tif, metadata)
    if ash_layer is None:
        print("No se generó capa de ceniza. Abortando.", file=sys.stderr)
        sys.exit(1)

    result = Image.alpha_composite(base_img.convert('RGBA'), ash_layer).convert('RGB')

    # 3. MapDrawer: capas vectoriales y decoraciones
    needs_mapper = args.layer or args.logo_pos is not None or args.legend_pos is not None
    if needs_mapper:
        mapper = MapDrawer(target_crs=metadata.get('crs'))
        mapper.set_image(result)
        bounds = metadata.get_mapdrawer_bounds()  # (ulx, uly, lrx, lry)
        if bounds:
            ulx, uly, lrx, lry = bounds
            # Detectar si los bounds están en coordenadas proyectadas (metros)
            # vs. geográficas (grados lat/lon): lat/lon siempre dentro de [-180,180]x[-90,90]
            is_projected = (abs(ulx) > 360 or abs(lrx) > 360 or
                            abs(uly) > 90  or abs(lry) > 90)
            if mapper.use_proj and is_projected:
                b = metadata['bounds']  # rasterio: (left, bottom, right, top)
                mapper.set_projected_bounds(min_x=b[0], min_y=b[1],
                                            max_x=b[2], max_y=b[3])
            else:
                mapper.set_bounds(*bounds)
        else:
            print("Advertencia: sin bounds en metadata; capas vectoriales pueden quedar mal.",
                  file=sys.stderr)

        # Capas vectoriales
        if args.layer and bounds:
            for layer_def in args.layer:
                parts = layer_def.split(':')
                name  = parts[0]
                color = parts[1] if len(parts) > 1 else 'white'
                width = float(parts[2]) if len(parts) > 2 else 1.0
                mapper.draw_layer(name, color=color, width=width)

        # Logo
        if args.logo_pos is not None:
            mapper.draw_logo(position=args.logo_pos)

        # Leyenda
        if args.legend_pos is not None:
            mapper.draw_legend(ASH_LEGEND_ITEMS, position=args.legend_pos)

        result = mapper.image

    # 4. Guardar
    out_ext = args.output.lower().rsplit('.', 1)[-1]
    if out_ext in ('tif', 'tiff'):
        # Guardar como GeoTIFF preservando CRS y geotransform de la imagen base
        base_bounds = metadata.get('bounds')
        base_crs    = metadata.get('crs')
        if base_bounds is None or base_crs is None:
            print("Advertencia: metadata incompleta; guardando GeoTIFF sin georeferencia.",
                  file=sys.stderr)
            result.save(args.output)
        else:
            import rasterio.transform
            result_arr = np.array(result)          # (H, W, 3) uint8
            h, w = result_arr.shape[:2]
            transform = rasterio.transform.from_bounds(
                base_bounds[0], base_bounds[1], base_bounds[2], base_bounds[3],
                w, h)
            with rasterio.open(
                args.output, 'w',
                driver='GTiff',
                height=h, width=w,
                count=3,
                dtype='uint8',
                crs=base_crs,
                transform=transform,
            ) as dst:
                for band_idx in range(3):
                    dst.write(result_arr[:, :, band_idx], band_idx + 1)
    else:
        result.save(args.output)
    print(f"Guardado: {args.output}")
