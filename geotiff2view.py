#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geotiff2viewer - Herramienta para crear vistas a partir de datos en formato GeoTiff.

Genera vistas en niveles de grises, con paleta o RGB, dependiendo del tipo de datos.
Usando mapdrawer puede sobreponer capas vectoriales (costas, fronteras), logos, timestamps y leyendas
sobre imágenes.

Autor: Alejandro Aguilar Sierra
LANOT - Laboratorio Nacional de Observación de la Tierra
"""

import os
import sys
import argparse
from PIL import Image
import aggdraw
import numpy as np

try:
    from mapdrawer import MapDrawer
except ImportError:
    # Fallback si el nombre del archivo o path varía
    try:
        import MapDrawer as md
        MapDrawer = md.MapDrawer
    except ImportError:
        pass

# Intentamos importar rasterio para lectura avanzada de GeoTIFF
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

VERBOSE = False

def debug_msg(msg):
    if VERBOSE:
        print(f"[DEBUG] {msg}", file=sys.stderr)

from PIL import ImageDraw

def draw_colorbar(draw, colormap, x, y, width, height, max_index=None):
    """
    Dibuja una barra de colores usando solo Pillow (PIL).
    
    Args:
        draw: Un objeto ImageDraw.Draw(img).
        outline_color: Color para el borde (ej. 'black' o (0,0,0)). Equivale al 'pen'.
        colormap: Lista/array de colores [r, g, b, r, g, b...].
        x, y: Coordenadas de inicio.
        width, height: Dimensiones de la barra.
        max_index: Índice máximo a dibujar (inclusivo).
    """
    if not colormap:
        return

    total_colors = len(colormap) // 3
    if max_index is not None:
        num_colors = min(max_index + 1, total_colors)
    else:
        num_colors = total_colors

    if num_colors == 0:
        return

    step = width / num_colors
    debug_msg(f"Colores {num_colors}, ancho {width}, step {step}")

    mode = draw.im.mode

    for i in range(num_colors):
        # Definir coordenadas del segmento
        x0 = x + i * step
        x1 = x + (i + 1) * step
        
        # En PIL, 'fill' sustituye al 'brush' de aggdraw
        # Usamos [x0, y, x1, y + height] como lista o tupla
        if mode == 'P' or mode == 'L':
            # Si la imagen tiene paleta, dibujamos usando el índice
            draw.rectangle([x0, y, x1, y + height], fill=i)
        else:
            # Si es RGB, usamos el color directo
            r = colormap[i*3]
            g = colormap[i*3+1]
            b = colormap[i*3+2]
            draw.rectangle([x0, y, x1, y + height], fill=(r, g, b))

    # Dibujar el contorno final si se especificó un color (equivalente al pen)
    #if outline_color:
    #    draw.rectangle([x, y, x + width, y + height], outline=outline_color)

    # Nota: PIL no requiere draw.flush(), los cambios se aplican inmediatamente


def load_cpt(cpt_path):
    """
    Lee un archivo CPT y devuelve una paleta compatible con PIL.
    Ajusta el tamaño de la paleta (2, 4, 16, 256) según los valores.
    Si existe N (No Data), se coloca en el último índice de la paleta ajustada.
    """
    if not os.path.exists(cpt_path):
        debug_msg(f"Archivo CPT no encontrado: {cpt_path}")
        print(f"Advertencia: CPT {cpt_path} no encontrado.", file=sys.stderr)
        return None, None, 0, 0

    colors = {}
    special = {}

    try:
        with open(cpt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # Eliminar comentarios y etiquetas
                if ";" in line:
                    line = line.split(";")[0].strip()
                if "#" in line:
                    line = line.split("#")[0].strip()

                parts = line.split()
                if len(parts) < 4:
                    continue

                # Detectar si es una línea de color especial (B, F, N)
                if parts[0] in ['B', 'F', 'N']:
                    try:
                        r, g, b = int(float(parts[1])), int(float(parts[2])), int(float(parts[3]))
                        special[parts[0]] = (r, g, b)
                    except ValueError:
                        pass
                    continue

                # Intentar parsear valores numéricos
                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    continue

                # Caso discreto: value R G B (4 valores)
                if len(vals) == 4:
                    val = int(vals[0])
                    colors[val] = (int(vals[1]), int(vals[2]), int(vals[3]))
                
                # Caso continuo/intervalo: v1 R1 G1 B1 v2 R2 G2 B2 (8 valores)
                elif len(vals) == 8:
                    v1, r1, g1, b1 = vals[0], vals[1], vals[2], vals[3]
                    v2, r2, g2, b2 = vals[4], vals[5], vals[6], vals[7]
                    
                    start_i = int(np.ceil(v1))
                    end_i = int(np.floor(v2))
                    span = v2 - v1
                    
                    for i in range(start_i, end_i + 1):
                        if span == 0:
                            f = 0.0
                        else:
                            f = (i - v1) / span
                        f = max(0.0, min(1.0, f))
                        
                        cr = int(r1 + f * (r2 - r1))
                        cg = int(g1 + f * (g2 - g1))
                        cb = int(b1 + f * (b2 - b1))
                        colors[i] = (cr, cg, cb)
        
        if not colors:
            debug_msg("No se encontraron colores válidos en el CPT.")
            return None, None, 0, 0

        # Calcular offset si los valores exceden 255
        min_val = min(colors.keys())
        max_val = max(colors.keys())
        offset = 0
        
        if max_val > 255:
            offset = int(min_val)
            debug_msg(f"Valores CPT > 255. Aplicando offset de -{offset} a la paleta y datos.")

        has_n = 'N' in special
        
        # Determinar tamaño de paleta (2, 4, 16, 256)
        # Usamos el máximo desplazado para calcular el tamaño necesario
        shifted_max = max_val - offset
        palette_size = 256
        for size in [2, 4, 16, 256]:
            limit = size - 1 if has_n else size
            if shifted_max < limit:
                palette_size = size
                break
        
        debug_msg(f"Tamaño de paleta calculado: {palette_size} (Max val: {max_val}, Has N: {has_n})")

        # Crear paleta (PIL requiere 768 enteros, rellenamos con 0)
        palette = [0] * 768
        
        # Rellenar colores discretos
        for val, rgb in colors.items():
            val = val - offset
            if 0 <= val < 256:
                palette[val*3] = rgb[0]
                palette[val*3+1] = rgb[1]
                palette[val*3+2] = rgb[2]
        
        # Rellenar N si existe
        n_idx = None
        if has_n:
            n_idx = palette_size - 1
            rgb = special['N']
            palette[n_idx*3] = rgb[0]
            palette[n_idx*3+1] = rgb[1]
            palette[n_idx*3+2] = rgb[2]
            debug_msg(f"Color N ({rgb}) asignado al índice {n_idx}")

        return palette, n_idx, offset, shifted_max
    except Exception as e:
        print(f"Error leyendo CPT: {e}", file=sys.stderr)
        return None, None, 0, 0

def normalize_band(band):
    """Normaliza una banda numpy a 0-255 (uint8)."""
    # Convertir a float para cálculos
    data = band.astype(float)
    # Manejar NaNs
    data = np.nan_to_num(data, nan=np.nanmin(data))
    
    min_val = np.min(data)
    max_val = np.max(data)
    debug_msg(f"Normalizando banda: min={min_val}, max={max_val}")
    
    if max_val == min_val:
        return np.zeros_like(data, dtype=np.uint8)
        
    # Escalar
    norm = (data - min_val) / (max_val - min_val)
    return (norm * 255).astype(np.uint8)

def load_geotiff(filepath, n_idx=None, offset=0, raw_values=False):
    """Lee un GeoTIFF y devuelve una imagen PIL."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    if HAS_RASTERIO:
        debug_msg(f"Abriendo con rasterio: {filepath}")
        with rasterio.open(filepath) as src:
            debug_msg(f"Metadatos: {src.meta}")
            # Leer datos (bands, h, w)
            data = src.read()
            if raw_values:
                # Modo raw: preservar valores para uso con paleta
                # Asumimos banda única o tomamos la primera
                band = data[0]
                
                # Manejo de NoData / NaN
                mask = None
                if np.issubdtype(band.dtype, np.floating):
                    mask = np.isnan(band)
                    if src.nodata is not None:
                        mask |= (band == src.nodata)
                    
                    # Aplicar offset y convertir
                    # Usamos nan_to_num para evitar errores con NaNs al restar/castear
                    data_shifted = np.nan_to_num(band) - offset
                    
                    # Definir límite superior para no invadir N si existe
                    upper_limit = 255
                    if n_idx is not None and n_idx == 255:
                        upper_limit = 254
                        
                    # Clip para evitar wrap-around (valores > 255 volviéndose 0)
                    img_data = np.clip(data_shifted, 0, upper_limit).astype(np.uint8)

                else:
                    # Lógica similar para enteros
                    data_shifted = band.astype(int) - offset
                    img_data = np.clip(data_shifted, 0, 255).astype(np.uint8)
                    if src.nodata is not None:
                        mask = (band == src.nodata)
                
                if n_idx is not None and mask is not None:
                    img_data[mask] = n_idx
           
                debug_msg(f"Máximo {np.nanmax(data)} y mínimo {np.nanmin(data)} de la banda.")
                
                return Image.fromarray(img_data, 'L')
            
            # Si ya es uint8, usar directamente
            if data.dtype == 'uint8':
                if src.count >= 3:
                    return Image.fromarray(np.dstack(data[:3]), 'RGB')
                else:
                    return Image.fromarray(data[0], 'L')
            
            # Normalizar si es float/int16
            if src.count >= 3:
                r = normalize_band(data[0])
                g = normalize_band(data[1])
                b = normalize_band(data[2])
                return Image.fromarray(np.dstack((r, g, b)), 'RGB')
            else:
                return Image.fromarray(normalize_band(data[0]), 'L')
    else:
        debug_msg("Rasterio no disponible, usando PIL.")
        print("Advertencia: rasterio no instalado. Usando PIL (funcionalidad limitada).", file=sys.stderr)
        img = Image.open(filepath)

        # PIL no puede guardar modo 'F' (float) como PNG. Debemos convertir.
        if img.mode == 'F':
            arr = np.array(img)
            if raw_values:
                # Modo raw: intentar preservar valores enteros aplicando offset
                # Similar a la lógica de rasterio
                data_shifted = np.nan_to_num(arr) - offset
                upper_limit = 254 if (n_idx is not None and n_idx == 255) else 255
                img_data = np.clip(data_shifted, 0, upper_limit).astype(np.uint8)
                return Image.fromarray(img_data, 'L')
            else:
                # Modo visualización: normalizar a 0-255
                return Image.fromarray(normalize_band(arr), 'L')

        return img

def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description="Convierte GeoTIFF a imagen visible.")
    parser.add_argument("input", help="Archivo GeoTIFF de entrada")
    parser.add_argument("--output", "-o", help="Archivo de salida")
    parser.add_argument("--cpt", "-p", help="Archivo de paleta de colores (CPT)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mostrar mensajes de depuración")
    
    args = parser.parse_args()
    VERBOSE = args.verbose
    
    output_path = args.output if args.output else f"{os.path.splitext(args.input)[0]}.png"
    
    palette = None
    n_idx = None
    offset = 0
    max_idx = None
    if args.cpt:
        palette, n_idx, offset, max_idx = load_cpt(args.cpt)

    print(f"Procesando {args.input}...")
    img = load_geotiff(args.input, n_idx=n_idx, offset=offset, raw_values=(palette is not None))

    if palette:
        if img.mode == 'L':
            img.putpalette(palette)
            barsz = img.height // 20
            draw_colorbar(ImageDraw.Draw(img), palette, 0, img.height-2*barsz, img.width, barsz, max_index=max_idx)
        else:
            print("Advertencia: La imagen no es de un solo canal (L), se ignora la paleta.", file=sys.stderr)

    # Si el formato de salida es JPEG, convertir a RGB (necesario si hay paleta)
    ext = os.path.splitext(output_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        img = img.convert('RGB')

    img.save(output_path)
    print(f"Guardado en {output_path}")

if __name__ == "__main__":
    main()
