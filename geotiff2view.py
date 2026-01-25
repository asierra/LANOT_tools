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
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import math
import re
from datetime import datetime

try:
    from mapdrawer import MapDrawer
except ImportError as e:
    # Fallback si el nombre del archivo o path varía
    try:
        import MapDrawer as md
        MapDrawer = md.MapDrawer
    except ImportError:
        # Si falla, mostramos el error original (probablemente falten dependencias como aggdraw)
        print(f"Advertencia: No se pudo importar MapDrawer. Detalle: {e}", file=sys.stderr)

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

def obtener_paso_redondo(raw_step):
    """Calcula un paso amigable (1, 2, 5 * 10^n)."""
    if raw_step <= 0: return 0
    exponente = math.floor(math.log10(raw_step))
    fraccion = raw_step / (10 ** exponente)
    
    if fraccion < 1.5: nice = 1
    elif fraccion < 3: nice = 2
    elif fraccion < 7: nice = 5
    else: nice = 10
    return nice * (10 ** exponente)

def generar_lista_alineada(val_min, val_max, num_intermedios):
    if val_min >= val_max:
        print("Error: El valor mínimo debe ser menor que el máximo.")
        sys.exit(1)

    # 1. Calcular el paso ideal
    rango_teorico = val_max - val_min
    raw_step = rango_teorico / (num_intermedios + 1)
    nice_step = obtener_paso_redondo(raw_step)
    
    # 2. Alinear el inicio al múltiplo del paso
    min_tmp = math.ceil(val_min / nice_step) * nice_step
    
    # 3. Determinar precisión visual
    precision = max(0, -math.floor(math.log10(nice_step))) if nice_step < 1 else 0

    # 4. Generar la secuencia
    lista_valores = []
    valor_actual = min_tmp
    
    # Condición: no exceder num_intermedios y mantenerse bajo el máximo
    while len(lista_valores) <= num_intermedios:
        if valor_actual >= val_max:
            break
        lista_valores.append(round(valor_actual, precision))
        valor_actual += nice_step
        
    return nice_step, min_tmp, lista_valores

def draw_value_row(draw, x0, y, width, min, max, num_intermedios, color, font_size=12):
    step, min_tmp, lista_valores = generar_lista_alineada(min, max, num_intermedios)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for value in lista_valores:
        x = x0 + (value - min) * width / (max - min)
        draw.text((x,y), str(value), fill=color, font=font)
    
def draw_label_row(draw, x0, y, width, min_val, max_val, offset, labels, color, font_size=12):
    """Dibuja etiquetas centradas en los bloques de color correspondientes."""
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    max_idx = int(max_val - offset)
    num_slots = max_idx + 1
    if num_slots <= 0: return
    
    step = width / num_slots
    
    for val, label in labels.items():
        idx = int(val - offset)
        if 0 <= idx <= max_idx:
            center_x = x0 + idx * step + step / 2
            try:
                l, t, r, b = draw.textbbox((0, 0), str(label), font=font)
                w_text = r - l
            except AttributeError:
                w_text, h_text = draw.textsize(str(label), font=font)
            
            draw.text((center_x - w_text / 2, y), str(label), fill=color, font=font)

def load_cpt(cpt_path, use_b_for_n=False, force_n=False):
    """
    Lee un archivo CPT y devuelve una paleta compatible con PIL.
    Ajusta el tamaño de la paleta (2, 4, 16, 256) según los valores.
    Si existe N (No Data), se coloca en el último índice de la paleta ajustada.
    """
    if not os.path.exists(cpt_path):
        debug_msg(f"Archivo CPT no encontrado: {cpt_path}")
        print(f"Advertencia: CPT {cpt_path} no encontrado.", file=sys.stderr)
        return None, None, None, 0, 0, 0, {}

    colors = {}
    special = {}
    labels = {}

    try:
        with open(cpt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                label_text = None
                # Separar etiqueta si existe (delimitada por ;)
                if ";" in line:
                    parts = line.split(";", 1)
                    line = parts[0].strip()
                    label_text = parts[1].strip()

                # Eliminar comentarios de la parte de datos
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
                vals = []
                for p in parts:
                    try:
                        vals.append(float(p))
                    except ValueError:
                        break
                
                n_vals = len(vals)

                # Caso discreto: value R G B (4 valores numéricos al inicio)
                if n_vals >= 4 and n_vals < 8:
                    val = int(vals[0])
                    colors[val] = (int(vals[1]), int(vals[2]), int(vals[3]))
                    
                    # Asignar etiqueta
                    if label_text:
                        labels[val] = label_text
                    elif len(parts) > 4: # Fallback para etiquetas separadas por espacio
                        labels[val] = " ".join(parts[4:])
                
                # Caso continuo/intervalo: v1 R1 G1 B1 v2 R2 G2 B2 (8 valores)
                elif n_vals >= 8:
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
            return None, None, None, 0, 0, 0, {}

        # Calcular offset si los valores exceden 255
        min_val = min(colors.keys())
        max_val = max(colors.keys())
        offset = 0
        
        if max_val > 255:
            offset = int(min_val)
            debug_msg(f"Valores CPT > 255. Aplicando offset de -{offset} a la paleta y datos.")

        has_n = 'N' in special or (use_b_for_n and 'B' in special) or force_n
        has_f = 'F' in special
        
        # Determinar tamaño de paleta (2, 4, 16, 256)
        # Usamos el máximo desplazado para calcular el tamaño necesario
        shifted_max = max_val - offset
        palette_size = 256
        for size in [2, 4, 16, 256]:
            reserved = 0
            if has_n: reserved = 1
            if has_f: reserved = max(reserved, 2)
            
            limit = size - reserved
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
            rgb = (0, 0, 0)
            if use_b_for_n and 'B' in special:
                rgb = special['B']
            elif 'N' in special:
                rgb = special['N']
            
            palette[n_idx*3] = rgb[0]
            palette[n_idx*3+1] = rgb[1]
            palette[n_idx*3+2] = rgb[2]
            debug_msg(f"Color N ({rgb}) asignado al índice {n_idx}")

        # Rellenar F si existe
        f_idx = None
        if has_f:
            f_idx = palette_size - 2
            rgb = special['F']
            palette[f_idx*3] = rgb[0]
            palette[f_idx*3+1] = rgb[1]
            palette[f_idx*3+2] = rgb[2]
            debug_msg(f"Color F ({rgb}) asignado al índice {f_idx}")

        return palette, n_idx, f_idx, offset, min_val, max_val, labels
    except Exception as e:
        print(f"Error leyendo CPT: {e}", file=sys.stderr)
        return None, None, None, 0, 0, 0, {}

def normalize_band(band, nodata_val=None):
    """Normaliza una banda numpy a 0-255 (uint8) y devuelve máscara de nodata."""
    data = band.astype(float)
    mask = np.zeros(data.shape, dtype=bool)
    
    # Detectar NaNs
    if np.issubdtype(band.dtype, np.floating):
        mask |= np.isnan(band)
        
    # Detectar nodata explícito
    if nodata_val is not None:
        if np.issubdtype(band.dtype, np.floating):
            mask |= np.isclose(data, nodata_val)
        else:
            mask |= (data == nodata_val)
    
    # Calcular min/max solo con datos válidos
    valid_data = data[~mask]
    
    if valid_data.size == 0:
        return np.zeros(data.shape, dtype=np.uint8), mask
        
    min_val = np.min(valid_data)
    max_val = np.max(valid_data)
    
    debug_msg(f"Normalizando banda: min={min_val}, max={max_val}")
    
    # Rellenar nodata con min_val para que al normalizar sea 0 (Negro)
    data[mask] = min_val
    
    if max_val == min_val:
        return np.zeros(data.shape, dtype=np.uint8), mask
        
    # Escalar
    norm = (data - min_val) / (max_val - min_val)
    return (norm * 255).astype(np.uint8), mask

def load_geotiff(filepath, n_idx=None, f_idx=None, offset=0, raw_values=False, transparent_nodata=False):
    """Lee un GeoTIFF y devuelve una imagen PIL."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    metadata = {}
    if HAS_RASTERIO:
        debug_msg(f"Abriendo con rasterio: {filepath}")
        with rasterio.open(filepath) as src:
            debug_msg(f"Metadatos: {src.meta}")
            
            # Extraer metadatos para MapDrawer
            if src.crs:
                metadata['crs'] = src.crs.to_string()
            metadata['bounds'] = src.bounds
            
            # Intentar extraer fecha de tags comunes
            tags = src.tags()
            debug_msg(f"Tags disponibles: {list(tags.keys())}")
            
            for key in ['TIFFTAG_DATETIME', 'DATETIME', 'date_created', 'time_coverage_start']:
                if key in tags:
                    metadata['timestamp'] = tags[key]
                    break
            
            # Intentar extraer satélite/plataforma
            for key in ['platform', 'satellite', 'spacecraft', 'mission', 'TIFFTAG_IMAGEDESCRIPTION']:
                if key in tags:
                    metadata['satellite'] = tags[key]
                    break

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
                    
                    # Definir límite superior para no invadir N o F si existen
                    upper_limit = 255
                    if n_idx is not None and n_idx == 255:
                        upper_limit = 254
                    
                    if f_idx is not None and f_idx <= upper_limit:
                        upper_limit = f_idx - 1
                        
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
                
                return Image.fromarray(img_data, 'L'), metadata
            
            # Si ya es uint8, usar directamente
            if data.dtype == 'uint8':
                if src.count >= 3:
                    return Image.fromarray(np.dstack(data[:3]), 'RGB'), metadata
                else:
                    return Image.fromarray(data[0], 'L'), metadata
            
            # Normalizar si es float/int16
            if src.count >= 3:
                r, m_r = normalize_band(data[0], src.nodata)
                g, m_g = normalize_band(data[1], src.nodata)
                b, m_b = normalize_band(data[2], src.nodata)
                img = Image.fromarray(np.dstack((r, g, b)), 'RGB')
                
                mask = m_r | m_g | m_b
                metadata['nodata_mask'] = mask
                
                if transparent_nodata:
                    # Combinar máscaras: si cualquiera es nodata, es transparente? 
                    # O solo si todos? Usualmente si todos. Asumamos unión por seguridad visual.
                    alpha = (~mask * 255).astype(np.uint8)
                    img.putalpha(Image.fromarray(alpha, 'L'))
                return img, metadata
            else:
                norm_data, mask = normalize_band(data[0], src.nodata)
                metadata['nodata_mask'] = mask
                img = Image.fromarray(norm_data, 'L')
                if transparent_nodata:
                    alpha = (~mask * 255).astype(np.uint8)
                    img.putalpha(Image.fromarray(alpha, 'L'))
                return img, metadata
    else:
        debug_msg("Rasterio no disponible, usando PIL.")
        print("Advertencia: rasterio no instalado. Usando PIL (funcionalidad limitada).", file=sys.stderr)
        img = Image.open(filepath)

        # PIL no puede guardar modo 'F' (float) como PNG. Debemos convertir.
        # También si es modo 'I' (16bit) y queremos aplicar paleta (raw_values), necesitamos convertir a 'L'.
        if img.mode == 'F' or (raw_values and img.mode.startswith('I')):
            arr = np.array(img)
            if raw_values:
                # Modo raw: intentar preservar valores enteros aplicando offset
                # Similar a la lógica de rasterio
                data_shifted = np.nan_to_num(arr) - offset
                
                upper_limit = 255
                if n_idx is not None and n_idx == 255:
                    upper_limit = 254
                if f_idx is not None and f_idx <= upper_limit:
                    upper_limit = f_idx - 1

                img_data = np.clip(data_shifted, 0, upper_limit).astype(np.uint8)
                return Image.fromarray(img_data, 'L'), metadata
            else:
                # Modo visualización: normalizar a 0-255
                norm_data, _ = normalize_band(arr)
                metadata['nodata_mask'] = mask
                return Image.fromarray(norm_data, 'L'), metadata

        return img, metadata

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

def main():
    global VERBOSE
    parser = argparse.ArgumentParser(description="Convierte GeoTIFF a imagen visible.")
    parser.add_argument("input", help="Archivo GeoTIFF de entrada")
    parser.add_argument("--output", "-o", help="Archivo de salida")
    parser.add_argument("--cpt", "-p", help="Archivo de paleta de colores (CPT)")
    parser.add_argument("--alpha", "-a", action="store_true", help="Hacer transparente el valor Nodata")
    parser.add_argument("--backcolor", "-b", action="store_true", help="Usar color B (Background) para Nodata")
    parser.add_argument("--invert", "-i", action="store_true", help="Invertir colores (blanco/negro o paleta)")
    parser.add_argument("--scale", "-s", type=float, help="Factor de escala para redimensionar la imagen (ej. 0.5)")
    parser.add_argument("--layer", action="append", help="Capa a dibujar: NOMBRE:COLOR:GROSOR (ej. COASTLINE:blue:0.5)")
    parser.add_argument("--logo-pos", type=int, choices=[0, 1, 2, 3], help="Posición del logo (0-3)")
    parser.add_argument("--logo-size", default="128", help="Tamaño del logo (píxeles o % del ancho)")
    parser.add_argument("--timestamp", help="Texto de la fecha/hora a mostrar.")
    parser.add_argument("--timestamp-pos", type=int, choices=[0, 1, 2, 3], help="Posición de la fecha (0-3).")
    parser.add_argument("--font-size", help="Tamaño de fuente (píxeles o % del ancho)")
    parser.add_argument("--font-color", default="yellow", help="Color de la fuente del timestamp")
    parser.add_argument("--legend-pos", type=int, choices=[0, 1, 2, 3], help="Posición de la leyenda (0-3)")
    parser.add_argument("--jpeg", "-j", action="store_true", help="Guardar salida en formato JPEG (por defecto PNG)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mostrar mensajes de depuración")
    
    args = parser.parse_args()
    VERBOSE = args.verbose
    
    default_ext = ".jpg" if args.jpeg else ".png"

    if args.output:
        # Si es un directorio (existente o termina en /), construir ruta con nombre de entrada
        if os.path.isdir(args.output) or args.output.endswith(os.sep):
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            output_path = os.path.join(args.output, f"{base_name}{default_ext}")
        else:
            output_path = args.output
    else:
        output_path = f"{os.path.splitext(args.input)[0]}{default_ext}"
    
    palette = None
    n_idx = None
    f_idx = None
    offset = 0
    cpt_min = 0
    cpt_max = 0
    labels = {}
    max_idx = None
    if args.cpt:
        palette, n_idx, f_idx, offset, cpt_min, cpt_max, labels = load_cpt(args.cpt, use_b_for_n=args.backcolor, force_n=args.alpha)
        debug_msg(f"Labels: {labels}")
        if palette:
            max_idx = int(cpt_max - offset)

    print(f"Procesando {args.input}...")
    img, metadata = load_geotiff(args.input, n_idx=n_idx, f_idx=f_idx, offset=offset, raw_values=(palette is not None), transparent_nodata=args.alpha)

    if args.invert:
        if palette and img.mode == 'L':
            debug_msg("Invirtiendo paleta...")
            # Calcular límite de datos para no tocar N/F
            limit = 255
            if n_idx is not None and n_idx == 255: limit = 254
            if f_idx is not None and f_idx <= limit: limit = f_idx - 1
            if max_idx is not None: limit = min(limit, max_idx)
            
            if limit >= 0:
                num_colors = limit + 1
                # Extraer segmento de datos
                segment = palette[0 : num_colors*3]
                # Revertir tripletas RGB
                triplets = [segment[i:i+3] for i in range(0, len(segment), 3)]
                triplets.reverse()
                reversed_segment = [val for t in triplets for val in t]
                # Aplicar
                palette[0 : num_colors*3] = reversed_segment
        else:
            debug_msg("Invirtiendo imagen...")
            # Invertir píxeles (blanco <-> negro), preservando alpha si existe
            if img.mode in ('RGBA', 'LA'):
                bands = img.split()
                # Invertir canales de color (todos menos el último alpha)
                inverted_bands = [ImageOps.invert(b) for b in bands[:-1]]
                inverted_bands.append(bands[-1])
                img = Image.merge(img.mode, tuple(inverted_bands))
            else:
                img = ImageOps.invert(img)
                # Si no hay alpha, restaurar el fondo negro usando la máscara
                if 'nodata_mask' in metadata:
                    mask = metadata['nodata_mask']
                    mask_img = Image.fromarray((mask * 255).astype(np.uint8), 'L')
                    if img.mode == 'RGB':
                        img.paste((0, 0, 0), mask=mask_img)
                    else:
                        img.paste(0, mask=mask_img)

    if args.scale:
        if args.scale <= 0:
            print("Error: El factor de escala debe ser mayor que 0.", file=sys.stderr)
            sys.exit(1)
        new_w = max(1, int(img.width * args.scale))
        new_h = max(1, int(img.height * args.scale))
        debug_msg(f"Escalando imagen de {img.size} a {(new_w, new_h)} (Factor: {args.scale})")
        resample_method = Image.Resampling.LANCZOS
        if palette:
            # Si hay paleta, usamos NEAREST para no alterar los índices de color
            resample_method = Image.Resampling.NEAREST
        img = img.resize((new_w, new_h), resample=resample_method)

    if palette:
        if img.mode == 'L':
            img.putpalette(palette)
            barsz = img.height // 20
            draw_colorbar(ImageDraw.Draw(img), palette, 0, img.height-2*barsz, img.width, barsz, max_index=max_idx)
            text_color = f_idx if f_idx is not None else 255
            if labels:
                draw_label_row(ImageDraw.Draw(img), 0, img.height-barsz, img.width, cpt_min, cpt_max, offset, labels, text_color, font_size=barsz//2)
            else:
                draw_value_row(ImageDraw.Draw(img), 0, img.height-barsz, img.width, cpt_min, cpt_max, 5, text_color, font_size=barsz//2)
            if args.alpha and n_idx is not None:
                img.info['transparency'] = n_idx
        else:
            print("Advertencia: La imagen no es de un solo canal (L), se ignora la paleta.", file=sys.stderr)

    # Integración con MapDrawer (Capas y Logo)
    if (args.logo_pos is not None or args.layer or args.timestamp_pos is not None or args.timestamp or args.legend_pos is not None):
        # MapDrawer requiere un modo de color directo (RGB/RGBA) para dibujar elementos con colores arbitrarios (capas, logos, texto).
        if img.mode == 'L' or img.mode == 'P':
            debug_msg("Convirtiendo imagen a RGB/RGBA para MapDrawer")
            mode = 'RGBA' if args.alpha else 'RGB'
            img = img.convert(mode)

        if 'MapDrawer' in globals():
            try:
                # Configurar CRS si está disponible
                target_crs = metadata.get('crs')
                mapper = MapDrawer(target_crs=target_crs)
                mapper.set_image(img)
                
                # Configurar Bounds si están disponibles
                if 'bounds' in metadata:
                    # rasterio bounds: left, bottom, right, top
                    b = metadata['bounds']
                    mapper.set_bounds(ulx=b[0], uly=b[3], lrx=b[2], lry=b[1])

                # Dibujar Capas
                if args.layer:
                    for layer_def in args.layer:
                        parts = layer_def.split(':')
                        name = parts[0]
                        color = parts[1] if len(parts) > 1 else 'yellow'
                        width = float(parts[2]) if len(parts) > 2 else 1.0
                        debug_msg(f"Dibujando capa {name} con color {color} y grosor {width}")
                        mapper.draw_layer(name, color=color, width=width)

                # Dibujar Logo
                if args.logo_pos is not None:
                    lsize = calculate_size(args.logo_size, img.width, 128)
                    debug_msg(f"Dibujando logo tamaño: {lsize} en posición {args.logo_pos}")
                    mapper.draw_logo(logosize=lsize, position=args.logo_pos)

                # Dibujar Timestamp
                ts_text = None
                ts_pos = args.timestamp_pos
                
                if args.timestamp:
                    ts_text = args.timestamp
                    if ts_pos is None: ts_pos = 2
                elif ts_pos is not None:
                    # 1. Intentar usar metadatos
                    if 'timestamp' in metadata:
                        ts_text = metadata['timestamp']
                        # Intentar normalizar formato TIFF estándar (YYYY:MM:DD HH:MM:SS)
                        try:
                            dt = datetime.strptime(ts_text, "%Y:%m:%d %H:%M:%S")
                            ts_text = dt.strftime("%Y/%m/%d %H:%MZ")
                        except ValueError:
                            pass

                    # 2. Intentar extraer del nombre del archivo si no hay metadatos
                    if not ts_text:
                        # Patrón 1: YYYYjjjHHMM (Julian)
                        match = re.search(r"(\d{4})(\d{3})(\d{4})", os.path.basename(args.input))
                        if match:
                            yyyy, jjj, hhmm = match.groups()
                            try:
                                dt = datetime.strptime(f"{yyyy}{jjj}{hhmm}", "%Y%j%H%M")
                                ts_text = dt.strftime("%Y/%m/%d %H:%MZ")
                            except ValueError:
                                pass
                        
                        # Patrón 2: YYYYMMDD_HHMMSS
                        if not ts_text:
                            match = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", os.path.basename(args.input))
                            if match:
                                try:
                                    dt = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
                                    ts_text = dt.strftime("%Y/%m/%d %H:%MZ")
                                except ValueError:
                                    pass

                # Fallback: Si no hay satélite en metadatos, buscar en nombre de archivo
                if 'satellite' not in metadata:
                    fname = os.path.basename(args.input).lower()
                    if fname.startswith('npp'):
                        metadata['satellite'] = 'Suomi NPP'
                    elif fname.startswith('j01') or 'noaa20' in fname:
                        metadata['satellite'] = 'NOAA-20'
                    elif fname.startswith('j02') or 'noaa21' in fname:
                        metadata['satellite'] = 'NOAA-21'
                    elif 'metopc' in fname:
                        metadata['satellite'] = 'Metop-C'
                    elif 'metopb' in fname:
                        metadata['satellite'] = 'Metop-B'
                    elif 'metopa' in fname:
                        metadata['satellite'] = 'Metop-A'

                if ts_text and ts_pos is not None:
                    # Si se detectó satélite y estamos en modo automático, agregarlo al texto
                    if not args.timestamp and 'satellite' in metadata:
                        ts_text = f"{metadata['satellite']} {ts_text}"
                        
                    default_fsize = max(15, int(img.width * 0.015))
                    fsize = calculate_size(args.font_size, img.width, default_fsize)
                    mapper.draw_fecha(ts_text, position=ts_pos, fontsize=fsize, color=args.font_color)

                # Dibujar Leyenda (MapDrawer)
                if args.legend_pos is not None and labels and palette:
                    legend_items = []
                    # Ordenar por valor para que la leyenda salga en orden
                    for val in sorted(labels.keys()):
                        label = labels[val]
                        idx = int(val - offset)
                        # Verificar límites de paleta
                        if 0 <= idx < 256:
                            r = palette[idx*3]
                            g = palette[idx*3+1]
                            b = palette[idx*3+2]
                            legend_items.append((label, (r, g, b)))
                    
                    if legend_items:
                        # Ajustar offset vertical si hay timestamp en la misma posición
                        v_offset = 0
                        default_fsize = max(15, int(img.width * 0.015))
                        lsize = calculate_size(args.font_size, img.width, default_fsize)
                        if args.timestamp_pos == args.legend_pos:
                             v_offset = int(lsize * 2.5)
                        mapper.draw_legend(legend_items, position=args.legend_pos, fontsize=lsize, vertical_offset=v_offset)
                    
            except Exception as e:
                print(f"Error en MapDrawer: {e}", file=sys.stderr)
        else:
            print("Advertencia: MapDrawer no disponible. Se omiten decoraciones.", file=sys.stderr)

    # Si el formato de salida es JPEG, convertir a RGB (necesario si hay paleta)
    ext = os.path.splitext(output_path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        if args.alpha:
            print("Advertencia: El formato JPEG no soporta transparencia (-a). Se ignorará.", file=sys.stderr)
        img = img.convert('RGB')

    img.save(output_path)
    print(f"Guardado en {output_path}")

if __name__ == "__main__":
    main()
