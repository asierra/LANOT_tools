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
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import math
import re
from datetime import datetime

from colorpalettetable import ColorPaletteTable
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

# Directorio global de paletas (instalación estándar)
GLOBAL_CPT_DIR = "/usr/local/share/lanot/colortables"

def debug_msg(msg):
    if VERBOSE:
        print(f"[DEBUG] {msg}", file=sys.stderr)

def normalize_band(band, nodata_val=None):
    """Normaliza una banda numpy a 0-255 (uint8) y devuelve máscara de nodata."""
    data = band.astype(float)
    debug_msg(f"Normalizando float mínimo {np.nanmin(data)} y máximo {np.nanmax(data)}.")
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

def load_geotiff(filepath, n_idx=None, f_idx=None, offset=0, raw_values=False, transparent_nodata=False, is_normalized=False, autoscale_vals=None):
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
                
                # Detectar scale/offset del TIFF para conversión a valores físicos
                tiff_scale = 1.0
                tiff_offset = 0.0
                
                if src.scales and src.scales[0] is not None:
                    tiff_scale = src.scales[0]
                if src.offsets and src.offsets[0] is not None:
                    tiff_offset = src.offsets[0]
                
                # Fallback a tags si no están en propiedades estándar
                if tiff_scale == 1.0 and tiff_offset == 0.0:
                    tags = src.tags()
                    if 'scale' in tags:
                        try: tiff_scale = float(tags['scale'])
                        except: pass
                    if 'offset' in tags:
                        try: tiff_offset = float(tags['offset'])
                        except: pass
                
                is_scaled = (tiff_scale != 1.0 or tiff_offset != 0.0)
                if is_scaled:
                    debug_msg(f"Aplicando escala TIFF: {tiff_scale} y offset: {tiff_offset}")

                # Manejo de NoData / NaN
                mask = None
                
                # Si es float o tiene escala, procesamos como float (valores físicos)
                if np.issubdtype(band.dtype, np.floating) or is_scaled:
                    band_float = band.astype(float)
                    
                    # Determinar máscara de nodata sobre valores crudos
                    if src.nodata is not None:
                        if np.issubdtype(band.dtype, np.floating):
                            mask = np.isclose(band, src.nodata) | np.isnan(band)
                        else:
                            mask = (band == src.nodata)
                    else:
                        if np.issubdtype(band.dtype, np.floating):
                            mask = np.isnan(band)
                        else:
                            mask = np.zeros(band.shape, dtype=bool)

                    # Aplicar escala si existe
                    if is_scaled:
                        # Ponemos NaN en nodata para evitar cálculos erróneos
                        band_float[mask] = np.nan
                        band_phys = band_float * tiff_scale + tiff_offset
                    else:
                        band_phys = band_float
                    
                    # Definir límite superior para no invadir N o F si existen
                    upper_limit = 255
                    if n_idx is not None and n_idx == 255:
                        upper_limit = 254
                    if f_idx is not None and f_idx <= upper_limit:
                        upper_limit = f_idx - 1

                    if autoscale_vals is not None:
                        cpt_min, cpt_max = autoscale_vals
                        debug_msg(f"Auto-escalando datos normalizados (0-1) al rango {cpt_min}-{cpt_max}")
                        scaled_data = band_phys * (cpt_max - cpt_min) + cpt_min
                        data_shifted = np.nan_to_num(scaled_data) - offset
                        img_data = np.clip(data_shifted, 0, upper_limit).astype(np.uint8)
                    # Si el offset es 0 y los datos son float, se asume que son
                    # datos normalizados (0-1) que deben ser escalados a la paleta.
                    elif offset == 0 and is_normalized:
                        debug_msg(f"Escalando datos float (0-1) al rango de la paleta (0-{upper_limit})")
                        # Clip para seguridad, escalar y convertir a entero
                        scaled_data = np.clip(band_phys, 0.0, 1.0) * upper_limit
                        img_data = np.nan_to_num(scaled_data, nan=n_idx if n_idx is not None else 0).astype(np.uint8)
                    else:
                        # Lógica original para datos float que no son 0-1 (ej. Kelvin)
                        data_shifted = np.nan_to_num(band_phys) - offset
                        img_data = np.clip(data_shifted, 0, upper_limit).astype(np.uint8)

                else:
                    # Lógica similar para enteros SIN escala
                    data_shifted = band.astype(int) - offset
                    img_data = np.clip(data_shifted, 0, 255).astype(np.uint8)
                    if src.nodata is not None:
                        mask = (band == src.nodata)
                
                if n_idx is not None and mask is not None:
                    img_data[mask] = n_idx
           
                debug_msg(f"Mínimo {np.nanmin(data)} y máximo {np.nanmax(data)} de la banda (crudo).")
                
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
                if autoscale_vals is not None:
                    cpt_min, cpt_max = autoscale_vals
                    arr_float = arr.astype(float)
                    scaled_data = arr_float * (cpt_max - cpt_min) + cpt_min
                    data_shifted = np.nan_to_num(scaled_data) - offset
                else:
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
    parser.add_argument("--logo-size", default="128", help="Tamaño del logo (píxeles o porcentaje del ancho)")
    parser.add_argument("--timestamp", help="Texto de la fecha/hora a mostrar.")
    parser.add_argument("--timestamp-pos", type=int, choices=[0, 1, 2, 3], help="Posición de la fecha (0-3).")
    parser.add_argument("--font-size", help="Tamaño de fuente (píxeles o porcentaje del ancho)")
    parser.add_argument("--font-color", default="yellow", help="Color de la fuente del timestamp")
    parser.add_argument("--legend-pos", type=int, choices=[0, 1, 2, 3], help="Posición de la leyenda (0-3)")
    parser.add_argument("--jpeg", "-j", action="store_true", help="Guardar salida en formato JPEG (por defecto PNG)")
    parser.add_argument("--save-metadata", help="Guardar metadatos (CRS, bounds, timestamp) en un archivo JSON.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mostrar mensajes de depuración")
    parser.add_argument("--autoscale", action="store_true", help="Forzar escalado de datos normalizados (0-1) al rango de la paleta.")
    
    args = parser.parse_args()
    VERBOSE = args.verbose
    
    default_ext = ".jpg" if args.jpeg else ".png"

    # Manejo de input múltiple (separado por comas)
    input_files = [x.strip() for x in args.input.split(',')]
    primary_input = input_files[0]

    # Determinar nombre base para el archivo de salida
    if len(input_files) > 1:
        # Concatenar nombres de archivos para reflejar composición, pero solo las diferencias
        names = [os.path.splitext(os.path.basename(f))[0] for f in input_files]
        prefix = os.path.commonprefix(names)
        
        # Calcular sufijo sobre lo que queda después del prefijo
        remainders = [n[len(prefix):] for n in names]
        suffix = ""
        if remainders and all(remainders):
            reversed_rem = [r[::-1] for r in remainders]
            suffix = os.path.commonprefix(reversed_rem)[::-1]
            
        diffs = [n[len(prefix) : len(n)-len(suffix)] for n in names]
        middle = "_".join([d for d in diffs if d])
        base_name = f"rgb_{prefix}{middle}{suffix}"
    else:
        base_name = os.path.splitext(os.path.basename(primary_input))[0]

    if args.output:
        # Si es un directorio (existente o termina en /), construir ruta con nombre de entrada
        if os.path.isdir(args.output) or args.output.endswith(os.sep):
            output_path = os.path.join(args.output, f"{base_name}{default_ext}")
        else:
            output_path = args.output
    else:
        output_path = f"{base_name}{default_ext}"
    
    cpt_obj = None
    palette = None
    n_idx = None
    f_idx = None
    offset = 0
    max_idx = None
    is_normalized = False
    autoscale_vals = None

    if args.cpt:
        cpt_path = args.cpt
        # Si no existe localmente, buscar en directorio global
        if not os.path.exists(cpt_path):
            global_path = os.path.join(GLOBAL_CPT_DIR, os.path.basename(cpt_path))
            if os.path.exists(global_path):
                cpt_path = global_path

        cpt_obj = ColorPaletteTable(cpt_path, use_b_for_n=args.backcolor, force_n=args.alpha)
        palette = cpt_obj.get_pil_palette()
        n_idx = cpt_obj.n_idx
        f_idx = cpt_obj.f_idx
        offset = cpt_obj.offset
        is_normalized = cpt_obj.is_normalized
        debug_msg(f"Labels: {cpt_obj.labels}")
        if palette:
            max_idx = int(cpt_obj.max_val - offset)
            
        if args.autoscale:
            autoscale_vals = (cpt_obj.min_val, cpt_obj.max_val)

    if len(input_files) == 3:
        print(f"Modo RGB: Procesando 3 archivos...")
        # Resolver rutas relativas al primer archivo
        base_dir = os.path.dirname(primary_input)
        resolved_files = [primary_input]
        for f in input_files[1:]:
            if not os.path.isabs(f) and not os.path.exists(f):
                resolved_files.append(os.path.join(base_dir, os.path.basename(f)))
            else:
                resolved_files.append(f)
        
        channels = []
        metadata = None
        masks = []
        
        for idx, fpath in enumerate(resolved_files):
            print(f"  Cargando canal {['R','G','B'][idx]}: {fpath}")
            # Cargar normalizado (0-255) ignorando paleta
            ch_img, ch_meta = load_geotiff(fpath, offset=offset, raw_values=False, transparent_nodata=False, is_normalized=is_normalized)
            
            if ch_img.mode != 'L':
                ch_img = ch_img.convert('L')
            
            channels.append(ch_img)
            if metadata is None:
                metadata = ch_meta
            
            if 'nodata_mask' in ch_meta:
                masks.append(ch_meta['nodata_mask'])
        
        img = Image.merge('RGB', tuple(channels))
        
        if args.alpha and masks:
            # Combinar máscaras (Unión de nodata): si falta dato en algún canal, es transparente
            final_mask = masks[0]
            for m in masks[1:]:
                if m.shape == final_mask.shape:
                    final_mask = final_mask | m
            alpha = (~final_mask * 255).astype(np.uint8)
            img.putalpha(Image.fromarray(alpha, 'L'))
    else:
        print(f"Procesando {args.input}...")
        img, metadata = load_geotiff(args.input, n_idx=n_idx, f_idx=f_idx, offset=offset, raw_values=(palette is not None), transparent_nodata=args.alpha, is_normalized=is_normalized, autoscale_vals=autoscale_vals)

    # Guardar metadatos externos si se solicita
    if args.save_metadata:
        meta_export = {}
        if 'crs' in metadata:
            meta_export['crs'] = metadata['crs']
        if 'bounds' in metadata:
            # rasterio bounds: left, bottom, right, top -> [minx, miny, maxx, maxy]
            meta_export['bounds'] = list(metadata['bounds'])
        if 'timestamp' in metadata:
            meta_export['timestamp'] = metadata['timestamp']
        if 'satellite' in metadata:
            meta_export['satellite'] = metadata['satellite']
            
        try:
            with open(args.save_metadata, 'w') as f:
                json.dump(meta_export, f, indent=2)
            if VERBOSE:
                print(f"Metadatos guardados en {args.save_metadata}")
        except Exception as e:
            print(f"Error guardando metadatos: {e}", file=sys.stderr)

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
            # Usar el método de la clase CPT
            cpt_obj.draw_legend(ImageDraw.Draw(img), 0, img.height-2*barsz, img.width, barsz, font_size=barsz//2)
            
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
                if args.legend_pos is not None and cpt_obj and cpt_obj.labels and palette:
                    legend_items = []
                    # Ordenar por valor para que la leyenda salga en orden
                    for val in sorted(cpt_obj.labels.keys()):
                        label = cpt_obj.labels[val]
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
