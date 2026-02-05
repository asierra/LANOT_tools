#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_cropper.py - Genera múltiples recortes a partir de una imagen.

Usa MapDrawer para recortar regiones definidas (por nombre o coordenadas)
desde una imagen base (generalmente Full Disk reproyectada a Lat/Lon).

Uso:
  python3 batch_cropper.py input.tif --regions mexico a1 a2 --layer COASTLINE:yellow
"""

import os
import sys
import argparse
from PIL import Image
from mapdrawer import MapDrawer

# Intentar importar rasterio para leer bounds automáticamente
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

def main():
    parser = argparse.ArgumentParser(description="Genera múltiples recortes de una imagen satelital.")
    parser.add_argument("input_image", help="Imagen de entrada (GeoTIFF o imagen proyectada)")
    parser.add_argument("--regions", "-r", nargs='+', required=True, 
                        help="Nombres de regiones a recortar (ej. mexico centroamerica)")
    parser.add_argument("--output-dir", "-o", default=".", help="Directorio de salida")
    parser.add_argument("--layer", action="append", 
                        help="Capas a dibujar en los recortes (ej. COASTLINE:cyan:1.0)")
    parser.add_argument("--global-bounds", nargs=4, type=float, metavar=('ULX', 'ULY', 'LRX', 'LRY'),
                        help="Forzar límites de la imagen de entrada (si no es GeoTIFF)")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_image):
        print(f"Error: No se encuentra {args.input_image}")
        sys.exit(1)

    # 1. Determinar límites de la imagen de entrada
    full_bounds = None
    
    if args.global_bounds:
        full_bounds = args.global_bounds
        print(f"Usando límites manuales para entrada: {full_bounds}")
    elif HAS_RASTERIO:
        try:
            with rasterio.open(args.input_image) as src:
                # rasterio: left, bottom, right, top
                # MapDrawer: ulx, uly, lrx, lry
                full_bounds = (src.bounds.left, src.bounds.top, src.bounds.right, src.bounds.bottom)
                print(f"Límites detectados (GeoTIFF): {full_bounds}")
        except Exception:
            pass
    
    if not full_bounds:
        print("Advertencia: No se pudieron detectar límites. Asumiendo Global (-180, 90, 180, -90).")
        print("Use --global-bounds ULX ULY LRX LRY si la imagen es diferente.")
        full_bounds = (-180.0, 90.0, 180.0, -90.0)

    # 2. Cargar imagen en memoria
    try:
        # Convertir a RGB para permitir capas de color
        original_img = Image.open(args.input_image).convert("RGB")
    except Exception as e:
        print(f"Error abriendo imagen: {e}")
        sys.exit(1)

    # 3. Inicializar MapDrawer
    # Asumimos proyección lineal (Plate Carrée) para la entrada si ya está reproyectada
    mapper = MapDrawer()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 4. Procesar regiones
    for region_name in args.regions:
        print(f"--- Procesando: {region_name} ---")
        
        # Obtener coordenadas del recorte
        crop_bounds = mapper.get_region_bounds(region_name)
        if not crop_bounds:
            print(f"  [!] Región '{region_name}' no encontrada en base de datos.")
            continue
            
        # Reiniciar estado del mapper con una COPIA de la imagen original completa
        mapper.set_image(original_img.copy())
        mapper.set_bounds(*full_bounds)
        
        # Aplicar recorte (modifica mapper.image y actualiza mapper.bounds)
        mapper.crop(*crop_bounds)
        
        # Dibujar capas (ahora sobre la imagen recortada)
        if args.layer:
            for layer_def in args.layer:
                parts = layer_def.split(':')
                name = parts[0]
                color = parts[1] if len(parts) > 1 else 'yellow'
                width = float(parts[2]) if len(parts) > 2 else 0.5
                mapper.draw_layer(name, color=color, width=width)
        
        # Guardar resultado
        out_filename = f"{region_name}.png"
        out_path = os.path.join(args.output_dir, out_filename)
        
        if mapper.image:
            mapper.image.save(out_path)
            print(f"  Guardado: {out_path}")
        else:
            print("  [!] Error: Imagen vacía tras recorte.")

if __name__ == "__main__":
    main()