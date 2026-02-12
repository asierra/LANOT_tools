#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera múltiples recortes a partir de una imagen.

Usa MapDrawer para recortar regiones definidas (por nombre o coordenadas)
desde una imagen base (generalmente Full Disk reproyectada a Lat/Lon).

Uso:
  python3 crea_recortes_rgb.py input.png

Autor: Alejandro Aguilar Sierra
LANOT - Laboratorio Nacional de Observación de la Tierra
"""

import sys
from PIL import Image
from pathlib import Path
from mapdrawer import MapDrawer, get_timestamp_from_filename

sectores = [
    "a1",
    "a2",
#    "a3",
#    "a4",
#    "a5",
#    "a6",
    "atlantic",
#    "caribe",
#    "local",
    "mexico",
]


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <imagen fulldisk lalo>")
        sys.exit(1)

    inputpath = Path(sys.argv[1])
    base = f"{inputpath.stem}_"
    try:
        # Convertir a RGB para permitir capas de color
        original_img = Image.open(inputpath).convert("RGB")
    except Exception as e:
        print(f"Error abriendo imagen: {e}")
        sys.exit(1)
    
    # Usamos proyección lineal (None) ya que la imagen ya es Lat/Lon (Plate Carrée)
    crs = None
    timestamp = get_timestamp_from_filename(inputpath)

    for s in sectores:
        outputpath = f"{base}{s}.png"
        print(f"Procesando sector: {s}")
        
        mapper = MapDrawer(target_crs=crs)
        # Usar una COPIA de la imagen original para no afectar las siguientes iteraciones
        mapper.set_image(original_img.copy())
        
        # Establecer límites de la imagen original (Full Disk Lat/Lon)
        if not mapper.load_bounds_from_csv('fulldisk'):
            print("Error cargando límites fulldisk")
            continue

        # Obtener coordenadas del recorte
        bounds = mapper.get_region_bounds(s)
        if bounds:
            mapper.crop(*bounds)
            
            if mapper.image:
                if timestamp:
                    # Calcular tamaño de fuente: 3% de la altura, mínimo 12px
                    font_size = max(12, int(mapper.image.height * 0.03))
                    mapper.draw_fecha(timestamp, position=2, fontsize=font_size)

                logo_size = int(mapper.image.width * 0.10)
                mapper.draw_logo(logosize=logo_size, position=0)

                try:
                    mapper.image.save(outputpath)
                    print(f"Imagen guardada en {outputpath}")
                except Exception as e:
                    print(f"Error guardando imagen: {e}", file=sys.stderr)
        else:
            print(f"Región '{s}' no encontrada.")


if __name__ == "__main__":
    main()