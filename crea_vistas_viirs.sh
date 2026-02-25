#!/bin/bash

# Configuración
SOURCE_DIR="/dataservice/npp-jpss1/viirs/level2"
DEST_DIR="/var/www/html/polar/jpss/viirs"

# 1. Obtener archivos más recientes que no hayan sido procesados

# 2. Preparamos los parámetros 

# Definimos los argumentos como un arreglo


# Elegimos la paleta
if "cloud_type" in filename:
	paleta="cloud_type.cpt"
elif "cloud_phase" in filename:
	paleta="phase.cpt"
elif "cld_temp_acha" in filename:
	paleta="cld_temp_acha.cpt"
elif "sst" in filename:
	paleta="sst.cpt"
elif "viirs_confidence_cat" in filename:
	paleta="viirs_confidence_cat.cpt"

args=(
  --layer COUNTRIES:yellow:0.5 
  --layer MEXSTATES:yellow:0.5 
  --logo-pos 0 
  --logo-size 0.2 
  --font-size 0.025 
  --timestamp-pos 1 
  --font-color white 
  -s 0.5 
  -j
  -p "$paleta"
)

# 3. Creamos las vistas
echo "Iniciando procesamiento..."

geotiff2view "$filename" "${args[@]}"

echo "Proceso completado."
