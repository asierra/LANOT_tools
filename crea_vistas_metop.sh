#!/bin/bash

# Configuración
SOURCE_DIR="/dataservice/noaa-metop/avhrr/vistas"
DEST_DIR="/var/www/html/polar/noaa-metop/avhrr/vistas"
PATTERN="*band[1-5]*"
THRESHOLD=600 

# 1. Obtener archivos
readarray -t RECENT_FILES < <(ls -t $SOURCE_DIR/$PATTERN 2>/dev/null | head -n 5)

if [ ${#RECENT_FILES[@]} -eq 0 ]; then
    echo "No se encontraron archivos."
    exit 1
fi

# 2. Comparación de timestamps (usando el más reciente del arreglo)
SOURCE_TIME=$(stat -c %Y "${RECENT_FILES[0]}")
LATEST_DEST_FILE=$(ls -t "$DEST_DIR" 2>/dev/null | head -n 1)

if [ -n "$LATEST_DEST_FILE" ]; then
    DEST_TIME=$(stat -c %Y "$DEST_DIR/$LATEST_DEST_FILE")
    DIFF=$(( SOURCE_TIME - DEST_TIME ))
    DIFF=${DIFF#-} 

    if [ "$DIFF" -lt "$THRESHOLD" ]; then
        echo "Diferencia menor a 10 min. Nada que procesar."
        exit 0
    fi
fi

# Definimos los argumentos como un arreglo
args=(
  --layer COUNTRIES:yellow:0.0005 
  --layer MEXSTATES:yellow:0.0005 
  --logo-pos 0 
  --logo-size 0.2 
  --font-size 0.025 
  --timestamp-pos 1 
  --font-color white 
  -s 0.5 
  -j
)

# 3. Procesamiento diferenciado
echo "Iniciando procesamiento..."
cd $DEST_DIR
for FILE in "${RECENT_FILES[@]}"; do
    FNAME=$(basename "$FILE")

    # Identificar banda para el comando final
    if [[ "$FNAME" == *"band1"* ]]; then B1_FULL="$FILE"; fi
    if [[ "$FNAME" == *"band2"* ]]; then B2_NAME="$FNAME"; fi
    if [[ "$FNAME" == *"band3"* ]]; then B3_NAME="$FNAME"; fi

    # Comandos por separado
    if [[ "$FNAME" == *"band4"* || "$FNAME" == *"band5"* ]]; then
        echo "Ejecutando proceso térmico para: $FNAME"
        geotiff2view "$FILE" "${args[@]}" -i
    else
        echo "Ejecutando proceso óptico para: $FNAME"
        geotiff2view "$FILE" "${args[@]}" 
    fi
done

# 4. Ejecución del comando fuera del loop
echo "-------------------------------------------"
if [ -n "$B1_FULL" ] && [ -n "$B2_NAME" ] && [ -n "$B3_NAME" ]; then
    echo "Generando vista compuesta..."
    geotiff2view "$B1_FULL,$B2_NAME,$B3_NAME" "${args[@]}"
else
    echo "Error: No se pudieron identificar todas las bandas (B1, B2, B3)."
fi

echo "Proceso completado."
