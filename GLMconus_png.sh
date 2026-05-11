#! /bin/bash
# GLMconus_png.sh
# Renderiza los 15 archivos GLM más recientes sobre el CONUS ABI C13 más reciente.
# Cada 5 minutos cae un CONUS y ~15 GLMs (1 por minuto).
#
# Dependencias: hpsv (hpsatviews), mapdrawer, glm_renderer
# Uso: ./GLMconus_png.sh

indirabi=/data1/input/abi/l1b/conus
indirglm=/data1/input/glm
outdir=/data/goes19/glm/vistas/conus
workingdir=/var/tmp/conusglm

if [ ! -d "$workingdir" ]; then
    echo "Creando directorio de trabajo $workingdir"
    mkdir -p "$workingdir"
fi

cd "$workingdir"

# --- Buscar el ABI C13 más reciente ---
fileabi=$(find "$indirabi" -maxdepth 1 -name "*C13*.nc" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z "$fileabi" ]; then
    echo "Error: No se encontró archivo ABI C13 en $indirabi" >&2
    exit 1
fi

# Nombre base para archivos intermedios y de salida
basename_abi=$(basename "$fileabi" .nc)

# --- Consultar los 15 GLMs más recientes ---
mapfile -t glm_files < <(find "$indirglm" -maxdepth 1 -name "*.nc" -printf "%T@ %p\n" 2>/dev/null | sort -rn | head -15 | cut -d' ' -f2-)
if [ ${#glm_files[@]} -eq 0 ]; then
    echo "Error: No se encontraron archivos GLM en $indirglm" >&2
    exit 1
fi

# Nombre de salida derivado del archivo GLM más reciente (mantiene convención anterior)
outfile=$(basename "${glm_files[0]}" .nc)

if [ -f "$outdir/$outfile.png" ]; then
    echo "Info: $outfile ya fue procesado. Saliendo."
    exit 0
fi

echo "Procesando ABI: $basename_abi"
echo "GLMs: ${#glm_files[@]} archivos"

# --- 1. Imagen base ABI C13 en escala de grises (GeoTIFF georreferenciado) ---
hpsv gray -i "$fileabi" -t -g 1.5 -o base.tif
if [ $? -ne 0 ] || [ ! -f base.tif ]; then
    echo "Error: hpsv no generó base.tif" >&2
    exit 1
fi

# --- 2+3. Overlay GLM + capas vectoriales + decoraciones ---
mapdrawer base.tif \
    --glm "${glm_files[@]}" \
    --glm-color yellow \
    --layer COASTLINE:white:1.0 \
    --layer MEXSTATES:white:0.5 \
    --logo-pos 0 \
    --timestamp-pos 3 \
    -o "$outfile.png"

if [ $? -ne 0 ] || [ ! -f "$outfile.png" ]; then
    echo "Error: mapdrawer falló al generar $outfile.png" >&2
    rm -f base.tif
    exit 1
fi

mv "$outfile.png" "$outdir/"
echo "Guardado: $outdir/$outfile.png"

# Limpiar directorio de trabajo
#rm -f base.tif
