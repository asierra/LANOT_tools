# Plan de Integración: MapDrawer en Geotiff2View

Este documento describe los pasos necesarios para integrar las capacidades de `mapdrawer` (dibujo de capas vectoriales, logos y timestamps) dentro de la herramienta `geotiff2view.py`, aprovechando los metadatos geográficos del archivo GeoTIFF de entrada.

## 0. Refactorar MapDrawer

Revisar y si hace falta corregir MapDrawer. Hay funciones repetidas como la lectura de cpt. ¿Cómo se resuelve esto?

## 1. Modificación de Argumentos (CLI)
Actualizar `argparse` en `geotiff2view.py` para aceptar las opciones de decoración que soporta `mapdrawer`.

**Nuevos argumentos a agregar:**
*   `--layer`: Argumento tipo *append* para agregar capas (ej. `COASTLINE:white:0.5`).
*   `--logo-pos`: Posición del logo (0-3).
*   `--logo-size`: Tamaño del logo en píxeles.
*   `--timestamp`: Texto explícito para la fecha/hora.
*   `--timestamp-pos`: Posición del timestamp (0-3).
*   `--font-size`: Tamaño de fuente para el timestamp.
*   `--font-color`: Color de la fuente.

## 2. Extracción de Metadatos Geográficos
Actualmente `load_geotiff` devuelve solo la imagen PIL. Necesitamos extraer la información de georreferenciación para configurar `MapDrawer`.

**Cambios en `load_geotiff`:**
*   Modificar la función para que devuelva una tupla: `(image, metadata)`.
*   **Con Rasterio:**
    *   Extraer `bounds` (left, bottom, right, top).
    *   Extraer `crs` (Sistema de coordenadas).
*   **Con PIL (Fallback):**
    *   Devolver metadatos vacíos o `None` si no se pueden determinar (lo que deshabilitaría las capas vectoriales, pero permitiría logos/timestamps).

## 3. Lógica de Timestamp
Implementar una estrategia para determinar la fecha y hora a mostrar:
1.  Si se proporciona `--timestamp` explícito, usarlo.
2.  Si no, intentar extraerlo de los metadatos del GeoTIFF (tags TIFF estándar o específicos de satélites).
3.  Si no, intentar extraerlo del nombre del archivo usando expresiones regulares (patrones comunes como `YYYYMMDD_HHMMSS` o `YYYYjjjHHMM`).
4.  Si todo falla, no dibujar timestamp y avisar en debug.

## 4. Inicialización y Configuración de MapDrawer
En la función `main()`, después de cargar la imagen y antes de guardarla:

1.  **Instanciar MapDrawer:**
    *   Pasar el CRS detectado en el paso 2 al constructor de `MapDrawer`.
2.  **Vincular Imagen:**
    *   `mapper.set_image(img)`
3.  **Configurar Límites (Bounds):**
    *   Convertir los bounds de rasterio (`left, bottom, right, top`) al formato de MapDrawer (`ulx, uly, lrx, lry`).
    *   `ulx = left`, `uly = top`, `lrx = right`, `lry = bottom`.
    *   Llamar a `mapper.set_bounds(...)`.

## 5. Dibujo de Elementos
Ejecutar las funciones de dibujo según los argumentos proporcionados:

1.  **Capas Vectoriales:**
    *   Iterar sobre los argumentos `--layer`.
    *   Parsear `NOMBRE:COLOR:GROSOR`.
    *   Llamar a `mapper.draw_layer(...)`.
2.  **Logo:**
    *   Si `--logo-pos` está definido, llamar a `mapper.draw_logo(...)`.
3.  **Timestamp:**
    *   Si se determinó un texto de fecha válido, llamar a `mapper.draw_fecha(...)`.

## 6. Flujo de Trabajo Resultante
1.  Parsear argumentos.
2.  Cargar CPT (si aplica).
3.  Cargar GeoTIFF -> Obtener `img` y `metadata`.
4.  Aplicar paleta a `img` (si es modo 'L').
5.  **[NUEVO]** Inicializar `MapDrawer` con `metadata`.
6.  **[NUEVO]** Dibujar capas, logo y fecha sobre `img`.
7.  Guardar imagen final.
