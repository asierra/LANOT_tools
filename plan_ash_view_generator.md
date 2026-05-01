# ash_view_generator.py — Diseño e implementación

## Descripción
`ash_view_generator.py` en LANOT_tools: módulo con función `render_ash_layer()` + bloque
`__main__` standalone. Carga un GeoTIFF base ABI (banda única uint8), extrae su CRS/bounds
vía rasterio, superpone un GeoTIFF de ceniza RGBA generado por `detect_ash.py`, dibuja
capas vectoriales con `MapDrawer` y guarda la salida como PNG o GeoTIFF georeferenciado.

**Patrón de referencia:** `glm_renderer.py` (función módulo + CLI, composita RGBA sobre la base).
`mapdrawer.py` (LANOT_tools) se usa sin modificar.

**Archivos de prueba:**
- Base: `/home/aguilars/lanot/ceniza/201907271301-1902/C13-C15_20192081601.tif`
- Ceniza: `/home/aguilars/lanot/ceniza/ash2019208/ceniza_20192081601.tif`

---

## Formato real de los datos

El GeoTIFF de ceniza producido por `detect_ash.py --clip <region>` (sin sufijo `_geo`)
ya es una imagen **RGBA de 4 bandas** directamente coloreada según `ash.cpt`:
- Banda 1-3: RGB → `(255,0,0)` ash, `(255,165,0)` probable, `(255,255,0)` posible
- Banda 4: alpha → 255 en píxeles con detección, 0 en fondo

No hay mapa de clases separado ni colormap embebido en el TIF. La leyenda usa
colores hardcoded que coinciden con `ash.cpt`.

> **Nota:** el sufijo `_geo` indica que `detect_ash.py` reproyectó a EPSG:4326 (lat/lon).
> Para este script usar la salida **sin** `_geo`, que conserva la proyección GOES original.

---

## Phase 1: Función módulo `render_ash_layer(ash_tif, metadata)`

**1. Leer ash GeoTIFF con rasterio**
- Verificar que tiene ≥ 4 bandas
- `src.read([1,2,3,4])` → arrays R, G, B, A
- `src.bounds` → ash_bounds en coordenadas proyectadas
- `src.crs` → para validar compatibilidad con la base

**2. Verificar compatibilidad de CRS**
- Comparar `src.crs` del ash con `metadata.get('crs')` (CRS de la base)
- Usar `pyproj.CRS.from_user_input(x).equals(y)` — comparación **semántica**, no textual
  (los WKT de ambos TIFs pueden diferir en precisión numérica del esferoide aunque
  representen la misma proyección GOES)
- Si son distintos: imprimir error con ambos CRS y `return None`
- Sin pyproj: advertencia y continuar

**3. Mapear ash sobre espacio de la imagen base (mapeo lineal)**
- `metadata.get('bounds')` → `(left, bottom, right, top)` en metros proyectados
- `metadata.get('image_size')` → `(base_w, base_h)`
- Calcular posición y tamaño en píxeles:
  - `x_offset = (ash_left - base_left) / base_span_x * base_w`
  - `y_offset = (base_top  - ash_top)  / base_span_y * base_h`
  - `ash_w_px = (ash_right - ash_left) / base_span_x * base_w`
  - `ash_h_px = (ash_top - ash_bottom) / base_span_y * base_h`
- Construir PIL RGBA del ash con `np.stack([R,G,B,A])`
- Redimensionar a `(ash_w_px, ash_h_px)` con `PIL.NEAREST`
- Pegar en canvas RGBA del tamaño de la base → devolver canvas

**Firma:** `render_ash_layer(ash_tif, metadata) → PIL RGBA Image | None`

---

## Phase 2: Script CLI (`__main__`)

**Argumentos CLI**
- Posicionales: `base_tif`, `ash_tif`
- `-o / --output` (default: `ash_output.png`; si termina en `.tif/.tiff` → GeoTIFF)
- `--layer NAME:COLOR:WIDTH` (repetible; ej: `MEXSTATES:white:1.0`)
- `--logo-pos 0-3`
- `--legend-pos 0-3`
- `--scale FLOAT` (factor de escala, default: 1.0)
- `--crs` (override CRS; opcional, por defecto toma del GeoTIFF base)

**Cargar imagen base y metadata**
- `rasterio.open(base_tif)` + `Metadata.from_rasterio(src)` → CRS y bounds
- Leer banda única con `src.read(1)`
- **Normalización min-max al rango 0–255**: los visores (QGIS, etc.) aplican
  este mismo stretch automáticamente a TIFs de banda única; al producir RGB
  hay que hacerlo explícitamente para reproducir la apariencia visual esperada
- Convertir a PIL RGB → `metadata['image_size'] = (base_w, base_h)`

**Composite de ceniza**
- `render_ash_layer(ash_tif, metadata)` → RGBA canvas
- `Image.alpha_composite(base_img.convert('RGBA'), ash_layer).convert('RGB')`

**MapDrawer para vectores y decoraciones**
- `mapper = MapDrawer(target_crs=metadata.get('crs'))`
- `mapper.set_image(result)`
- Detección de bounds proyectados vs. geográficos:
  - Si `|ulx| > 360` o `|uly| > 90` → bounds en metros → `mapper.set_projected_bounds()`
  - Si no → `mapper.set_bounds(*metadata.get_mapdrawer_bounds())`
  - **Crítico:** pasar bounds en metros directamente a `set_bounds()` rompe el clipping
    de shapefiles (que trabajan en lat/lon); hay que usar `set_projected_bounds()`.
- `mapper.draw_layer(name, color, width)` para cada `--layer`
- `mapper.draw_logo()` y `mapper.draw_legend(ASH_LEGEND_ITEMS)` si se piden

**Guardar salida**
- Extensión `.png/.jpg` → `PIL Image.save()`
- Extensión `.tif/.tiff` → `rasterio.open(..., driver='GTiff')` con CRS y
  `rasterio.transform.from_bounds(...)` para preservar la georeferencia

---

## Archivos relevantes
- `glm_renderer.py` — plantilla de estructura (función módulo + CLI)
- `mapdrawer.py` — API usada: `MapDrawer(target_crs)`, `set_image`, `set_bounds`,
  `set_projected_bounds`, `draw_layer`, `draw_logo`, `draw_legend(items)`
- `metadata.py` — `Metadata.from_rasterio(src)`, `get_mapdrawer_bounds()` → (ulx, uly, lrx, lry)

## Verificación
```bash
python ash_view_generator.py \
  .../C13-C15_20192081601.tif \
  .../ceniza_20192081601.tif \
  -o test.png --layer MEXSTATES:white:1.0 --legend-pos 2 --logo-pos 3

python -c "from ash_view_generator import render_ash_layer; print('import OK')"
```

## Decisiones y hallazgos
- Sin modificar `mapdrawer.py` ni ningún módulo existente
- El ash GeoTIFF ya es RGBA coloreado; **no hay** mapa de clases ni `colormap()` en el TIF
- Comparación de CRS con `CRS.equals()` (semántica), no por WKT string (falsos negativos
  por diferencias de precisión numérica entre TIFs del mismo pipeline)
- Bounds GOES en metros requieren `set_projected_bounds()` en MapDrawer, no `set_bounds()`
- Normalización min-max de la banda base reproduce el auto-stretch de QGIS en RGB
- Salida GeoTIFF: `rasterio.open(...driver='GTiff')` con mismos CRS y bounds que la base
