# Plan: `--colorbar` en mapdrawer leyendo colormap del TIFF

## Contexto
- El TIFF `noaa20_viirs_CldTopTemp_20260508_192713_wgs84_fit.tif` tiene su propia paleta como tag `colormap` (formato `val,r,g,b` por líneas, pares de segmentos continuos), más `offset` y `scale`.
- `ColorPaletteTable` en `colorpalettetable.py` ya tiene `draw_legend()` + `_draw_colorbar()` para gradiente continuo.
- `mapdrawer.py` tiene `draw_legend()` (leyenda categórica) y `parse_cpt()` — pero NO importa `ColorPaletteTable` y NO tiene colorbar de gradiente.
- `mapdrawer.py` ya tiene rasterio disponible (`HAS_RASTERIO`) para leer tags del TIFF.

## Responsabilidad de cada módulo (límites claros)

| Módulo | Hace | No hace |
|---|---|---|
| `colorpalettetable.py` | Parsear y renderizar paletas (CPT files **y** strings de colormap TIFF) | No importa rasterio |
| `mapdrawer.py` | Lee tags del TIFF (ya tiene rasterio), orquesta decoraciones | No duplica lógica de colorbar |
| `geotiff2view.py` | Sin cambios | — |

---

## Fase 1 — Refactor interno en `colorpalettetable.py`

1. Extraer la lógica de `load()` que convierte `self.segments` → paleta densa en un método privado `_build_palette_from_segments()`. Esto evita duplicar esa lógica en el factory.

## Fase 2 — Factory `ColorPaletteTable.from_tiff_colormap()` en `colorpalettetable.py`

2. Agregar `@classmethod from_tiff_colormap(cls, colormap_str, tiff_offset=None, tiff_scale=None)`:
   - Parsea el string multilinea `val,r,g,b` que devuelve `src.tags()['colormap']`
   - Agrupa en pares de líneas → `segments = [(v1,r1,g1,b1, v2,r2,g2,b2), ...]` con `v_high > v_low`
   - Calcula `min_val`, `max_val`, `offset`, `scale_factor` como lo hace `load()` para paletas continuas
   - Llama `_build_palette_from_segments()` para generar `self.palette`
   - **Sin ningún `import rasterio`** — recibe solo strings/floats

El formato del tag (visto con rasterio):
```
300.000000,127,0,127
293.333344,127,0,127
293.333344,150,75,20
286.666656,150,75,20
...
```
Cada par de líneas = un segmento continuo: `(v_high, r,g,b) → (v_low, r,g,b)`.

## Fase 3 — Argumento `--colorbar` en `mapdrawer.py`

3. Agregar al inicio: `from colorpalettetable import ColorPaletteTable` (con `try/except` graceful).

4. Agregar argumento `--colorbar` (`action='store_true'`) al parser.

5. En `main()`, si `--colorbar`, construir `cpt_obj` con esta prioridad:
   - **Primero**: si el input es TIFF y rasterio disponible, leer tag `colormap` via `src.tags()` → `ColorPaletteTable.from_tiff_colormap(colormap_str, tiff_offset, tiff_scale)`
   - **Fallback**: si no hay tag `colormap` pero se pasó `--cpt` → `ColorPaletteTable(cpt_path)`
   - **Si ninguno**: advertencia y saltar sin dibujar
   - Dibujar **al final** (después de layers/logo/timestamp, antes de guardar):
     ```python
     barsz = img.height // 20
     cpt_obj.draw_legend(ImageDraw.Draw(img), 0, img.height - 2*barsz, img.width, barsz, font_size=barsz//2)
     ```

6. `--colorbar` y `--cpt` son ortogonales:
   - `--colorbar` solo → usa colormap del TIFF (si existe)
   - `--colorbar --cpt archivo.cpt` → usa el CPT externo (fallback)
   - `--cpt archivo.cpt` solo (sin `--colorbar`) → sigue funcionando para leyenda categórica (recuadros)

---

## Archivos a modificar
- `colorpalettetable.py` — Fases 1 y 2 (refactor `_build_palette_from_segments` + factory `from_tiff_colormap`)
- `mapdrawer.py` — Fase 3 (import CPT, argumento `--colorbar`, lógica en `main()`)
- `geotiff2view.py` — **Sin cambios**

## Scope excluido
- No se porta `make_south_room` a `mapdrawer.py`
- No se modifica la leyenda categórica `draw_legend()` de `mapdrawer`

## Uso esperado

**Con TIFF que tiene `colormap` embebido:**
```bash
mapdrawer noaa20_viirs_CldTopTemp_20260508_192713_wgs84_fit.tif \
  --colorbar --layer COASTLINE:white:1.0 --timestamp-pos 2 --logo-pos 3 -o output.png
```

**Con TIFF/PNG sin `colormap`, asignando paleta externa:**
```bash
mapdrawer imagen.tif --cpt cld_temp_acha.cpt --colorbar \
  --layer COASTLINE:white:1.0 -o output.png
```

## Verificación
1. Correr el comando con `--colorbar` sobre el TIFF de prueba
2. Verificar visualmente que el gradiente va de 300K (izquierda) a 200K (derecha) con los colores correctos
3. `python -m pytest tests/` para detectar regresiones
