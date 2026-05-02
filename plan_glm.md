# glm_renderer.py — Diseño e implementación

## Estado: IMPLEMENTADO ✓

Módulo `glm_renderer.py` con función `render_glm_layer()` + bloque `__main__` standalone.
Integrado también como opción `--glm` dentro de `mapdrawer.py`.

---

## Arquitectura implementada

### 1. Ingesta vectorizada (NetCDF4)
Lee la lista de NetCDFs GLM, extrae `event_lon` y `event_lat` de cada archivo con
`nc.variables[...][:]`, y usa `numpy.concatenate` para obtener dos arreglos
unidimensionales con todos los eventos de la ventana temporal completa.

También extrae el rango temporal buscando los atributos globales
`product_time`, `time_coverage_start` o `event_time_offset` (en ese orden de
preferencia), y almacena `glm_time_start` / `glm_time_end` en el objeto `metadata`.

### 2. Transformación en bloque (PyProj)
Una sola instancia de `Transformer.from_crs("epsg:4326", crs_str, always_xy=True)`,
llamada una vez sobre los arreglos completos. Sin bucles sobre eventos individuales.
Los puntos fuera de los bounds se filtran con una máscara booleana numpy.

### 3. Histograma 2D → densidad FED (`numpy.histogram2d`)
```python
density, _, _ = np.histogram2d(x_proj, y_proj, bins=[x_bins, y_bins])
density = np.flipud(density.T)  # Y: fila 0 = top de imagen
```
Produce un campo de densidad de eventos por píxel (Flash Event Density) en memoria,
sin loops, en milisegundos.

### 4. Glow estilo CIRA (`scipy.ndimage.gaussian_filter` + alpha logarítmico)
El problema de alpha lineal es que en tormentas activas los picos de densidad
(50-100 eventos/píxel) se amplifican y empasta el overlay, tapando las nubes.
Solución adoptada:

```python
smooth_density = gaussian_filter(density, sigma=2.0)
mask = smooth_density > 0.15
rgba_array[mask, :3] = base_color
alpha = np.clip(np.log1p(smooth_density[mask]) * 60, 0, 200).astype(np.uint8)
rgba_array[mask, 3] = alpha
```

- `gaussian_filter(sigma=2.0)` expande puntos a manchas de luz suaves
- `np.log1p` comprime picos extremos (1 evento → alpha ≈ 41; 50 eventos → alpha topa en 200)
- Tope en **200** (no 255) garantiza que la textura de nubes sea siempre visible

**Fallback sin scipy:** alpha lineal `density * 40`, tope en 250.

### 5. Composición final (PIL)
```python
Image.fromarray(rgba_array, 'RGBA')
# El llamador hace:
result = Image.alpha_composite(base_img.convert('RGBA'), glm_layer)
```

### 6. Integración en mapdrawer.py
`mapdrawer.py` acepta `--glm FILE [FILE ...]` y `--glm-color COLOR` (yellow/magenta/white).
Después de dibujar capas vectoriales llama a `render_glm_layer()` y compone la capa
sobre la imagen antes de dibujar logo/timestamp.

---

## Firma pública

```python
render_glm_layer(glm_files, metadata, base_color=(255, 255, 0)) → PIL.Image | None
```

- `glm_files`: lista de rutas NetCDF GLM
- `metadata`: instancia `Metadata` con `'crs'`, `'bounds'`; opcionalmente `'image_size'`
- `base_color`: RGB del glow (default amarillo CIRA)
- Almacena `metadata['glm_time_start']` y `metadata['glm_time_end']` como efecto lateral

## Uso CLI standalone

```bash
glm_renderer.py base.tif archivo1.nc archivo2.nc ... -o salida.png --color yellow
```

---

## Decisiones de diseño y hallazgos

| Decisión | Razón |
|---|---|
| Alpha lineal → logarítmico | Alpha lineal empastaba completamente las nubes en tormentas activas |
| Tope alpha en 200 (no 255) | Siempre queda visible la textura subyacente de las nubes |
| sigma=2.0 para Gaussian | Radio ~2px da glow natural sin sobredifusión |
| Umbral `> 0.15` post-Gaussian | Elimina ruido de píxeles casi vacíos que generaban halos finos |
| Metadata como efecto lateral | Permite que `mapdrawer` arme el timestamp ABI/GLM unificado |

## Dependencias

- **Requeridas:** numpy, netCDF4, pyproj, Pillow
- **Opcional:** scipy (sin ella: fallback a alpha lineal sin Gaussian blur)
