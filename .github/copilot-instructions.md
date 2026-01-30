# LANOT_tools - AI Coding Agent Instructions

## Project Overview
LANOT_tools is a suite for processing and visualizing GeoTIFF satellite imagery for LANOT (Laboratorio Nacional de Observación de la Tierra). The core architecture consists of four tightly-coupled Python modules that work as both CLI tools and importable libraries.

## Core Components

### 1. **metadata.py** - Metadata Container (NEW)
Lightweight dict-based wrapper for managing GeoTIFF metadata throughout processing pipelines.

**Key features:**
- Dict-like interface: `metadata['key']`, `metadata.get('key')`, `'key' in metadata`
- Helper methods: `get_mapdrawer_bounds()` converts rasterio bounds format to MapDrawer format
- Factory methods: `from_rasterio(src)`, `from_json_file(path)`, `from_dict(data)`
- Serialization: `save_json(path)`, `to_dict()`

**Usage pattern:**
```python
# Extract from GeoTIFF
with rasterio.open(file) as src:
    metadata = Metadata.from_rasterio(src)

# Add arbitrary fields
metadata['nodata_mask'] = mask
metadata['units'] = 'K'

# Use helpers
ulx, uly, lrx, lry = metadata.get_mapdrawer_bounds()

# Serialize for sidecar files
metadata.save_json('data.json')

# Load from sidecar
metadata = Metadata.from_json_file('data.json')
```

### 2. **geotiff2view.py** - GeoTIFF to Image Converter
Primary entry point for converting GeoTIFF satellite data to viewable images (PNG/JPEG). Handles:
- Single-band grayscale with optional color palettes (CPT files)
- RGB composites from 3 separate GeoTIFF files
- NoData transparency and special value handling
- Metadata extraction from TIFF tags (timestamp, satellite, CRS, bounds)

**Key patterns:**
- Returns tuple `(image, metadata)` from `load_geotiff()` - metadata is now a Metadata instance
- Uses `Metadata.from_rasterio(src)` for automatic metadata extraction
- Supports auto-scaling normalized (0-1) float data to palette range via `--autoscale`

### 3. **mapdrawer.py** - Map Overlay System
Draws vector layers, logos, legends, and timestamps over images. Core class is `MapDrawer`.

**Projection handling:**
- Uses `pyproj` for coordinate transformations (WGS84 lat/lon → projected CRS)
- Predefined GOES satellite projections: `goes16`, `goes17`, `goes18`, `goes19`
- Supports EPSG codes and Proj4 strings via `--crs` or `target_crs` parameter
- Fallback to linear projection if `pyproj` unavailable

**Critical workflow:**
```python
mapper = MapDrawer(target_crs='goes16')  # Initialize with projection
mapper.set_image(img)                     # Attach PIL Image
mapper.set_bounds(ulx, uly, lrx, lry)    # Geographic bounds (lon/lat)
mapper.draw_layer('COASTLINE', 'white', 1.0)
mapper.draw_logo(position=3)              # 0=UL, 1=UR, 2=LL, 3=LR
```

**Layer system:**
- Predefined layers: `COASTLINE`, `COUNTRIES`, `MEXSTATES`
- Reads from GeoPackage (`.gpkg`) or Shapefiles in `/usr/local/share/lanot/gpkg/`
- Implements shapefile caching via `_shp_cache` dict
- Boundary clipping for performance on large vector datasets

### 4. **colorpalettetable.py** - CPT Palette Manager
Handles GMT-style Color Palette Table (CPT) files for data visualization.

**CPT format support:**
- Discrete palettes: `value R G B [label]`
- Continuous palettes: `val1 r1 g1 b1 val2 r2 g2 b2`
- Special values: `B` (background), `F` (foreground), `N` (NoData)
- Normalized palettes (0-1 range) detected via `is_normalized` flag

**Key methods:**
- `get_pil_palette()` - Returns flat RGB list for PIL putpalette()
- `get_legend()` - Generates PIL Image with color bar and labels
- Handles offset/scale transformations for physical units (e.g., Kelvin → Celsius)

**Included CPT palettes:**
- `sst.cpt` - Sea Surface Temperature (268-310K, continuous gradient)
- `cld_temp_acha.cpt` - Cloud-top Temperature from AWG Cloud Height Algorithm (180-295K, 10 discrete classes)
- `phase.cpt` - Cloud phase classification (0-5 discrete: clear/water/supercooled/mixed/ice/unknown)
- `rainbow.cpt` - General-purpose normalized gradient (0.0-1.0)

## Installation & Deployment

**Server installation pattern:**
- Installs to `/opt/lanot-tools/` with isolated virtualenv
- Creates global wrapper scripts in `/usr/local/bin/` (mapdrawer, geotiff2view)
- Resources (CPT files, logos, vector data) in `/usr/local/share/lanot/`
- Use `sudo ./install.sh` for full setup, `./uninstall.sh` to remove

**Development mode:**
```bash
pip install -e .  # Changes reflect immediately without reinstall
```

## File Structure Conventions
- Entry points defined in [setup.py](setup.py) `console_scripts`
- Global resource paths: `/usr/local/share/lanot/colortables/` (CPT), `/usr/local/share/lanot/gpkg/` (vector)
- Fallback search: Check local path, then global installation directories

## Command-Line Patterns

Both tools support chained processing:
```bash
# GeoTIFF → colored image with map overlays
geotiff2view input.tif --cpt sst.cpt --alpha \
  --layer COASTLINE:white:1.0 --logo-pos 3 -o output.png

# Post-process existing image
mapdrawer image.png --recorte conus --crs goes16 \
  --layer COUNTRIES:gray:0.5 --timestamp "2026-01-30 12:00 UTC"
```

**Critical CLI arguments:**
- `--layer NAME:COLOR:WIDTH` - Can be specified multiple times
- `--logo-pos`, `--timestamp-pos`, `--legend-pos` use 0-3 corner indices
- `--recorte` - Predefined regions (`conus`, `fulldisk`) load from `PREDEFINED_REGIONS`
- `--scale` - Resize factor applies before all drawing operations

## Integration Points

**geotiff2view → mapdrawer integration:**
Geotiff2view instantiates MapDrawer internally when `--layer`, `--logo-pos`, or `--legend-pos` specified:
1. Extracts CRS/bounds from GeoTIFF metadata via `Metadata.from_rasterio(src)`
2. Uses `metadata.get_mapdrawer_bounds()` for automatic bounds format conversion
3. Passes metadata to MapDrawer for geo-referenced drawing

**External metadata JSON:**
Both tools support `--metadata file.json` for non-GeoTIFF images using `Metadata.from_json_file()`:
```json
{
  "crs": "goes16",
  "bounds": [minx, miny, maxx, maxy],
  "timestamp": "2026-01-30T12:00:00Z",
  "satellite": "GOES-16"
}
```

## NoData Handling Strategy
- Float data: Check for NaN and explicit nodata values from rasterio
- Integer data: Compare against nodata value
- CPT files define NoData index (`N` or `B` → `n_idx`, `f_idx`)
- Transparency: `--alpha` creates alpha channel where nodata mask is True
- When combining RGB: Union of all channel masks determines final transparency

## Debugging
Enable verbose output with `--verbose` or `-v` to see:
- Coordinate transformation samples
- Bounds calculations for projected CRS
- CPT loading details and palette ranges
- Vector layer clipping statistics

Set global `VERBOSE = True` to trigger `debug_msg()` calls throughout.

## Dependencies
**Required:** Pillow, fiona, pyproj, numpy
**Optional:** rasterio (GeoTIFF metadata extraction, highly recommended)

When optional dependencies missing, tools degrade gracefully:
- No rasterio → Uses PIL for image reading (loses geo-metadata)
- No pyproj → Linear projection only (no GOES/EPSG support)

## Current Development Focus
See [TODO.txt](TODO.txt) and [plan_integrar_md.md](plan_integrar_md.md) for:
- Refactoring duplicate CPT reading logic between geotiff2view/mapdrawer
- Histogram equalization and gamma correction for grayscale enhancement
- Processing multiple regions from single GeoTIFF without reloading
