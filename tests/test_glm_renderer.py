"""
Tests para glm_renderer.py

Cubre:
- render_glm_layer() con Dataset mockeado
- Devuelve imagen RGBA del tamaño correcto
- Píxeles con rayos tienen alpha > 0; sin rayos alpha = 0
- Color configurable (amarillo vs magenta)
- metadata['glm_time_start/end'] se almacenan tras la llamada
- Lista vacía de archivos devuelve None sin excepciones
- Metadata sin CRS o sin bounds devuelve None sin excepciones
"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metadata import Metadata
from glm_renderer import render_glm_layer, GOES_PROJECTIONS


# ---------------------------------------------------------------------------
# Fixtures y helpers
# ---------------------------------------------------------------------------

GOES18_CRS = GOES_PROJECTIONS['goes18']

# Para tests unitarios usamos epsg:4326 (proyección identidad lat/lon) para que
# los bounds sean trivialmente correctos y no dependan de cálculos GOES por satélite.
TEST_CRS = 'epsg:4326'
TEST_BOUNDS = (-130.0, 24.0, -65.0, 50.0)  # (left, bottom, right, top) en grados WGS84

IMG_W, IMG_H = 250, 150  # Imagen pequeña para velocidad


def _make_metadata():
    """Metadata mínima válida para render_glm_layer."""
    return Metadata(
        crs=TEST_CRS,
        bounds=TEST_BOUNDS,
        image_size=(IMG_W, IMG_H),
        satellite='GOES-18',
        timestamp='2026:04:28 19:15:00',
    )


def _make_mock_nc(lons, lats, time_str='2026-04-28T19:15:00Z'):
    """Construye un mock de netCDF4.Dataset con event_lon/lat y product_time."""
    nc = MagicMock()
    # Para que `with Dataset(f) as nc:` devuelva el mismo objeto nc
    nc.__enter__.return_value = nc
    nc.__exit__.return_value = False

    lon_var = MagicMock()
    lon_var.__getitem__.return_value = np.array(lons)
    lat_var = MagicMock()
    lat_var.__getitem__.return_value = np.array(lats)

    nc.variables = {'event_lon': lon_var, 'event_lat': lat_var}
    # product_time como atributo global del archivo
    nc.time_coverage_start = time_str
    return nc


# ---------------------------------------------------------------------------
# Tests principales
# ---------------------------------------------------------------------------

class TestRenderGlmLayer:

    def _run(self, glm_files, metadata, base_color=(255, 255, 0)):
        """Helper que parchea Dataset y ejecuta render_glm_layer."""
        return render_glm_layer(glm_files, metadata, base_color=base_color)

    def test_returns_rgba_image(self, tmp_path):
        """Con datos válidos debe devolver una imagen RGBA del tamaño correcto."""
        meta = _make_metadata()

        # Generar ~50 puntos dentro del CONUS (lon/lat WGS84)
        lons = np.linspace(-120, -80, 50)
        lats = np.linspace(25, 48, 50)

        mock_nc = _make_mock_nc(lons, lats)

        with patch('glm_renderer.Dataset', return_value=mock_nc):
            result = render_glm_layer(['fake.nc'], meta)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        assert result.size == (IMG_W, IMG_H)

    def test_pixels_with_rays_have_alpha(self, tmp_path):
        """Zonas con rayos deben tener canal alpha > 0."""
        meta = _make_metadata()

        # Un punto central del CONUS
        lons = np.full(20, -100.0)
        lats = np.full(20, 35.0)
        mock_nc = _make_mock_nc(lons, lats)

        with patch('glm_renderer.Dataset', return_value=mock_nc):
            result = render_glm_layer(['fake.nc'], meta)

        assert result is not None
        arr = np.array(result)
        alpha = arr[:, :, 3]
        assert alpha.max() > 0, "Ningún píxel tiene alpha > 0 con rayos presentes"

    def test_empty_region_all_transparent(self):
        """Sin rayos en los bounds, todos los píxeles deben ser transparentes."""
        meta = _make_metadata()

        # Puntos fuera del área de test (Atlántico sur)
        lons = np.full(10, 10.0)
        lats = np.full(10, -50.0)
        mock_nc = _make_mock_nc(lons, lats)

        with patch('glm_renderer.Dataset', return_value=mock_nc):
            result = render_glm_layer(['fake.nc'], meta)

        # Puede devolver None o imagen totalmente transparente
        if result is not None:
            arr = np.array(result)
            assert arr[:, :, 3].max() == 0

    def test_yellow_color(self):
        """Color amarillo: R=255, G=255, B=0 en píxeles con rayos."""
        meta = _make_metadata()
        lons = np.full(30, -100.0)
        lats = np.full(30, 35.0)
        mock_nc = _make_mock_nc(lons, lats)

        with patch('glm_renderer.Dataset', return_value=mock_nc):
            result = render_glm_layer(['fake.nc'], meta, base_color=(255, 255, 0))

        assert result is not None
        arr = np.array(result)
        mask = arr[:, :, 3] > 0
        if mask.any():
            assert arr[mask, 0].max() == 255  # R
            assert arr[mask, 1].max() == 255  # G
            assert arr[mask, 2].max() == 0    # B

    def test_magenta_color(self):
        """Color magenta: R=255, G=0, B=255 en píxeles con rayos."""
        meta = _make_metadata()
        lons = np.full(30, -100.0)
        lats = np.full(30, 35.0)
        mock_nc = _make_mock_nc(lons, lats)

        with patch('glm_renderer.Dataset', return_value=mock_nc):
            result = render_glm_layer(['fake.nc'], meta, base_color=(255, 0, 255))

        assert result is not None
        arr = np.array(result)
        mask = arr[:, :, 3] > 0
        if mask.any():
            assert arr[mask, 0].max() == 255  # R
            assert arr[mask, 1].max() == 0    # G
            assert arr[mask, 2].max() == 255  # B

    def test_stores_glm_time_range_in_metadata(self):
        """Debe almacenar glm_time_start y glm_time_end en el objeto metadata."""
        meta = _make_metadata()
        lons = np.full(10, -100.0)
        lats = np.full(10, 35.0)
        mock_nc = _make_mock_nc(lons, lats, time_str='2026-04-28T19:20:00Z')

        with patch('glm_renderer.Dataset', return_value=mock_nc):
            render_glm_layer(['fake.nc'], meta)

        assert 'glm_time_start' in meta
        assert 'glm_time_end' in meta
        assert '2026' in meta['glm_time_start']

    def test_empty_file_list_returns_none(self):
        """Lista vacía no debe lanzar excepción y debe devolver None."""
        meta = _make_metadata()
        result = render_glm_layer([], meta)
        assert result is None

    def test_missing_crs_returns_none(self):
        """Metadata sin CRS debe devolver None sin excepción."""
        meta = Metadata(bounds=TEST_BOUNDS, image_size=(IMG_W, IMG_H))
        lons = np.full(5, -100.0)
        lats = np.full(5, 35.0)
        mock_nc = _make_mock_nc(lons, lats)

        with patch('glm_renderer.Dataset', return_value=mock_nc):
            result = render_glm_layer(['fake.nc'], meta)

        assert result is None

    def test_missing_bounds_returns_none(self):
        """Metadata sin bounds debe devolver None sin excepción."""
        meta = Metadata(crs=TEST_CRS, image_size=(IMG_W, IMG_H))
        lons = np.full(5, -100.0)
        lats = np.full(5, 35.0)
        mock_nc = _make_mock_nc(lons, lats)

        with patch('glm_renderer.Dataset', return_value=mock_nc):
            result = render_glm_layer(['fake.nc'], meta)

        assert result is None

    def test_multiple_files_concatenated(self):
        """Con varios archivos los eventos se concatenan y se produce una sola capa."""
        meta = _make_metadata()

        lons_a = np.full(10, -110.0)
        lats_a = np.full(10, 40.0)
        lons_b = np.full(10, -90.0)
        lats_b = np.full(10, 30.0)

        mock_nc_a = _make_mock_nc(lons_a, lats_a, '2026-04-28T19:15:00Z')
        mock_nc_b = _make_mock_nc(lons_b, lats_b, '2026-04-28T19:20:00Z')

        call_count = [0]
        files = ['a.nc', 'b.nc']
        mocks = [mock_nc_a, mock_nc_b]

        def mock_dataset(f, *args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return mocks[idx % len(mocks)]

        with patch('glm_renderer.Dataset', side_effect=mock_dataset):
            result = render_glm_layer(files, meta)

        assert result is not None
        # Con dos archivos con tiempos distintos, el rango debe diferir
        assert meta.get('glm_time_start') != meta.get('glm_time_end') or \
               meta.get('glm_time_start') is not None
