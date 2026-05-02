"""
Tests para ash_view_generator.py

Cubre:
- render_ash_layer() con formato nuevo (1 banda uint8 + colormap)
- render_ash_layer() con formato legacy (4 bandas RGBA)
- El valor 0 (fondo) siempre transparente aunque el colormap lo marque opaco
- Píxeles con detección tienen alpha > 0
- Superposición correcta sobre el canvas del tamaño de la base
- Ceniza fuera de los bounds devuelve None sin excepción
- Metadata sin bounds o sin image_size devuelve None sin excepción
- Sin rasterio devuelve None sin excepción
"""

import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metadata import Metadata
from ash_view_generator import render_ash_layer


# ---------------------------------------------------------------------------
# Constantes de prueba
# ---------------------------------------------------------------------------

# Bounds en metros (proyección GOES), imagen pequeña para velocidad
BASE_LEFT, BASE_BOTTOM, BASE_RIGHT, BASE_TOP = -3_000_000, 1_500_000, -1_500_000, 3_000_000
BASE_W, BASE_H = 100, 80

# Ceniza que cae justo en el centro de la imagen base
ASH_LEFT  = (BASE_LEFT  + BASE_RIGHT)  / 2 - 200_000
ASH_RIGHT = (BASE_LEFT  + BASE_RIGHT)  / 2 + 200_000
ASH_TOP   = (BASE_TOP   + BASE_BOTTOM) / 2 + 200_000
ASH_BOTTOM= (BASE_TOP   + BASE_BOTTOM) / 2 - 200_000
ASH_W, ASH_H = 20, 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata(**kwargs):
    defaults = dict(
        crs='epsg:4326',
        bounds=(BASE_LEFT, BASE_BOTTOM, BASE_RIGHT, BASE_TOP),
        image_size=(BASE_W, BASE_H),
    )
    defaults.update(kwargs)
    return Metadata(**defaults)


class _FakeBounds:
    def __init__(self, left, bottom, right, top):
        self.left   = left
        self.bottom = bottom
        self.right  = right
        self.top    = top


def _make_mock_src_1band(band_data, colormap_dict, nodata=255):
    """Mock de rasterio.DatasetReader para 1 banda uint8 + colormap."""
    src = MagicMock()
    src.__enter__ = lambda s: s
    src.__exit__ = MagicMock(return_value=False)
    src.count = 1
    src.crs = None
    src.bounds = _FakeBounds(ASH_LEFT, ASH_BOTTOM, ASH_RIGHT, ASH_TOP)
    src.read.return_value = band_data
    src.colormap.return_value = colormap_dict
    return src


def _make_mock_src_4band(r, g, b, a):
    """Mock de rasterio.DatasetReader para 4 bandas RGBA (formato legacy)."""
    src = MagicMock()
    src.__enter__ = lambda s: s
    src.__exit__ = MagicMock(return_value=False)
    src.count = 4
    src.crs = None
    src.bounds = _FakeBounds(ASH_LEFT, ASH_BOTTOM, ASH_RIGHT, ASH_TOP)
    src.read.return_value = np.array([r, g, b, a])
    return src


# Colormap mínimo que simula detect_ash.py: 0=fondo negro opaco, 1=rojo, 2=naranja, 3=amarillo
SAMPLE_COLORMAP = {
    0:   (0,   0,   0, 255),   # fondo — debe forzarse a transparente
    1:   (255, 0,   0, 255),   # ceniza
    2:   (255, 165, 0, 255),   # probable
    3:   (255, 255, 0, 255),   # posible
    255: (0,   0,   0,   0),   # nodata
}


# ---------------------------------------------------------------------------
# Tests: formato nuevo (1 banda + colormap)
# ---------------------------------------------------------------------------

class TestRenderAshLayer1Band:

    def _run(self, band_data, colormap=None, meta=None):
        if colormap is None:
            colormap = SAMPLE_COLORMAP
        if meta is None:
            meta = _make_metadata()
        mock_src = _make_mock_src_1band(band_data, colormap)
        with patch('rasterio.open', return_value=mock_src):
            return render_ash_layer('fake.tif', meta)

    def test_returns_rgba_image_correct_size(self):
        """Debe devolver imagen RGBA del tamaño de la imagen base."""
        band = np.ones((ASH_H, ASH_W), dtype=np.uint8)  # todo valor 1 (ceniza)
        result = self._run(band)
        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        assert result.size == (BASE_W, BASE_H)

    def test_ash_pixels_have_alpha(self):
        """Píxeles con clase 1 (ceniza) deben tener alpha > 0 en el canvas."""
        band = np.ones((ASH_H, ASH_W), dtype=np.uint8)
        result = self._run(band)
        assert result is not None
        arr = np.array(result)
        assert arr[:, :, 3].max() > 0

    def test_background_pixel_transparent(self):
        """El valor 0 (fondo) debe ser siempre transparente aunque el colormap diga 255."""
        band = np.zeros((ASH_H, ASH_W), dtype=np.uint8)  # todo fondo
        result = self._run(band)
        # Canvas puede ser todo transparente (ningún pixel de ceniza)
        if result is not None:
            arr = np.array(result)
            assert arr[:, :, 3].max() == 0, \
                "El fondo (valor 0) debe ser transparente pero tiene alpha > 0"

    def test_ash_color_red(self):
        """Píxeles de ceniza (clase 1) deben ser rojos en el canal RGB."""
        band = np.ones((ASH_H, ASH_W), dtype=np.uint8)
        result = self._run(band)
        assert result is not None
        arr = np.array(result)
        mask = arr[:, :, 3] > 0
        if mask.any():
            assert arr[mask, 0].max() == 255   # R
            assert arr[mask, 1].max() == 0     # G
            assert arr[mask, 2].max() == 0     # B

    def test_colormap_read_error_uses_empty_lut(self):
        """Si colormap() lanza excepción, la función no debe propagar el error."""
        meta = _make_metadata()
        band = np.ones((ASH_H, ASH_W), dtype=np.uint8)
        mock_src = _make_mock_src_1band(band, {})
        mock_src.colormap.side_effect = Exception("no colormap")
        with patch('rasterio.open', return_value=mock_src):
            result = render_ash_layer('fake.tif', meta)
        # Puede devolver imagen o None, pero nunca debe lanzar excepción
        # (el assert implícito es que llegamos aquí sin crash)


# ---------------------------------------------------------------------------
# Tests: formato legacy (4 bandas RGBA)
# ---------------------------------------------------------------------------

class TestRenderAshLayer4Band:

    def _run(self, r, g, b, a, meta=None):
        if meta is None:
            meta = _make_metadata()
        mock_src = _make_mock_src_4band(r, g, b, a)
        with patch('rasterio.open', return_value=mock_src):
            return render_ash_layer('fake.tif', meta)

    def test_returns_rgba_image_correct_size(self):
        r = np.full((ASH_H, ASH_W), 255, dtype=np.uint8)
        g = np.zeros((ASH_H, ASH_W), dtype=np.uint8)
        b = np.zeros((ASH_H, ASH_W), dtype=np.uint8)
        a = np.full((ASH_H, ASH_W), 255, dtype=np.uint8)
        result = self._run(r, g, b, a)
        assert result is not None
        assert result.mode == 'RGBA'
        assert result.size == (BASE_W, BASE_H)

    def test_transparent_pixels_stay_transparent(self):
        """Alpha=0 en ash no debe aparecer en el canvas."""
        r = np.full((ASH_H, ASH_W), 255, dtype=np.uint8)
        g = np.zeros((ASH_H, ASH_W), dtype=np.uint8)
        b = np.zeros((ASH_H, ASH_W), dtype=np.uint8)
        a = np.zeros((ASH_H, ASH_W), dtype=np.uint8)  # todo transparente
        result = self._run(r, g, b, a)
        if result is not None:
            arr = np.array(result)
            assert arr[:, :, 3].max() == 0


# ---------------------------------------------------------------------------
# Tests: manejo de errores y casos borde
# ---------------------------------------------------------------------------

class TestRenderAshLayerEdgeCases:

    def test_missing_bounds_returns_none(self):
        meta = Metadata(image_size=(BASE_W, BASE_H))
        band = np.ones((ASH_H, ASH_W), dtype=np.uint8)
        mock_src = _make_mock_src_1band(band, SAMPLE_COLORMAP)
        with patch('rasterio.open', return_value=mock_src):
            result = render_ash_layer('fake.tif', meta)
        assert result is None

    def test_missing_image_size_returns_none(self):
        meta = Metadata(bounds=(BASE_LEFT, BASE_BOTTOM, BASE_RIGHT, BASE_TOP))
        band = np.ones((ASH_H, ASH_W), dtype=np.uint8)
        mock_src = _make_mock_src_1band(band, SAMPLE_COLORMAP)
        with patch('rasterio.open', return_value=mock_src):
            result = render_ash_layer('fake.tif', meta)
        assert result is None

    def test_ash_outside_base_no_visible_pixels(self):
        """Ceniza totalmente fuera de los bounds de la base → None o canvas transparente."""
        meta = _make_metadata()
        # Ash muy lejos a la derecha
        far_src = _make_mock_src_1band(
            np.ones((ASH_H, ASH_W), dtype=np.uint8), SAMPLE_COLORMAP)
        far_src.bounds = _FakeBounds(BASE_RIGHT + 1_000_000,
                                     BASE_BOTTOM,
                                     BASE_RIGHT + 2_000_000,
                                     BASE_TOP)
        with patch('rasterio.open', return_value=far_src):
            result = render_ash_layer('fake.tif', meta)
        # La función puede devolver None (dimensión 0) o un canvas todo transparente
        if result is not None:
            arr = np.array(result)
            assert arr[:, :, 3].max() == 0, \
                "Ceniza fuera de bounds no debe dejar píxeles visibles en el canvas"

    def test_wrong_band_count_returns_none(self):
        """2 bandas (ni 1 ni ≥4) debe devolver None."""
        meta = _make_metadata()
        src = MagicMock()
        src.__enter__ = lambda s: s
        src.__exit__ = MagicMock(return_value=False)
        src.count = 2
        src.crs = None
        src.bounds = _FakeBounds(ASH_LEFT, ASH_BOTTOM, ASH_RIGHT, ASH_TOP)
        with patch('rasterio.open', return_value=src):
            result = render_ash_layer('fake.tif', meta)
        assert result is None

    def test_no_rasterio_returns_none(self):
        """Sin rasterio debe devolver None sin crash."""
        meta = _make_metadata()
        with patch('ash_view_generator.HAS_RASTERIO', False):
            result = render_ash_layer('fake.tif', meta)
        assert result is None
