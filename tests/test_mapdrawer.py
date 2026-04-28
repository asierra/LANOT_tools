"""
Tests para mapdrawer.py

Cubre:
- MapDrawer.set_image() / set_bounds() sin proyección
- draw_fecha() modifica la imagen (no lanza excepción, imagen no toda negra)
- draw_logo() sin logo en disco no lanza excepción
- overlay_glm() delega en render_glm_layer() con los parámetros correctos
  y compone la capa sobre self.image
"""

import os
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mapdrawer import MapDrawer
from metadata import Metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_image(w=200, h=100, color=(50, 50, 50)):
    """Imagen PIL RGB rellena de un color sólido."""
    return Image.new('RGB', (w, h), color)


def _rgba_layer(w=200, h=100, color=(255, 255, 0, 128)):
    """Imagen RGBA con color sólido semitransparente."""
    return Image.new('RGBA', (w, h), color)


def _meta_with_bounds():
    return Metadata(
        crs='goes18',
        bounds=(-3627271.29, 1583174.66, 1382771.93, 4589200.59),
        satellite='GOES-18',
        timestamp='2026:04:28 19:15:00',
        image_size=(200, 100),
    )


# ---------------------------------------------------------------------------
# set_image / set_bounds
# ---------------------------------------------------------------------------

class TestSetters:
    def test_set_image(self):
        mapper = MapDrawer()
        img = _solid_image()
        mapper.set_image(img)
        assert mapper.image is img

    def test_set_bounds_updates_bounds_dict(self):
        mapper = MapDrawer()
        mapper.set_bounds(-130.0, 50.0, -60.0, 20.0)
        assert mapper.bounds['ulx'] == pytest.approx(-130.0)
        assert mapper.bounds['uly'] == pytest.approx(50.0)
        assert mapper.bounds['lrx'] == pytest.approx(-60.0)
        assert mapper.bounds['lry'] == pytest.approx(20.0)

    def test_image_none_by_default(self):
        mapper = MapDrawer()
        assert mapper.image is None


# ---------------------------------------------------------------------------
# draw_fecha
# ---------------------------------------------------------------------------

class TestDrawFecha:
    def test_no_exception_with_string(self):
        mapper = MapDrawer()
        mapper.set_image(_solid_image())
        # No debe lanzar excepción
        mapper.draw_fecha("2026/04/28 19:27Z", position=2, fontsize=15)

    def test_no_exception_with_datetime(self):
        from datetime import datetime
        mapper = MapDrawer()
        mapper.set_image(_solid_image())
        dt = datetime(2026, 4, 28, 19, 27)
        mapper.draw_fecha(dt, position=0, fontsize=15)

    def test_image_modified(self):
        """La imagen no debe ser idéntica tras draw_fecha (se escribió texto)."""
        mapper = MapDrawer()
        img = _solid_image(color=(50, 50, 50))
        mapper.set_image(img)
        original = np.array(img.copy())
        mapper.draw_fecha("2026/04/28 19:27Z", position=2, fontsize=20, color='white')
        after = np.array(mapper.image)
        assert not np.array_equal(original, after), \
            "La imagen no cambió tras draw_fecha"

    def test_no_exception_without_image(self):
        """Llamar draw_fecha sin imagen no debe lanzar excepción."""
        mapper = MapDrawer()
        mapper.draw_fecha("2026/04/28 19:27Z")

    @pytest.mark.parametrize("position", [0, 1, 2, 3])
    def test_all_positions(self, position):
        mapper = MapDrawer()
        mapper.set_image(_solid_image(w=400, h=200))
        mapper.draw_fecha("2026/04/28 19:27Z", position=position, fontsize=15)


# ---------------------------------------------------------------------------
# draw_logo
# ---------------------------------------------------------------------------

class TestDrawLogo:
    def test_no_exception_when_logo_missing(self):
        """Si el logo no existe en el path, no debe lanzar excepción."""
        mapper = MapDrawer(lanot_dir='/no/existe')
        mapper.set_image(_solid_image())
        # Sólo imprime advertencia; no debe lanzar
        mapper.draw_logo(position=3)

    def test_no_exception_without_image(self):
        mapper = MapDrawer(lanot_dir='/no/existe')
        mapper.draw_logo(position=0)


# ---------------------------------------------------------------------------
# overlay_glm
# ---------------------------------------------------------------------------

class TestOverlayGlm:
    def test_calls_render_glm_layer_with_correct_args(self):
        """overlay_glm debe llamar a render_glm_layer con los archivos y metadata."""
        mapper = MapDrawer()
        img = _solid_image()
        mapper.set_image(img)
        meta = _meta_with_bounds()

        glm_files = ['a.nc', 'b.nc']
        fake_layer = _rgba_layer(w=200, h=100, color=(255, 255, 0, 100))

        # render_glm_layer se importa localmente en overlay_glm, hay que parchear
        # el módulo glm_renderer directamente, no mapdrawer
        with patch('glm_renderer.render_glm_layer', return_value=fake_layer) as mock_render:
            mapper.overlay_glm(glm_files, meta, color=(255, 255, 0))

        mock_render.assert_called_once()
        call_args = mock_render.call_args
        assert call_args[0][0] == glm_files
        assert call_args[0][1] is meta
        assert call_args[1]['base_color'] == (255, 255, 0)

    def test_image_size_injected_into_metadata(self):
        """overlay_glm debe inyectar image_size en metadata antes de llamar al renderer."""
        mapper = MapDrawer()
        img = _solid_image(w=300, h=150)
        mapper.set_image(img)
        meta = _meta_with_bounds()
        meta.pop('image_size', None)  # eliminar si existía

        with patch('glm_renderer.render_glm_layer', return_value=None):
            mapper.overlay_glm(['fake.nc'], meta)

        assert meta.get('image_size') == (300, 150)

    def test_image_composited_when_layer_returned(self):
        """Si render_glm_layer devuelve una capa, la imagen debe cambiar."""
        mapper = MapDrawer()
        base_color = (50, 50, 50)
        img = _solid_image(w=200, h=100, color=base_color)
        mapper.set_image(img)
        meta = _meta_with_bounds()

        # Capa completamente amarilla y semitransparente
        fake_layer = _rgba_layer(w=200, h=100, color=(255, 255, 0, 200))

        with patch('glm_renderer.render_glm_layer', return_value=fake_layer):
            mapper.overlay_glm(['fake.nc'], meta)

        result_arr = np.array(mapper.image.convert('RGB'))
        # La imagen resultante debe diferir del fondo gris oscuro
        assert result_arr.mean() > np.array(img.convert('RGB')).mean()

    def test_image_unchanged_when_layer_is_none(self):
        """Si render_glm_layer devuelve None, la imagen no debe cambiar."""
        mapper = MapDrawer()
        img = _solid_image()
        mapper.set_image(img)
        original_arr = np.array(img.copy())
        meta = _meta_with_bounds()

        with patch('glm_renderer.render_glm_layer', return_value=None):
            mapper.overlay_glm(['fake.nc'], meta)

        after_arr = np.array(mapper.image.convert('RGB'))
        assert np.array_equal(original_arr, after_arr)

    def test_no_exception_without_image(self):
        """overlay_glm sin imagen establecida no debe lanzar excepción."""
        mapper = MapDrawer()
        meta = _meta_with_bounds()
        # No se llega a importar render_glm_layer porque la guardia de imagen sale antes
        mapper.overlay_glm(['fake.nc'], meta)

    def test_glm_renderer_import_error_handled(self):
        """Si glm_renderer no se puede importar, no debe lanzar excepción."""
        mapper = MapDrawer()
        mapper.set_image(_solid_image())
        meta = _meta_with_bounds()

        with patch.dict('sys.modules', {'glm_renderer': None}):
            # ImportError al intentar from glm_renderer import render_glm_layer
            # El método debe capturarlo gracefully
            try:
                mapper.overlay_glm(['fake.nc'], meta)
            except ImportError:
                pytest.fail("overlay_glm lanzó ImportError en lugar de manejarlo")
