"""
Tests para colorpalettetable.py

Cubre:
- Carga sin excepciones de todos los .cpt incluidos en el repo
- Propiedades básicas: palette (768 valores), min_val, max_val
- is_normalized True para rainbow.cpt, False para sst.cpt
- n_idx definido en phase.cpt (tiene valor N)
- apply_to_data() devuelve array uint8 dentro de rango
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from colorpalettetable import ColorPaletteTable

REPO_DIR = os.path.join(os.path.dirname(__file__), '..')

# Rutas a los CPT incluidos en el repo
CPT_FILES = [
    'sst.cpt',
    'cld_temp_acha.cpt',
    'cld_height_acha.cpt',
    'cld_emiss.cpt',
    'cloud_type.cpt',
    'phase.cpt',
    'viirs_confidence_cat.cpt',
    'rainbow.cpt',
]


# ---------------------------------------------------------------------------
# Carga básica de todos los CPT del repo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cpt_name", CPT_FILES)
def test_load_no_exception(cpt_name):
    path = os.path.join(REPO_DIR, cpt_name)
    if not os.path.exists(path):
        pytest.skip(f"{cpt_name} no encontrado en el repo")
    cpt = ColorPaletteTable(path)
    # La carga no debe lanzar excepción y debe tener algún dato
    assert cpt.palette is not None or cpt.colors or cpt.segments


@pytest.mark.parametrize("cpt_name", CPT_FILES)
def test_palette_length(cpt_name):
    """get_pil_palette() debe devolver lista de 768 enteros (256 * 3 canales)."""
    path = os.path.join(REPO_DIR, cpt_name)
    if not os.path.exists(path):
        pytest.skip(f"{cpt_name} no encontrado en el repo")
    cpt = ColorPaletteTable(path)
    palette = cpt.get_pil_palette()
    assert palette is not None, f"{cpt_name}: palette es None"
    assert len(palette) == 768, f"{cpt_name}: longitud {len(palette)} != 768"


@pytest.mark.parametrize("cpt_name", CPT_FILES)
def test_palette_values_in_range(cpt_name):
    """Todos los valores de la paleta deben estar en [0, 255]."""
    path = os.path.join(REPO_DIR, cpt_name)
    if not os.path.exists(path):
        pytest.skip(f"{cpt_name} no encontrado en el repo")
    cpt = ColorPaletteTable(path)
    palette = cpt.get_pil_palette()
    if palette is None:
        pytest.skip(f"{cpt_name}: sin paleta generada")
    assert min(palette) >= 0
    assert max(palette) <= 255


# ---------------------------------------------------------------------------
# Propiedades específicas de CPTs conocidos
# ---------------------------------------------------------------------------

class TestSstCpt:
    @pytest.fixture(autouse=True)
    def load(self):
        path = os.path.join(REPO_DIR, 'sst.cpt')
        if not os.path.exists(path):
            pytest.skip("sst.cpt no encontrado")
        self.cpt = ColorPaletteTable(path)

    def test_not_normalized(self):
        assert self.cpt.is_normalized is False

    def test_range_physical(self):
        # SST debe cubrir al menos el rango 268-310 K
        assert self.cpt.min_val <= 270
        assert self.cpt.max_val >= 308


class TestRainbowCpt:
    @pytest.fixture(autouse=True)
    def load(self):
        path = os.path.join(REPO_DIR, 'rainbow.cpt')
        if not os.path.exists(path):
            pytest.skip("rainbow.cpt no encontrado")
        self.cpt = ColorPaletteTable(path)

    def test_is_normalized(self):
        assert self.cpt.is_normalized is True

    def test_range_zero_to_one(self):
        assert self.cpt.min_val == pytest.approx(0.0, abs=0.01)
        assert self.cpt.max_val == pytest.approx(1.0, abs=0.01)


class TestPhaseCpt:
    @pytest.fixture(autouse=True)
    def load(self):
        path = os.path.join(REPO_DIR, 'phase.cpt')
        if not os.path.exists(path):
            pytest.skip("phase.cpt no encontrado")
        self.cpt = ColorPaletteTable(path)

    def test_has_nodata_index(self):
        """phase.cpt tiene entrada N (NoData), debe asignar n_idx."""
        assert self.cpt.n_idx is not None

    def test_discrete_colors(self):
        """phase.cpt es discreto con 6 clases."""
        assert len(self.cpt.colors) >= 5


# ---------------------------------------------------------------------------
# apply_to_data
# ---------------------------------------------------------------------------

class TestApplyToData:
    def test_output_dtype_uint8(self):
        path = os.path.join(REPO_DIR, 'sst.cpt')
        if not os.path.exists(path):
            pytest.skip("sst.cpt no encontrado")
        cpt = ColorPaletteTable(path)
        data = np.linspace(cpt.min_val, cpt.max_val, 100)
        result = cpt.apply_to_data(data)
        assert result.dtype == np.uint8

    def test_output_in_range(self):
        path = os.path.join(REPO_DIR, 'sst.cpt')
        if not os.path.exists(path):
            pytest.skip("sst.cpt no encontrado")
        cpt = ColorPaletteTable(path)
        data = np.linspace(cpt.min_val, cpt.max_val, 100)
        result = cpt.apply_to_data(data)
        assert result.min() >= 0
        assert result.max() <= 255


# ---------------------------------------------------------------------------
# from_tiff_colormap
# ---------------------------------------------------------------------------

COLORMAP_CONTINUO = """\
300.000000,127,0,127
293.333344,127,0,127
293.333344,150,75,20
286.666656,150,75,20
286.666656,255,171,127
280.000000,255,171,127
280.000000,255,127,0
273.333344,255,127,0
"""

COLORMAP_DISCRETO = """\
-0.500000,0,0,0
0.500000,0,0,0
0.500100,2,188,252
1.500000,2,188,252
1.500100,1,251,0
2.500000,1,251,0
"""


class TestFromTiffColormap:
    def test_continuous_palette_length(self):
        cpt = ColorPaletteTable.from_tiff_colormap(COLORMAP_CONTINUO)
        assert cpt.palette is not None
        assert len(cpt.palette) == 768

    def test_continuous_range(self):
        cpt = ColorPaletteTable.from_tiff_colormap(COLORMAP_CONTINUO)
        assert cpt.min_val == pytest.approx(273.333344, abs=0.01)
        assert cpt.max_val == pytest.approx(300.0, abs=0.01)

    def test_continuous_gradient(self):
        """Los extremos deben tener colores diferentes (hay gradiente real)."""
        cpt = ColorPaletteTable.from_tiff_colormap(COLORMAP_CONTINUO)
        color_min = tuple(cpt.palette[0:3])
        color_max = tuple(cpt.palette[253*3:253*3+3])
        assert color_min != color_max

    def test_discrete_palette_length(self):
        cpt = ColorPaletteTable.from_tiff_colormap(COLORMAP_DISCRETO)
        assert cpt.palette is not None
        assert len(cpt.palette) == 768

    def test_discrete_flat_blocks(self):
        """En paleta discreta cada segmento debe ser plano (mismo color en ambos extremos)."""
        cpt = ColorPaletteTable.from_tiff_colormap(COLORMAP_DISCRETO)
        # El segmento 1 corresponde a clase 1 (color 2,188,252)
        # El pixel al inicio y al final del bloque deben ser idénticos
        for seg in cpt.segments:
            v1, r1, g1, b1, v2, r2, g2, b2 = seg
            assert (r1, g1, b1) == (r2, g2, b2), f"Segmento {v1}-{v2} no es plano"

    def test_palette_values_in_range(self):
        for colormap in (COLORMAP_CONTINUO, COLORMAP_DISCRETO):
            cpt = ColorPaletteTable.from_tiff_colormap(colormap)
            assert min(cpt.palette) >= 0
            assert max(cpt.palette) <= 255
