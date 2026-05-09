"""
Tests para metadata.py

Cubre:
- Acceso dict-like
- get_mapdrawer_bounds()
- from_dict / from_json_file / save_json
- enrich_from_filename()
- format_timestamp()
- format_timestamp_glm()   ← nueva funcionalidad GLM
"""

import json
import os
import sys
import tempfile

import pytest

# Asegurar que el directorio raíz del proyecto esté en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metadata import Metadata


# ---------------------------------------------------------------------------
# Acceso dict-like
# ---------------------------------------------------------------------------

class TestDictInterface:
    def test_set_and_get_item(self):
        m = Metadata()
        m['crs'] = 'goes18'
        assert m['crs'] == 'goes18'

    def test_contains(self):
        m = Metadata(satellite='GOES-18')
        assert 'satellite' in m
        assert 'bounds' not in m

    def test_get_with_default(self):
        m = Metadata()
        assert m.get('missing', 42) == 42
        assert m.get('missing') is None

    def test_keys_and_items(self):
        m = Metadata(a=1, b=2)
        assert set(m.keys()) == {'a', 'b'}
        assert dict(m.items()) == {'a': 1, 'b': 2}

    def test_pop(self):
        m = Metadata(x=10)
        val = m.pop('x')
        assert val == 10
        assert 'x' not in m

    def test_pop_missing_with_default(self):
        m = Metadata()
        assert m.pop('nope', 0) == 0

    def test_repr(self):
        m = Metadata(crs='goes16')
        assert 'goes16' in repr(m)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

class TestBounds:
    def test_get_mapdrawer_bounds_from_rasterio_tuple(self):
        # rasterio: (left, bottom, right, top)
        m = Metadata(bounds=(-130.0, 20.0, -60.0, 50.0))
        ulx, uly, lrx, lry = m.get_mapdrawer_bounds()
        assert ulx == -130.0  # minx
        assert uly == 50.0    # maxy
        assert lrx == -60.0   # maxx
        assert lry == 20.0    # miny

    def test_get_mapdrawer_bounds_none_when_missing(self):
        m = Metadata()
        assert m.get_mapdrawer_bounds() is None


# ---------------------------------------------------------------------------
# Serialización
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_returns_copy(self):
        m = Metadata(crs='goes18', satellite='GOES-18')
        d = m.to_dict()
        assert d == {'crs': 'goes18', 'satellite': 'GOES-18'}
        d['extra'] = 'val'
        assert 'extra' not in m  # no afecta el original

    def test_from_dict(self):
        m = Metadata.from_dict({'crs': 'goes16', 'satellite': 'GOES-16'})
        assert m['crs'] == 'goes16'
        assert m['satellite'] == 'GOES-16'

    def test_save_and_load_json(self, tmp_path):
        m = Metadata(crs='goes18', bounds=[-130, 20, -60, 50])
        path = str(tmp_path / 'meta.json')
        m.save_json(path)
        m2 = Metadata.from_json_file(path)
        assert m2['crs'] == 'goes18'
        assert m2['bounds'] == [-130, 20, -60, 50]

    def test_from_json_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Metadata.from_json_file('/no/existe/meta.json')

    def test_from_json_file_real(self):
        """Carga el sidecar JSON que ya existe en el repo."""
        repo_json = os.path.join(os.path.dirname(__file__), '..', '20260410_1927_G18_m1.json')
        if not os.path.exists(repo_json):
            pytest.skip("Archivo JSON de ejemplo no encontrado en el repo")
        m = Metadata.from_json_file(repo_json)
        # Debe tener al menos alguna clave
        assert len(list(m.keys())) > 0


# ---------------------------------------------------------------------------
# enrich_from_filename
# ---------------------------------------------------------------------------

class TestEnrichFromFilename:
    def test_goes_abi_filename(self):
        """Detecta sensor ABI desde nombre de archivo GOES estándar."""
        m = Metadata()
        m.enrich_from_filename('OR_ABI-L1b-RadC-M6C13_G18_s20261001800000_e20261001809000_c20261001809088.nc')
        # El sensor ABI siempre se detecta
        assert m.get('sensor') == 'ABI'

    def test_abi_channel_from_simple_filename(self):
        """Detecta banda C13 cuando aparece delimitada por _ en el nombre."""
        m = Metadata()
        m.enrich_from_filename('goes18_abi_C13_20260428_192700.tif')
        assert m.get('band') == 'C13'
        assert m.get('sensor') == 'ABI'

    def test_viirs_filename_timestamp(self):
        m = Metadata()
        m.enrich_from_filename('npp_viirs_cld_temp_acha_20260127_074806_wgs84_geo_750m.tif')
        assert m.get('timestamp') == '2026:01:27 07:48:06'
        assert m.get('satellite') == 'Suomi NPP'
        assert m.get('sensor') == 'VIIRS'

    def test_viirs_product_sst(self):
        m = Metadata()
        m.enrich_from_filename('npp_viirs_sst_20260127_185932_wgs84_geo_750m.tif')
        assert m.get('product') == 'SST'

    def test_viirs_product_cloud_type(self):
        m = Metadata()
        m.enrich_from_filename('npp_viirs_cloud_type_20260203_200738_wgs84_geo_750m.tif')
        assert m.get('product') == 'Cloud Type'

    def test_noaa20_satellite(self):
        m = Metadata()
        m.enrich_from_filename('noaa20_viirs_sst_20260101_120000.tif')
        assert m.get('satellite') == 'NOAA-20'

    def test_does_not_overwrite_existing(self):
        m = Metadata(satellite='Custom')
        m.enrich_from_filename('npp_viirs_sst_20260127_185932.tif')
        assert m['satellite'] == 'Custom'  # no sobreescrito

    def test_cldtoptemp_product_and_units(self):
        m = Metadata()
        m.enrich_from_filename('noaa20_viirs_CldTopTemp_20260508_192713_wgs84_fit.tif')
        assert m.get('product') == 'Cloud Top Temp'
        assert m.get('units') == 'K'

    def test_cldtophght_product_and_units(self):
        m = Metadata()
        m.enrich_from_filename('noaa20_viirs_CldTopHght_20260508_192713_wgs84_fit.tif')
        assert m.get('product') == 'Cloud Top Height'
        assert m.get('units') == 'm'

    def test_cloudphase_product_no_units(self):
        m = Metadata()
        m.enrich_from_filename('noaa20_viirs_CloudPhase_20260508_192713_wgs84_fit.tif')
        assert m.get('product') == 'Cloud Phase'
        assert m.get('units') is None

    def test_cld_temp_acha_product_and_units(self):
        m = Metadata()
        m.enrich_from_filename('npp_viirs_cld_temp_acha_20260127_074806_wgs84_geo_750m.tif')
        assert m.get('product') == 'Cloud Top Temp'
        assert m.get('units') == 'K'


# format_timestamp
# ---------------------------------------------------------------------------

class TestFormatTimestamp:
    def test_tiff_format(self):
        m = Metadata(timestamp='2026:04:10 19:27:00')
        result = m.format_timestamp()
        assert '2026/04/10' in result
        assert '19:27' in result

    def test_iso_format(self):
        m = Metadata(timestamp='2026-04-10T19:27:00Z')
        result = m.format_timestamp()
        assert '2026/04/10' in result

    def test_with_satellite(self):
        m = Metadata(satellite='GOES-18', timestamp='2026:04:10 19:27:00')
        result = m.format_timestamp(include_satellite=True)
        assert 'GOES-18' in result

    def test_without_satellite(self):
        m = Metadata(satellite='GOES-18', timestamp='2026:04:10 19:27:00')
        result = m.format_timestamp(include_satellite=False)
        assert 'GOES-18' not in result

    def test_with_band(self):
        m = Metadata(satellite='GOES-18', timestamp='2026:04:10 19:27:00', band='C13')
        result = m.format_timestamp(include_product=True)
        assert 'C13' in result

    def test_no_timestamp_returns_satellite_only(self):
        m = Metadata(satellite='GOES-18')
        result = m.format_timestamp(include_satellite=True)
        assert result == 'GOES-18'

    def test_empty_returns_none(self):
        m = Metadata()
        assert m.format_timestamp() is None


# ---------------------------------------------------------------------------
# format_timestamp_glm  (nueva funcionalidad)
# ---------------------------------------------------------------------------

class TestFormatTimestampGlm:
    def _meta_with_glm(self):
        return Metadata(
            satellite='GOES-18',
            band='C13',
            timestamp='2026:04:28 19:15:00',
            glm_time_start='2026:04:28 19:15:00',
            glm_time_end='2026:04:28 19:30:00',
        )

    def test_contains_satellite(self):
        m = self._meta_with_glm()
        result = m.format_timestamp_glm()
        assert 'GOES-18' in result

    def test_contains_abi_and_glm_label(self):
        m = self._meta_with_glm()
        result = m.format_timestamp_glm()
        assert 'ABI' in result
        assert 'GLM' in result

    def test_contains_band(self):
        m = self._meta_with_glm()
        result = m.format_timestamp_glm()
        assert 'C13' in result

    def test_contains_abi_date(self):
        m = self._meta_with_glm()
        result = m.format_timestamp_glm()
        assert '2026/04/28' in result

    def test_contains_glm_time_range(self):
        m = self._meta_with_glm()
        result = m.format_timestamp_glm()
        assert '19:15' in result
        assert '19:30' in result

    def test_same_start_end_no_dash(self):
        m = Metadata(
            satellite='GOES-18',
            timestamp='2026:04:28 19:15:00',
            glm_time_start='2026:04:28 19:15:00',
            glm_time_end='2026:04:28 19:15:00',
        )
        result = m.format_timestamp_glm()
        assert '–' not in result

    def test_fallback_to_format_timestamp_without_glm(self):
        m = Metadata(satellite='GOES-18', timestamp='2026:04:28 19:15:00')
        result = m.format_timestamp_glm()
        # Sin datos GLM debe comportarse igual que format_timestamp
        expected = m.format_timestamp(include_satellite=True, include_product=True)
        assert result == expected
