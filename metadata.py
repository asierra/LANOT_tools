#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metadata - Lightweight metadata container for GeoTIFF and satellite imagery.

Provides a dict-based container with helper methods for common operations like
bounds format conversion and JSON serialization.

Autor: Alejandro Aguilar Sierra
LANOT - Laboratorio Nacional de Observación de la Tierra
"""

import json
import os
import re
from datetime import datetime

class Metadata:
    """
    Lightweight wrapper around dict for GeoTIFF metadata with conversion helpers.
    
    Stores metadata as a dictionary with helper methods for common operations:
    - Bounds format conversion (rasterio ↔ MapDrawer)
    - JSON serialization/deserialization
    - Factory methods for common sources (rasterio, JSON files)
    
    Usage:
        # Create and populate
        meta = Metadata(crs='goes16', bounds=(-120, 20, -80, 50))
        meta['timestamp'] = '2026-01-30T12:00:00Z'
        
        # Dict-like access
        if 'crs' in meta:
            print(meta['crs'])
        
        # Helper methods
        ulx, uly, lrx, lry = meta.get_mapdrawer_bounds()
        meta.save_json('output.json')
    """
    
    def __init__(self, **kwargs):
        """Initialize with optional keyword arguments as metadata fields."""
        self._data = kwargs
    
    def __getitem__(self, key):
        """Dict-like access: metadata['key']"""
        return self._data[key]
    
    def __setitem__(self, key, value):
        """Dict-like assignment: metadata['key'] = value"""
        self._data[key] = value
    
    def get(self, key, default=None):
        """Dict-like get with default: metadata.get('key', default)"""
        return self._data.get(key, default)
    
    def __contains__(self, key):
        """Support 'in' operator: 'key' in metadata"""
        return key in self._data
    
    def keys(self):
        """Return dictionary keys."""
        return self._data.keys()
    
    def items(self):
        """Return dictionary items."""
        return self._data.items()
    
    def __repr__(self):
        """String representation for debugging."""
        return f"Metadata({self._data})"
    
    def get_mapdrawer_bounds(self):
        """
        Convert bounds to MapDrawer format.
        
        Transforms rasterio-style bounds (left, bottom, right, top) to
        MapDrawer format (ulx, uly, lrx, lry) which is (minx, maxy, maxx, miny).
        
        Returns:
            tuple or None: (ulx, uly, lrx, lry) or None if bounds not available
        """
        if 'bounds' not in self._data:
            return None
        b = self._data['bounds']
        # rasterio: (left, bottom, right, top)
        # MapDrawer: (ulx, uly, lrx, lry) = (minx, maxy, maxx, miny)
        return (b[0], b[3], b[2], b[1])
    
    def to_dict(self):
        """
        Export to plain dict for JSON serialization.
        
        Returns:
            dict: Copy of internal data dictionary
        """
        return self._data.copy()
    
    @classmethod
    def from_dict(cls, data):
        """
        Create Metadata instance from dictionary.
        
        Args:
            data (dict): Dictionary with metadata fields
            
        Returns:
            Metadata: New instance with data
        """
        return cls(**data)
    
    @classmethod
    def from_rasterio(cls, src):
        """
        Extract metadata from rasterio dataset.
        
        Extracts CRS, bounds, and common TIFF tags for timestamp and satellite info.
        
        Args:
            src: Rasterio dataset handle
            
        Returns:
            Metadata: Instance populated with rasterio metadata
        """
        meta = cls()
        
        # Extract CRS
        if src.crs:
            meta['crs'] = src.crs.to_string()
        
        # Extract bounds (left, bottom, right, top)
        meta['bounds'] = src.bounds
        
        # Extract tags for timestamp
        tags = src.tags()
        for key in ['TIFFTAG_DATETIME', 'DATETIME', 'date_created', 'time_coverage_start']:
            if key in tags:
                meta['timestamp'] = tags[key]
                break
        
        # Extract satellite/platform info
        for key in ['platform', 'satellite', 'spacecraft', 'mission', 'TIFFTAG_IMAGEDESCRIPTION']:
            if key in tags:
                meta['satellite'] = tags[key]
                break

        # Extract band/channel info
        for key in ['band', 'band_id', 'channel', 'sensor_band']:
            if key in tags:
                meta['band'] = tags[key]
                break
        
        # Extract product info if not a band
        if 'band' not in meta:
            for key in ['product', 'product_name', 'algorithm']:
                if key in tags:
                    meta['product'] = tags[key]
                    break
        
        return meta
    
    @classmethod
    def from_json_file(cls, filepath):
        """
        Load metadata from JSON sidecar file.
        
        Args:
            filepath (str): Path to JSON file
            
        Returns:
            Metadata: Instance loaded from JSON
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Metadata file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_json(self, filepath):
        """
        Save metadata to JSON file.
        
        Args:
            filepath (str): Output path for JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def enrich_from_filename(self, filename):
        """
        Fill missing metadata fields by parsing the file name.

        Only sets fields that are not already present (fallback logic).
        Detected fields: timestamp, satellite, sensor, band, product.

        Satellite tokens (prefix or substring):
            npp            → Suomi NPP
            noaa20 / j01   → NOAA-20
            noaa21 / j02   → NOAA-21
            noaa22 / j03   → NOAA-22
            metopc         → Metop-C
            metopb         → Metop-B
            metopa         → Metop-A

        Sensor tokens (substring):
            viirs → VIIRS
            avhrr → AVHRR
            modis → MODIS
            abi   → ABI
            mhs   → MHS
            amsu  → AMSU-A

        Timestamp patterns (in order of priority):
            YYYYMMDD_HHMMSS  (e.g. 20260325_195149)
            YYYYjjjHHMM      (Julian day, e.g. 2026084195149)

        Band patterns:
            M01–M16, I01–I05, C01–C16, DNB, band01 …

        Product keywords (substring, case-insensitive):
            sst, true_color, cloud_phase, cloud_type,
            cld_temp_acha, cld_height_acha, cld_emiss_acha,
            dnb, fire

        Args:
            filename (str): Path or basename of the file.

        Returns:
            self: For chaining.
        """
        basename = os.path.basename(filename)
        lower = basename.lower()

        # --- timestamp ---
        if 'timestamp' not in self:
            # Pattern 1: YYYYMMDD_HHMMSS
            m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", basename)
            if m:
                try:
                    dt = datetime.strptime("".join(m.groups()), "%Y%m%d%H%M%S")
                    self['timestamp'] = dt.strftime("%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass

            # Pattern 2: YYYYjjjHHMM (Julian day)
            if 'timestamp' not in self:
                m = re.search(r"(\d{4})(\d{3})(\d{4})", basename)
                if m:
                    yyyy, jjj, hhmm = m.groups()
                    try:
                        dt = datetime.strptime(f"{yyyy}{jjj}{hhmm}", "%Y%j%H%M")
                        self['timestamp'] = dt.strftime("%Y:%m:%d %H:%M:%S")
                    except ValueError:
                        pass

        # --- satellite ---
        if 'satellite' not in self:
            if lower.startswith('npp') or '_npp_' in lower:
                self['satellite'] = 'Suomi NPP'
            elif 'noaa20' in lower or lower.startswith('j01'):
                self['satellite'] = 'NOAA-20'
            elif 'noaa21' in lower or lower.startswith('j02'):
                self['satellite'] = 'NOAA-21'
            elif 'noaa22' in lower or lower.startswith('j03'):
                self['satellite'] = 'NOAA-22'
            elif 'metopc' in lower:
                self['satellite'] = 'Metop-C'
            elif 'metopb' in lower:
                self['satellite'] = 'Metop-B'
            elif 'metopa' in lower:
                self['satellite'] = 'Metop-A'

        # --- sensor ---
        if 'sensor' not in self:
            if 'viirs' in lower:
                self['sensor'] = 'VIIRS'
            elif 'avhrr' in lower:
                self['sensor'] = 'AVHRR'
            elif 'modis' in lower:
                self['sensor'] = 'MODIS'
            elif 'abi' in lower:
                self['sensor'] = 'ABI'
            elif 'mhs' in lower:
                self['sensor'] = 'MHS'
            elif 'amsu' in lower:
                self['sensor'] = 'AMSU-A'

        # --- band ---
        if 'band' not in self:
            m = re.search(r"(?:^|[_.])((?:M|I|C)\d{1,2}|DNB|band\d+)(?:[_.]|$)",
                          basename, re.IGNORECASE)
            if m:
                self['band'] = m.group(1).upper()

        # --- product (only if no band detected) ---
        if not self.get('band') and 'product' not in self:
            product_map = [
                ('true_color',       'True Color'),
                ('sst',              'SST'),
                ('cloud_phase',      'Cloud Phase'),
                ('cloud_type',       'Cloud Type'),
                ('cld_temp_acha',    'Cloud Top Temp'),
                ('cld_height_acha',  'Cloud Top Height'),
                ('cld_emiss_acha',   'Cloud Emissivity'),
                ('fire',             'Fire'),
                ('dnb',              'DNB'),
            ]
            for token, label in product_map:
                if token in lower:
                    self['product'] = label
                    break

        return self

    def format_timestamp(self, fmt="%Y/%m/%d %H:%MZ", include_satellite=True,
                         include_sensor=False):
        """
        Return a display string combining satellite/sensor info and the timestamp.

        Normalizes the stored timestamp from TIFF format (YYYY:MM:DD HH:MM:SS)
        or ISO format to the requested display format.

        Args:
            fmt (str): strftime format for the date portion.
            include_satellite (bool): Prepend satellite name if available.
            include_sensor (bool): Prepend sensor name if available.

        Returns:
            str or None: Formatted string, or None if no timestamp present.
        """
        if 'timestamp' not in self:
            return None

        ts = self['timestamp']
        # Normalize TIFF standard format
        for parse_fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
                          "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%MZ"):
            try:
                dt = datetime.strptime(ts, parse_fmt)
                ts = dt.strftime(fmt)
                break
            except ValueError:
                continue

        parts = []
        if include_satellite and 'satellite' in self:
            parts.append(self['satellite'])
        if include_sensor and 'sensor' in self:
            parts.append(self['sensor'])
        parts.append(ts)
        return " ".join(parts)
