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
