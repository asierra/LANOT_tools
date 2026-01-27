from setuptools import setup, find_packages

setup(
    name="lanot-tools",
    version="0.1.0",
    description="Herramientas y utilidades comunes para LANOT (Laboratorio Nacional de Observación de la Tierra)",
    author="Abraham Sierra",
    py_modules=["mapdrawer", "geotiff2view", "colorpalettetable"],
    install_requires=[
        "Pillow",
        "fiona",
        "pyproj",
        "numpy",
        "rasterio",
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'mapdrawer=mapdrawer:main',
            'geotiff2view=geotiff2view:main',
        ],
    },
)
