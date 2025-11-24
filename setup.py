from setuptools import setup, find_packages

setup(
    name="lanot-tools",
    version="0.1.0",
    description="Herramientas y utilidades comunes para LANOT (Laboratorio Nacional de Observación de la Tierra)",
    author="Abraham Sierra",
    packages=find_packages(),
    install_requires=[
        "Pillow",
        "aggdraw",
        "pyshp",
        "pyproj",
        "numpy"
    ],
    python_requires=">=3.8",
)
