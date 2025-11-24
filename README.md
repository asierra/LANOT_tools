# LANOT_tools

Paquete de utilidades para LANOT. Contiene `mapdrawer` como primer módulo.

Instalación:
```bash
pip install .
# o
pip install git+https://github.com/asierra/LANOT_tools.git
```

Uso básico:
```python
from PIL import Image
from lanot import MapDrawer

img = Image.open("imagen.png").convert("RGB")
m = MapDrawer()  # usa LANOT_DATA o /usr/local/share/lanot
m.set_image(img)
m.draw_logo()
img.save("out.png")
```

