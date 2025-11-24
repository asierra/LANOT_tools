import os
import math

try:
    from PIL import Image
except ImportError:
    Image = None

# Import opcional de aggdraw, shapefile, pyproj, numpy
try:
    import aggdraw
except Exception:
    aggdraw = None

try:
    import shapefile as shp
except Exception:
    shp = None

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except Exception:
    HAS_PYPROJ = False

try:
    import numpy as np
except Exception:
    np = None

class MapDrawer:
    """
    Clase básica para dibujar decoraciones sobre imágenes.
    - lanot_dir: ruta a recursos (logos, shapefiles, cpt).
      Si es None, se usa la variable de entorno LANOT_DATA o '/usr/local/share/lanot'.
    """

    def __init__(self, lanot_dir=None, target_crs=None):
        if lanot_dir is None:
            self.lanot_dir = os.environ.get("LANOT_DATA", "/usr/local/share/lanot")
        else:
            self.lanot_dir = lanot_dir

        if not os.path.exists(self.lanot_dir):
            # No detener; es solo una advertencia para el usuario.
            print(f"Advertencia: directorio de recursos {self.lanot_dir} no existe.")

        self.image = None
        self._shp_cache = {}
        self._layers = {
            'COASTLINE': 'shapefiles/ne_10m_coastline.shp',
            'COUNTRIES': 'shapefiles/ne_10m_admin_0_countries.shp',
        }

        self.use_proj = False
        self.transformer = None
        if target_crs and HAS_PYPROJ:
            self.transformer = Transformer.from_crs("epsg:4326", target_crs, always_xy=True)
            self.use_proj = True

        # Bounds en lon/lat (ULX, ULY, LRX, LRY)
        self.bounds = {'ulx': 0., 'uly': 0., 'lrx': 0., 'lry': 0.}
        self.proj_bounds = {'min_x': 0., 'max_y': 0., 'width': 1., 'height': 1.}

    def set_image(self, pil_image):
        if Image is None:
            raise RuntimeError("Pillow no está instalado.")
        self.image = pil_image

    def set_bounds(self, ulx, uly, lrx, lry):
        self.bounds['ulx'] = ulx
        self.bounds['uly'] = uly
        self.bounds['lrx'] = lrx
        self.bounds['lry'] = lry

    def add_layer(self, key, rel_path):
        self._layers[key.upper()] = rel_path

    def list_layers(self):
        return sorted(self._layers.keys())

    def _geo2pixel_linear(self, lon, lat):
        w = self.image.width
        h = self.image.height
        b = self.bounds
        width_span = b['lrx'] - b['ulx']
        height_span = b['uly'] - b['lry']
        if width_span == 0 or height_span == 0:
            return 0, 0
        u = int(w * (lon - b['ulx']) / width_span)
        v = int(h * (b['uly'] - lat) / height_span)
        return u, v

    def draw_logo(self, logosize=128, position=3):
        if self.image is None:
            return
        if Image is None:
            print("Pillow no disponible.")
            return

        logo_path = os.path.join(self.lanot_dir, 'logos', 'lanot_negro_sn-128.png')
        try:
            logo = Image.open(logo_path).convert("RGBA")
        except Exception:
            print("Logo no encontrado en", logo_path)
            return

        aspect = logo.height / logo.width
        new_h = int(logosize * aspect)
        logo = logo.resize((logosize, new_h), Image.Resampling.LANCZOS)

        pos_x = position & 1
        pos_y = position >> 1
        x = self.image.width - logosize - 10 if pos_x else 10
        y = self.image.height - new_h - 10 if pos_y else 10
        try:
            self.image.paste(logo, (x, y), logo)
        except Exception:
            # En caso de imagen sin alpha, pegar sin máscara
            self.image.paste(logo, (x, y))

    def draw_layer(self, key, color='yellow', width=0.5):
        if self.image is None:
            return
        layer_key = key.upper()
        if layer_key not in self._layers:
            print(f"Capa {key} no registrada. Capas: {self.list_layers()}")
            return
        rel_path = self._layers[layer_key]
        full_path = os.path.join(self.lanot_dir, rel_path)
        if shp is None:
            print("pyshp no instalado; no se pueden dibujar shapefiles.")
            return
        try:
            if full_path not in self._shp_cache:
                self._shp_cache[full_path] = shp.Reader(full_path)
        except Exception as e:
            print("Error leyendo shapefile:", e)
            return

        sf = self._shp_cache[full_path]
        if aggdraw is None:
            print("aggdraw no instalado; no se puede dibujar.")
            return
        draw = aggdraw.Draw(self.image)
        pen = aggdraw.Pen(color, width)
        for shape in sf.shapeRecords():
            points = shape.shape.points
            if not points:
                continue
            coords = []
            for lon, lat in points:
                u, v = self._geo2pixel_linear(lon, lat)
                coords.extend((u, v))
            if len(coords) >= 4:
                draw.line(coords, pen)
        draw.flush()

    def parse_cpt(self, cpt_path):
        items = []
        if not os.path.isabs(cpt_path):
            cpt_path = os.path.join(self.lanot_dir, cpt_path)
        if not os.path.exists(cpt_path):
            print("CPT no encontrado:", cpt_path)
            return items
        try:
            with open(cpt_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(('#', 'B', 'F', 'N')):
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
                            label = " ".join(parts[4:]) if len(parts) > 4 else parts[0]
                            items.append((label, (r, g, b)))
                        except Exception:
                            continue
        except Exception as e:
            print("Error leyendo CPT:", e)
        return items

