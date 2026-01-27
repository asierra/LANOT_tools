import os
import sys
import math
import numpy as np
from PIL import ImageFont, ImageDraw

class ColorPaletteTable:
    """
    Maneja la carga, transformación y visualización de archivos CPT (Color Palette Table).
    Soporta formatos discretos, continuos y normalizados (0-1).
    """
    def __init__(self, path: str = None, use_b_for_n=False, force_n=False):
        self.path = path
        self.colors = {}       # valor -> (r, g, b)
        self.special = {}      # 'B', 'F', 'N' -> (r, g, b)
        self.labels = {}       # valor -> etiqueta
        self.is_normalized = False
        self.min_val = 0
        self.max_val = 0
        self.offset = 0
        self.n_idx = None      # Índice para NoData
        self.f_idx = None      # Índice para Foreground
        self.palette_size = 256
        self.palette = None    # Lista plana para PIL [r,g,b, ...]
        
        if path:
            self.load(path, use_b_for_n, force_n)

    def load(self, path, use_b_for_n=False, force_n=False):        
        """
        Lee un archivo CPT y configura la paleta.
        """
        if not os.path.exists(path):
            print(f"Advertencia: CPT {path} no encontrado.", file=sys.stderr)
            return

        self.colors = {}
        self.special = {}
        self.labels = {}
        self.is_normalized = False

        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    
                    label_text = None
                    if ";" in line:
                        parts = line.split(";", 1)
                        line = parts[0].strip()
                        label_text = parts[1].strip()

                    if "#" in line:
                        line = line.split("#")[0].strip()

                    parts = line.split()
                    if len(parts) < 4:
                        continue

                    if parts[0] in ['B', 'F', 'N']:
                        try:
                            r, g, b = int(float(parts[1])), int(float(parts[2])), int(float(parts[3]))
                            self.special[parts[0]] = (r, g, b)
                        except ValueError:
                            pass
                        continue

                    vals = []
                    for p in parts:
                        try:
                            vals.append(float(p))
                        except ValueError:
                            break
                    
                    n_vals = len(vals)

                    # Caso discreto
                    if n_vals >= 4 and n_vals < 8:
                        val = int(vals[0])
                        self.colors[val] = (int(vals[1]), int(vals[2]), int(vals[3]))
                        if label_text:
                            self.labels[val] = label_text
                        elif len(parts) > 4:
                            self.labels[val] = " ".join(parts[4:])
                    
                    # Caso continuo
                    elif n_vals >= 8:
                        v1, r1, g1, b1 = vals[0], vals[1], vals[2], vals[3]
                        v2, r2, g2, b2 = vals[4], vals[5], vals[6], vals[7]

                        if v1 >= 0 and v2 <= 1.0 and v1 < v2:
                            self.is_normalized = True
                            start_idx = int(round(v1 * 255))
                            end_idx = int(round(v2 * 255))
                            span_val = v2 - v1

                            for i in range(start_idx, end_idx + 1):
                                p = i / 255.0
                                f = 0.0 if span_val <= 0 else (p - v1) / span_val
                                f = max(0.0, min(1.0, f))
                                cr = int(r1 + f * (r2 - r1)); cg = int(g1 + f * (g2 - g1)); cb = int(b1 + f * (b2 - b1))
                                self.colors[i] = (cr, cg, cb)
                        else:
                            start_i = int(np.ceil(v1))
                            end_i = int(np.floor(v2))
                            span = v2 - v1
                            for i in range(start_i, end_i + 1):
                                f = 0.0 if span == 0 else (i - v1) / span
                                f = max(0.0, min(1.0, f))
                                cr = int(r1 + f * (r2 - r1)); cg = int(g1 + f * (g2 - g1)); cb = int(b1 + f * (b2 - b1))
                                self.colors[i] = (cr, cg, cb)
            
            if not self.colors:
                return

            self.min_val = min(self.colors.keys())
            self.max_val = max(self.colors.keys())
            self.offset = 0
            
            if self.max_val > 255:
                self.offset = int(self.min_val)

            has_n = 'N' in self.special or (use_b_for_n and 'B' in self.special) or force_n
            has_f = 'F' in self.special
            
            shifted_max = self.max_val - self.offset
            self.palette_size = 256
            for size in [2, 4, 16, 256]:
                reserved = 0
                if has_n: reserved = 1
                if has_f: reserved = max(reserved, 2)
                if shifted_max < (size - reserved):
                    self.palette_size = size
                    break
            
            # Generar lista plana para PIL
            self.palette = [0] * 768
            for val, rgb in self.colors.items():
                idx = int(val - self.offset)
                if 0 <= idx < 256:
                    self.palette[idx*3] = rgb[0]
                    self.palette[idx*3+1] = rgb[1]
                    self.palette[idx*3+2] = rgb[2]
            
            if has_n:
                self.n_idx = self.palette_size - 1
                rgb = (0, 0, 0)
                if use_b_for_n and 'B' in self.special:
                    rgb = self.special['B']
                elif 'N' in self.special:
                    rgb = self.special['N']
                self.palette[self.n_idx*3 : self.n_idx*3+3] = rgb

            if has_f:
                self.f_idx = self.palette_size - 2
                rgb = self.special['F']
                self.palette[self.f_idx*3 : self.f_idx*3+3] = rgb

        except Exception as e:
            print(f"Error leyendo CPT: {e}", file=sys.stderr)

    def get_pil_palette(self):
        """Devuelve la lista de 768 enteros que requiere PIL."""
        return self.palette

    def apply_to_data(self, data: np.ndarray):
        """Convierte datos brutos a índices de paleta."""
        upper_limit = self.palette_size - 1
        if self.n_idx is not None: 
            if self.n_idx == upper_limit: upper_limit -= 1
        if self.f_idx is not None and self.f_idx <= upper_limit:
            upper_limit = self.f_idx - 1
            
        scaled = np.clip(data - self.offset, 0, upper_limit).astype(np.uint8)
        return scaled

    def draw_legend(self, draw, x, y, width, height, font_size=12, color='white'):
        """Dibuja la barra de colores y las etiquetas."""
        if not self.palette:
            return

        # Dibujar barra de colores
        max_idx = int(self.max_val - self.offset)
        self._draw_colorbar(draw, x, y, width, height, max_index=max_idx)
        
        # Dibujar etiquetas o valores
        text_y = y + height
        text_color = self.f_idx if self.f_idx is not None else 255 # Usar un color visible (blanco/foreground)
        
        if self.labels:
            self._draw_label_row(draw, x, text_y, width, self.min_val, self.max_val, self.offset, self.labels, color, font_size)
        else:
            self._draw_value_row(draw, x, text_y, width, self.min_val, self.max_val, 5, color, font_size)

    def _draw_colorbar(self, draw, x, y, width, height, max_index=None):
        total_colors = 256
        num_colors = min(max_index + 1, total_colors) if max_index is not None else total_colors
        if num_colors <= 0: return

        step = width / num_colors
        mode = draw.im.mode

        for i in range(num_colors):
            x0 = x + i * step
            x1 = x + (i + 1) * step
            
            if mode == 'P' or mode == 'L':
                draw.rectangle([x0, y, x1, y + height], fill=i)
            else:
                r = self.palette[i*3]
                g = self.palette[i*3+1]
                b = self.palette[i*3+2]
                draw.rectangle([x0, y, x1, y + height], fill=(r, g, b))

    def _draw_value_row(self, draw, x0, y, width, min_val, max_val, num_intermedios, color, font_size):
        step, min_tmp, lista_valores = self._generar_lista_alineada(min_val, max_val, num_intermedios)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        for value in lista_valores:
            x = x0 + (value - min_val) * width / (max_val - min_val)
            draw.text((x, y), str(value), fill=color, font=font)

    def _draw_label_row(self, draw, x0, y, width, min_val, max_val, offset, labels, color, font_size):
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        max_idx = int(max_val - offset)
        num_slots = max_idx + 1
        if num_slots <= 0: return
        step = width / num_slots
        
        for val, label in labels.items():
            idx = int(val - offset)
            if 0 <= idx <= max_idx:
                center_x = x0 + idx * step + step / 2
                try:
                    l, t, r, b = draw.textbbox((0, 0), str(label), font=font)
                    w_text = r - l
                except AttributeError:
                    w_text, h_text = draw.textsize(str(label), font=font)
                draw.text((center_x - w_text / 2, y), str(label), fill=color, font=font)

    def _obtener_paso_redondo(self, raw_step):
        if raw_step <= 0: return 0
        exponente = math.floor(math.log10(raw_step))
        fraccion = raw_step / (10 ** exponente)
        if fraccion < 1.5: nice = 1
        elif fraccion < 3: nice = 2
        elif fraccion < 7: nice = 5
        else: nice = 10
        return nice * (10 ** exponente)

    def _generar_lista_alineada(self, val_min, val_max, num_intermedios):
        if val_min >= val_max: return 1, val_min, []
        rango_teorico = val_max - val_min
        raw_step = rango_teorico / (num_intermedios + 1)
        nice_step = self._obtener_paso_redondo(raw_step)
        min_tmp = math.ceil(val_min / nice_step) * nice_step
        precision = max(0, -math.floor(math.log10(nice_step))) if nice_step < 1 else 0
        lista_valores = []
        valor_actual = min_tmp
        while len(lista_valores) <= num_intermedios:
            if valor_actual >= val_max: break
            lista_valores.append(round(valor_actual, precision))
            valor_actual += nice_step
        return nice_step, min_tmp, lista_valores
