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
        self.scale_factor = 1.0
        self.n_idx = None      # Índice para NoData
        self.f_idx = None      # Índice para Foreground
        self.palette_size = 256
        self.units = None
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
        self.units = None
        self.segments = []

        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    if line.startswith("#"):
                        # Check for metadata comments like # UNIT = K
                        if "UNIT" in line:
                            try:
                                self.units = line.split('=')[1].strip()
                            except IndexError:
                                pass # Malformed UNIT line
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
                        self.segments.append((v1, r1, g1, b1, v2, r2, g2, b2))

            # Procesar segmentos continuos si existen
            if self.segments:
                self.min_val = min(s[0] for s in self.segments)
                self.max_val = max(s[4] for s in self.segments)
                
                # Detectar si es normalizada (0-1)
                if self.min_val >= 0 and self.max_val <= 1.0:
                    self.is_normalized = True
                    self.offset = 0
                    self.scale_factor = 255.0
                else:
                    # Caso continuo físico (ej. Kelvin): Escalar al rango completo 0-255
                    self.offset = self.min_val
                    
                    # Determinar espacio reservado para N y F
                    has_n = 'N' in self.special or (use_b_for_n and 'B' in self.special) or force_n
                    has_f = 'F' in self.special
                    
                    limit = 255
                    if has_n: limit -= 1
                    if has_f: limit -= 1
                    
                    # Calcular factor de escala para llenar el espacio disponible
                    if self.max_val > self.min_val:
                        self.scale_factor = float(limit) / (self.max_val - self.min_val)
                    
                # Generar paleta densa (0-255)
                self.palette = [0] * 768
                self.palette_size = 256
                
                # Rellenar gradiente
                for i in range(256):
                    # Calcular valor físico correspondiente a este índice
                    val = self.min_val + (i / self.scale_factor) if self.scale_factor > 0 else self.min_val
                    
                    # Buscar segmento y interpolar
                    r, g, b = 0, 0, 0
                    for v1, r1, g1, b1, v2, r2, g2, b2 in self.segments:
                        if v1 <= val <= v2:
                            span = v2 - v1
                            f = 0.0 if span == 0 else (val - v1) / span
                            f = max(0.0, min(1.0, f))
                            r = int(r1 + f * (r2 - r1))
                            g = int(g1 + f * (g2 - g1))
                            b = int(b1 + f * (b2 - b1))
                            break
                    
                    self.palette[i*3] = r
                    self.palette[i*3+1] = g
                    self.palette[i*3+2] = b
            
            if not self.colors and not self.segments:
                return

            # Lógica para paletas discretas (solo si no hay segmentos continuos)
            if not self.segments:
                self.min_val = min(self.colors.keys())
                self.max_val = max(self.colors.keys())
                self.offset = 0
                
                if self.max_val > 255:
                    self.offset = int(self.min_val)
                    
                # Generar lista plana para PIL (Lógica original discreta)
                self.palette = [0] * 768
                for val, rgb in self.colors.items():
                    idx = int(val - self.offset)
                    if 0 <= idx < 256:
                        self.palette[idx*3] = rgb[0]
                        self.palette[idx*3+1] = rgb[1]
                        self.palette[idx*3+2] = rgb[2]

            # Aplicar colores especiales (N, F, B)
            has_n = 'N' in self.special or (use_b_for_n and 'B' in self.special) or force_n
            has_f = 'F' in self.special
            
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
            
        scaled = np.clip((data - self.offset) * self.scale_factor, 0, upper_limit).astype(np.uint8)
        return scaled

    def draw_legend(self, draw, x, y, width, height, font_size=12, color='white'):
        """Dibuja la barra de colores y las etiquetas."""
        if not self.palette:
            return

        # Dibujar barra de colores
        max_idx = int((self.max_val - self.offset) * self.scale_factor)
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

        mode = draw.im.mode

        # Iterar sobre cada píxel horizontal del ancho de la barra de color
        for i in range(int(width)):
            # Mapear la posición del píxel a un índice de color en la paleta
            # Esto convierte la posición (0 a width-1) a un índice (0 a num_colors-1)
            color_idx = int((i / width) * num_colors)
            # Asegurarse de que el índice no se salga del rango
            color_idx = min(color_idx, num_colors - 1)

            px = x + i

            if mode == 'P' or mode == 'L':
                # Para imágenes con paleta, dibujar con el índice de color
                draw.line([px, y, px, y + height], fill=color_idx)
            else:
                # Para imágenes RGB, obtener el color de la paleta y dibujar
                r, g, b = self.palette[color_idx*3 : color_idx*3+3]
                draw.line([px, y, px, y + height], fill=(r, g, b))

    def _draw_value_row(self, draw, x0, y, width, min_val, max_val, num_intermedios, color, font_size):
        step, min_tmp, lista_valores = self._generar_lista_alineada(min_val, max_val, num_intermedios)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        def get_w(text):
            try:
                l, t, r, b = draw.textbbox((0, 0), text, font=font)
                return r - l
            except AttributeError:
                w, h = draw.textsize(text, font=font)
                return w

        items = []
        for value in lista_valores:
            x = x0 + (value - min_val) * width / (max_val - min_val)
            items.append({'text': str(value), 'x': x})

        if self.units:
            unit_w = get_w(self.units)
            unit_x = x0 + width - unit_w - 2  # Alinear a la derecha
            if items:
                last = items[-1]
                if last['x'] + get_w(last['text']) > unit_x - 5:
                    items.pop()  # Quitar último número si no hay espacio
            items.append({'text': self.units, 'x': unit_x})

        for item in items:
            draw.text((item['x'], y), item['text'], fill=color, font=font)

    def _draw_label_row(self, draw, x0, y, width, min_val, max_val, offset, labels, color, font_size):

        max_idx = int(max_val - offset)
        num_slots = max_idx + 1
        if num_slots <= 0: return
        step = width / num_slots

        # Ajuste dinámico de fuente para evitar superposición
        visible_labels = [str(l) for v, l in labels.items() if 0 <= int(v - offset) <= max_idx]
        if visible_labels:
            max_len = max(len(l) for l in visible_labels)
            if max_len > 0:
                # Estimar tamaño: font_size <= step / (chars * 0.5)
                calc_size = int(step / (max_len * 0.5))
                font_size = max(8, min(font_size, calc_size))

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        
        def get_w(text):
            try:
                l, t, r, b = draw.textbbox((0, 0), text, font=font)
                return r - l
            except AttributeError:
                w, h = draw.textsize(text, font=font)
                return w

        items = []
        for val, label in sorted(labels.items()):
            idx = int(val - offset)
            if 0 <= idx <= max_idx:
                center_x = x0 + idx * step + step / 2
                text = str(label)
                
                # Truncar si excede el ancho disponible
                max_w = step - 2
                if get_w(text) > max_w:
                    while len(text) > 0 and get_w(text + ".") > max_w:
                        text = text[:-1]
                    if text:
                        text += "."
                
                items.append({'text': text, 'x': center_x - get_w(text) / 2})

        if self.units:
            unit_w = get_w(self.units)
            unit_x = x0 + width - unit_w - 2
            if items:
                last = items[-1]
                if last['x'] + get_w(last['text']) > unit_x - 5:
                    items.pop()
            items.append({'text': self.units, 'x': unit_x})

        for item in items:
            draw.text((item['x'], y), item['text'], fill=color, font=font)

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
