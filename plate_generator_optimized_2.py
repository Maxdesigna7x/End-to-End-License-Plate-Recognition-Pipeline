import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import cv2
import glob
from tqdm import tqdm
import math
import json
from multiprocessing import Pool, cpu_count
from functools import lru_cache
import io

# --- Constantes Globales ---
ANCHO_PLACA = 400
ALTO_PLACA = 200
UPSCALE_FACTOR = 2
ANCHO_TRABAJO = ANCHO_PLACA * UPSCALE_FACTOR
ALTO_TRABAJO = ALTO_PLACA * UPSCALE_FACTOR

DIR_FUENTES = r"C:\Universidad\Seminario\Roberto_Plates\V6\fonts"
DIR_PLANTILLAS = r"C:\Universidad\Seminario\Roberto_Plates\V6\plantillas_generadas"
DIR_DATASET = r"C:\Universidad\Seminario\Roberto_Plates\V6\datasets\plate_test"

os.makedirs(DIR_FUENTES, exist_ok=True)
os.makedirs(DIR_PLANTILLAS, exist_ok=True)
os.makedirs(DIR_DATASET, exist_ok=True)

# --- OPTIMIZACIÓN 1: Cargar fuentes y plantillas UNA VEZ ---
FUENTES_DISPONIBLES = glob.glob(os.path.join(DIR_FUENTES, "**/*.ttf"), recursive=True)
FUENTES_DISPONIBLES.extend(glob.glob(os.path.join(DIR_FUENTES, "**/*.otf"), recursive=True))

if not FUENTES_DISPONIBLES:
    print(f"ADVERTENCIA: No se encontró ninguna fuente en '{DIR_FUENTES}'.")
else:
    print(f"Cargadas {len(FUENTES_DISPONIBLES)} fuentes desde '{DIR_FUENTES}'.")

# Cargar plantillas en memoria (evitar I/O repetido)
PLANTILLAS_CACHE = {}
def cargar_plantillas_en_memoria():
    """Carga todas las plantillas en memoria al inicio"""
    global PLANTILLAS_CACHE
    lista_plantillas = glob.glob(os.path.join(DIR_PLANTILLAS, "*.jpg"))
    lista_plantillas.extend(glob.glob(os.path.join(DIR_PLANTILLAS, "*.png")))
    
    print(f"Cargando {len(lista_plantillas)} plantillas en memoria...")
    for ruta in tqdm(lista_plantillas, desc="Cargando plantillas"):
        try:
            img = Image.open(ruta)
            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')
            PLANTILLAS_CACHE[ruta] = img.copy()
            img.close()
        except Exception as e:
            print(f"Error cargando {ruta}: {e}")
    
    return list(PLANTILLAS_CACHE.keys())

# --- OPTIMIZACIÓN 2: Cache de fuentes ---
FONT_CACHE = {}
def get_cached_font(font_path, size):
    """Cachea fuentes ya cargadas para evitar I/O repetido"""
    key = (font_path, size)
    if key not in FONT_CACHE:
        try:
            FONT_CACHE[key] = ImageFont.truetype(font_path, size)
        except Exception as e:
            print(f"Error cargando fuente {font_path}: {e}")
            return None
    return FONT_CACHE[key]

# --- Funciones de utilidad (sin cambios mayores) ---
def get_random_color(tipo='fuerte'):
    if tipo == 'tenue':
        r, g, b = (random.randint(200, 240), random.randint(200, 240), random.randint(200, 240))
        if r > 230 and g > 230 and b > 230: r -= 30
        return (r, g, b)
    else:
        r, g, b = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        if r < 50 and g < 50 and b < 50: g += 100
        return (r, g, b)

def es_oscuro(color_rgb):
    r, g, b = color_rgb
    luminancia = (0.2126 * r + 0.7152 * g + 0.0722 * b)
    return luminancia < 128

def dibujar_bordes_variables(draw, color_borde):
    ancho_fino_lateral = max(2, int(ANCHO_PLACA * 0.01))
    draw.rectangle((0, 0, ancho_fino_lateral, ALTO_PLACA), fill=color_borde)
    draw.rectangle((ANCHO_PLACA - ancho_fino_lateral, 0, ANCHO_PLACA, ALTO_PLACA), fill=color_borde)

    alto_fino = max(2, int(ALTO_PLACA * 0.015))
    alto_grueso_max = int(ALTO_PLACA * 0.20)
    alto_grueso_min = int(ALTO_PLACA * 0.15)
    
    grosor_superior = random.randint(alto_grueso_min, alto_grueso_max) if random.random() < 0.5 else alto_fino
    grosor_inferior = random.randint(alto_grueso_min, alto_grueso_max) if random.random() < 0.5 else alto_fino

    draw.rectangle((0, 0, ANCHO_PLACA, grosor_superior), fill=color_borde)
    draw.rectangle((0, ALTO_PLACA - grosor_inferior, ANCHO_PLACA, ALTO_PLACA), fill=color_borde)

def agregar_fondo_central_tenue(draw):
    color = get_random_color('tenue')
    top_margin = random.randint(int(ALTO_PLACA * 0.20), int(ALTO_PLACA * 0.30))
    bottom_margin = random.randint(int(ALTO_PLACA * 0.20), int(ALTO_PLACA * 0.30))
    y0, y1 = top_margin, ALTO_PLACA - bottom_margin
    if y1 <= y0: y1 = y0 + 1
    draw.rectangle((0, y0, ANCHO_PLACA, y1), fill=color)
    return color

def generar_texto_placa():
    letras = string.ascii_uppercase
    numeros = string.digits
    
    formato = random.choice(['LLL-NN-NN', 'A-NNN-LLL', 'LLL-NNN', 'LLLLLLL', 'NNNNNNN'])
    
    if formato == 'LLL-NN-NN':
        placa = f"{''.join(random.choices(letras, k=3))}-{''.join(random.choices(numeros, k=2))}-{''.join(random.choices(numeros, k=2))}"
    elif formato == 'A-NNN-LLL':
        placa = f"{random.choice(letras)}-{''.join(random.choices(numeros, k=3))}-{''.join(random.choices(letras, k=3))}"
    elif formato == 'LLL-NNN':
        placa = f"{''.join(random.choices(letras, k=3))}-{''.join(random.choices(numeros, k=3))}"
    elif formato == 'LLLLLLL':
        placa = ''.join(random.choices(letras, k=7))
    elif formato == 'NNNNNNN':
        placa = ''.join(random.choices(numeros, k=7))

    decision = random.random()
    if formato in ['LLLLLLL', 'NNNNNNN']:
        if decision < 0.3:
            posicion = random.randint(2, 5)
            placa = f"{placa[:posicion]} {placa[posicion:]}"
    else:
        if decision < 0.2:
            placa = placa.replace("-", "")
        elif decision < 0.4:
            placa = placa.replace("-", " ")
    
    return placa

def get_pixel_width(fuente, texto):
    try:
        bbox = fuente.getbbox(texto)
        return bbox[2] - bbox[0]
    except Exception:
        return fuente.getlength(texto)

def centrar_texto(draw, texto, x_centro, y_centro, fuente, color=(0,0,0)):
    try:
        draw.text((x_centro, y_centro), texto, font=fuente, fill=color, anchor="mm")
    except TypeError:
        bbox = draw.textbbox((0, 0), texto, font=fuente)
        ancho_texto, alto_texto = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pos_x, pos_y = x_centro - (ancho_texto / 2), y_centro - (alto_texto / 2)
        draw.text((pos_x, pos_y), texto, font=fuente, fill=color)

# --- NUEVO: probabilidad de dibujar texto a color en la imagen original (la máscara NO cambia) ---
COLORED_TEXT_PROB = 0.5  # 0.5 = la mitad de las imágenes tendrán texto a color

# --- OPTIMIZACIÓN 3: Estampado optimizado ---
def estampar_placa(ruta_plantilla, lista_fuentes, texto_placa):
    """Versión optimizada con cache de fuentes y límite de iteraciones.
       Ahora devuelve: (img_pil_rgb, mask_pil_L, lista_bboxes_limpios)"""
    
    # Usar plantilla desde cache
    if ruta_plantilla not in PLANTILLAS_CACHE:
        return None, None, None
    
    img = PLANTILLAS_CACHE[ruta_plantilla].copy()
    img = img.resize((ANCHO_TRABAJO, ALTO_TRABAJO), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(img)
    ANCHO_IMG_LOCAL, ALTO_IMG_LOCAL = img.size
    
    size_placa = int(ALTO_IMG_LOCAL * 0.44)
    max_ancho_permitido = ANCHO_IMG_LOCAL * 0.85
    
    # OPTIMIZACIÓN: Límite de intentos para evitar bucles infinitos
    max_intentos = 10
    for intento in range(max_intentos):
        try:
            ruta_font_principal = random.choice(lista_fuentes)
            size_guion = int(size_placa * (0.31 / 0.44))
            
            # Usar cache de fuentes
            fnt_placa = get_cached_font(ruta_font_principal, size_placa)
            if fnt_placa is None:
                continue
            
            try:
                ruta_guion = os.path.join(DIR_FUENTES, "arial.ttf")
                fnt_guion_generico = get_cached_font(ruta_guion, size_guion)
                if fnt_guion_generico is None:
                    fnt_guion_generico = fnt_placa
            except:
                fnt_guion_generico = fnt_placa
        except Exception as e:
            if intento == max_intentos - 1:
                return None, None, None
            continue
        
        # Calcular ancho total
        ancho_total_calculado = 0
        espacio_char = int(ANCHO_IMG_LOCAL * 0.005)
        espacio_guion = int(ANCHO_IMG_LOCAL * 0.015)

        for char in texto_placa:
            if char == '-':
                ancho_total_calculado += get_pixel_width(fnt_guion_generico, char) + espacio_guion
            else:
                ancho_total_calculado += get_pixel_width(fnt_placa, char) + espacio_char
        
        if ancho_total_calculado <= max_ancho_permitido:
            break
        else:
            ratio = max_ancho_permitido / ancho_total_calculado
            size_placa = int(size_placa * ratio * 0.98)
            if size_placa < 10:
                if intento == max_intentos - 1:
                    return None, None, None
                continue

    # --- NUEVO: decidir si esta imagen tendrá texto a color ---
    use_colored_text = (random.random() < COLORED_TEXT_PROB)

    # Dibujar texto
    # color_texto por defecto (negro/blanco) — será sobrescrito por color aleatorio si use_colored_text == True
    color_texto_default = (255, 255, 255) if "_oscura" in ruta_plantilla else (0, 0, 0)
    if use_colored_text:
        # color random fuerte para el texto (solo en la imagen original)
        color_texto = get_random_color('fuerte')
    else:
        color_texto = color_texto_default

    POS_Y_CENTRO = ALTO_IMG_LOCAL / 2
    pos_x_actual = (ANCHO_IMG_LOCAL / 2) - (ancho_total_calculado / 2)
    
    lista_bboxes_limpios = []
    
    bbox_letra_control = fnt_placa.getbbox("X")
    control_y_centro_local = (bbox_letra_control[3] + bbox_letra_control[1]) / 2
    draw_y_letra_equivalente = POS_Y_CENTRO - control_y_centro_local
    CONTROL_Y1 = draw_y_letra_equivalente + bbox_letra_control[1]
    CONTROL_Y2 = draw_y_letra_equivalente + bbox_letra_control[3]

    # Crear máscara limpia (blanco fondo, negro texto) - la máscara NO cambia, siempre texto negro
    mask_img = Image.new('L', (ANCHO_IMG_LOCAL, ALTO_IMG_LOCAL), 255)
    mask_draw = ImageDraw.Draw(mask_img)

    for char in texto_placa:
        if char == '-':
            font_actual = fnt_guion_generico
            espacio = espacio_guion
        else:
            font_actual = fnt_placa
            espacio = espacio_char

        bbox_glyph = font_actual.getbbox(char)
        char_ancho_pixels = bbox_glyph[2] - bbox_glyph[0]
        
        char_centro_y_local = (bbox_glyph[3] + bbox_glyph[1]) / 2
        draw_y = POS_Y_CENTRO - char_centro_y_local
        draw_x = pos_x_actual - bbox_glyph[0]
        
        # Dibujar en la imagen final: si use_colored_text es True, usamos color aleatorio; si no, el color por defecto.
        draw.text((draw_x, draw_y), char, font=font_actual, fill=color_texto)
        # Dibujar en la máscara (negro = texto) — siempre negro
        mask_draw.text((draw_x, draw_y), char, font=font_actual, fill=0)
        
        if char == '-':
            abs_x1 = draw_x + bbox_glyph[0]
            abs_x2 = draw_x + bbox_glyph[2]
            abs_y1 = CONTROL_Y1
            abs_y2 = CONTROL_Y2
        else:
            abs_x1 = draw_x + bbox_glyph[0]
            abs_y1 = draw_y + bbox_glyph[1]
            abs_x2 = draw_x + bbox_glyph[2]
            abs_y2 = draw_y + bbox_glyph[3]

        char_bbox_data = {
            "char": char,
            "bbox_limpio": [abs_x1, abs_y1, abs_x2, abs_y2]
        }
        lista_bboxes_limpios.append(char_bbox_data)
        
        pos_x_actual += char_ancho_pixels + espacio

    return img, mask_img, lista_bboxes_limpios

    # Dibujar texto
    color_texto = (255, 255, 255) if "_oscura" in ruta_plantilla else (0, 0, 0)
    POS_Y_CENTRO = ALTO_IMG_LOCAL / 2
    pos_x_actual = (ANCHO_IMG_LOCAL / 2) - (ancho_total_calculado / 2)
    
    lista_bboxes_limpios = []
    
    bbox_letra_control = fnt_placa.getbbox("X")
    control_y_centro_local = (bbox_letra_control[3] + bbox_letra_control[1]) / 2
    draw_y_letra_equivalente = POS_Y_CENTRO - control_y_centro_local
    CONTROL_Y1 = draw_y_letra_equivalente + bbox_letra_control[1]
    CONTROL_Y2 = draw_y_letra_equivalente + bbox_letra_control[3]

    # Crear máscara limpia (blanco fondo, negro texto)
    mask_img = Image.new('L', (ANCHO_IMG_LOCAL, ALTO_IMG_LOCAL), 255)
    mask_draw = ImageDraw.Draw(mask_img)

    for char in texto_placa:
        if char == '-':
            font_actual = fnt_guion_generico
            espacio = espacio_guion
        else:
            font_actual = fnt_placa
            espacio = espacio_char

        bbox_glyph = font_actual.getbbox(char)
        char_ancho_pixels = bbox_glyph[2] - bbox_glyph[0]
        
        char_centro_y_local = (bbox_glyph[3] + bbox_glyph[1]) / 2
        draw_y = POS_Y_CENTRO - char_centro_y_local
        draw_x = pos_x_actual - bbox_glyph[0]
        
        # Dibujar en la imagen final coloreada
        draw.text((draw_x, draw_y), char, font=font_actual, fill=color_texto)
        # Dibujar en la máscara (negro = texto)
        mask_draw.text((draw_x, draw_y), char, font=font_actual, fill=0)
        
        if char == '-':
            abs_x1 = draw_x + bbox_glyph[0]
            abs_x2 = draw_x + bbox_glyph[2]
            abs_y1 = CONTROL_Y1
            abs_y2 = CONTROL_Y2
        else:
            abs_x1 = draw_x + bbox_glyph[0]
            abs_y1 = draw_y + bbox_glyph[1]
            abs_x2 = draw_x + bbox_glyph[2]
            abs_y2 = draw_y + bbox_glyph[3]

        char_bbox_data = {
            "char": char,
            "bbox_limpio": [abs_x1, abs_y1, abs_x2, abs_y2]
        }
        lista_bboxes_limpios.append(char_bbox_data)
        
        pos_x_actual += char_ancho_pixels + espacio

    return img, mask_img, lista_bboxes_limpios

# --- OPTIMIZACIÓN 4: Augmentations vectorizadas ---
def aplicar_augmentations_realistas(img_pil, bboxes_limpios, mask_pil=None):
    """Versión optimizada con operaciones vectorizadas.
       Ahora acepta mask_pil (PIL 'L') y devuelve:
       img_augmented_cv (BGR, con augmentations),
       img_rotated_cv (BGR, solo warp/perspective sin blur/noise),
       mask_rotated_cv (single channel, warped mask) o None,
       (x_placa, y_placa, w_placa, h_placa),
       lista_bboxes_transformados
    """
    
    # Parámetros de configuración
    BLUR_MULTIPLIER_MIN = 1.5
    BLUR_MULTIPLIER_MAX = 3
    NOISE_MULTIPLIER_MIN = 1.5
    NOISE_MULTIPLIER_MAX = 3
    
    PROB_BLUR = 1
    PROB_NOISE = 1
    PROB_JPG = 1
    PROB_OCCLUSION = 0.5
    PROB_INVERT = 0.4

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    alto, ancho = img_cv.shape[:2]

    MIN_BRIGHTNESS_OFFSET = -100
    MAX_ROTATION_Z = 30
    MAX_YAW_SHEAR = 0.6

    # Convertir mask si se dio
    mask_cv = None
    if mask_pil is not None:
        if mask_pil.mode != 'L':
            mask_pil = mask_pil.convert('L')
        mask_cv = np.array(mask_pil)  # single channel, 0..255

    if random.random() < PROB_INVERT:
        img_cv = cv2.bitwise_not(img_cv)
    
    if random.random() < 0.5:
        b, g, r = random.randint(80, 150), random.randint(80, 150), random.randint(80, 150)
        borderValue = (b, g, r)
    else:
        gris = random.randint(50, 200)
        borderValue = (gris, gris, gris)

    # Transformación geométrica
    pts_origen = np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]])
    angulo = math.radians(random.uniform(-MAX_ROTATION_Z, MAX_ROTATION_Z))
    cos_a, sin_a = math.cos(angulo), math.sin(angulo)
    shear_factor = random.uniform(-MAX_YAW_SHEAR, MAX_YAW_SHEAR) * 0.5
    escala = random.uniform(0.6, 1.0)
    cx, cy = ancho / 2, alto / 2
    
    pts_destino = []
    for (x, y) in pts_origen:
        x_scaled, y_scaled = (x - cx) * escala, (y - cy) * escala
        x_rot, y_rot = x_scaled * cos_a - y_scaled * sin_a, x_scaled * sin_a + y_scaled * cos_a
        x_shear, y_shear = x_rot + y_rot * shear_factor, y_rot
        pts_destino.append([x_shear + cx, y_shear + cy])
    
    pts_destino = np.float32(pts_destino)
    matriz_perspectiva = cv2.getPerspectiveTransform(pts_origen, pts_destino)

    # warpPerspective: crear imagen rotada (sin aplicar blur/noise/occlusion)
    img_rotated_cv = cv2.warpPerspective(img_cv, matriz_perspectiva, (ancho, alto), borderValue=borderValue)

    # Aplicar misma transformación a la máscara (si existe), usando INTER_NEAREST para mantener bordes nítidos
    mask_rotated_cv = None
    if mask_cv is not None:
        mask_rotated_cv = cv2.warpPerspective(mask_cv, matriz_perspectiva, (ancho, alto), borderValue=255, flags=cv2.INTER_NEAREST)

    # Ahora tomar img_rotated_cv y seguir con augmentations (solo sobre la imagen, no sobre la máscara)
    img_cv = img_rotated_cv.copy()

    # Oclusiones
    if random.random() < PROB_OCCLUSION:
        num_manchas = random.randint(1, 4)
        for _ in range(num_manchas):
            x_centro = random.randint(0, ancho - 1)
            y_centro = random.randint(0, alto - 1)
            radio = random.randint(2, 8)
            color_mancha = random.randint(20, 100)
            cv2.circle(img_cv, (x_centro, y_centro), radio, (color_mancha, color_mancha, color_mancha), -1)

    # Brillo y contraste (operación vectorizada)
    if random.random() < 0.2:
        brillo_offset = int(random.uniform(MIN_BRIGHTNESS_OFFSET, -50))
        contraste = random.uniform(0.6, 1.0)
    else:
        brillo_offset = int(random.uniform(-60, 60))
        contraste = random.uniform(0.7, 1.4)
    
    img_cv = cv2.convertScaleAbs(img_cv, alpha=contraste, beta=brillo_offset)
    
    # Blur
    if random.random() < PROB_BLUR:
        current_blur_multiplier = random.uniform(BLUR_MULTIPLIER_MIN, BLUR_MULTIPLIER_MAX)
        k = int(random.choice([3, 5]) * current_blur_multiplier)
        k = max(3, k if k % 2 != 0 else k + 1)
        sigma = 0.8 * current_blur_multiplier
        img_cv = cv2.GaussianBlur(img_cv, (k, k), sigma)

    # Ruido (vectorizado)
    if random.random() < PROB_NOISE:
        current_noise_multiplier = random.uniform(NOISE_MULTIPLIER_MIN, NOISE_MULTIPLIER_MAX)
        desviacion = random.uniform(5, 25) * current_noise_multiplier
        ruido = np.random.normal(0, desviacion, img_cv.shape).astype(np.float32)
        img_cv = np.clip(img_cv.astype(np.float32) + ruido, 0, 255).astype(np.uint8)

    # Compresión JPEG
    if random.random() < PROB_JPG:
        calidad_jpg = random.randint(30, 75)
        _, buffer = cv2.imencode(".jpg", img_cv, [cv2.IMWRITE_JPEG_QUALITY, calidad_jpg])
        img_cv = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    # Cálculo de bounding boxes (vectorizado)
    corners_placa = np.float32([[0, 0], [ancho, 0], [ancho, alto], [0, alto]]).reshape(4, 1, 2)
    corners_placa_tf = cv2.perspectiveTransform(corners_placa, matriz_perspectiva)

    lista_bboxes_transformados = []
    for item in bboxes_limpios:
        x1, y1, x2, y2 = item["bbox_limpio"]
        char_corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(4, 1, 2)
        char_corners_tf = cv2.perspectiveTransform(char_corners, matriz_perspectiva)
        lista_bboxes_transformados.append({
            "char": item["char"],
            "corners_tf": char_corners_tf
        })
    
    (x_placa, y_placa, w_placa, h_placa) = cv2.boundingRect(np.int32(corners_placa_tf))
    
    padding_x = int(w_placa * 0.20)
    padding_y = int(h_placa * 0.20)
    x_placa = max(0, x_placa - padding_x)
    y_placa = max(0, y_placa - padding_y)
    w_placa = min(ancho - x_placa, w_placa + padding_x * 2)
    h_placa = min(alto - y_placa, h_placa + padding_y * 2)

    return img_cv, img_rotated_cv, mask_rotated_cv, (x_placa, y_placa, w_placa, h_placa), lista_bboxes_transformados

# --- OPTIMIZACIÓN 5: Función worker para multiprocessing ---
# --- NUEVO: inicializador para cada worker ---
def init_worker(plantilla_paths, fuentes):
    """
    Inicializa variables globales en cada worker:
    - carga PLANTILLAS_CACHE (resized a ANCHO_TRABAJO x ALTO_TRABAJO para ahorrar tiempo luego)
    - asigna FUENTES_DISPONIBLES globalmente
    """
    global PLANTILLAS_CACHE, FUENTES_DISPONIBLES, FONT_CACHE
    # reasignar lista de fuentes global en worker (solo rutas)
    FUENTES_DISPONIBLES = list(fuentes)
    FONT_CACHE = {}  # cache local por proceso

    PLANTILLAS_CACHE = {}
    for ruta in plantilla_paths:
        try:
            img = Image.open(ruta)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # reducir/resamplear ahora para evitar hacerlo en cada tarea
            img_rs = img.resize((ANCHO_TRABAJO, ALTO_TRABAJO), Image.Resampling.LANCZOS)
            PLANTILLAS_CACHE[ruta] = img_rs.copy()
            img.close()
        except Exception as e:
            # no abortar al inicializar si hay plantillas malas
            print(f"[worker init {os.getpid()}] Error cargando {ruta}: {e}")

    # opcional: avisar (útil para debug)
    # print(f"[worker {os.getpid()}] cargadas {len(PLANTILLAS_CACHE)} plantillas, {len(FUENTES_DISPONIBLES)} fuentes")

# --- AJUSTE: generar_imagen_worker usa FUENTES_DISPONIBLES global y captura excepciones ---
def generar_imagen_worker(args):
    """Worker function para procesamiento paralelo. args = (idx, ruta_plantilla, seed)"""
    try:
        idx, ruta_plantilla, seed = args
        # semilla por proceso/tarea
        random.seed(seed + idx)
        np.random.seed((seed + idx) & 0xFFFFFFFF)

        # texto
        texto_placa = generar_texto_placa()

        # estampar (usa PLANTILLAS_CACHE global cargada por init_worker)
        img_limpia_pil, mask_limpia_pil, bboxes_limpios = estampar_placa(ruta_plantilla, FUENTES_DISPONIBLES, texto_placa)

        if img_limpia_pil is None:
            return None

        # augmentations (devuelve img_augmented, img_rotated_clean, mask_rotated, bbox crop)
        img_aumentada_grande, img_rotated_grande, mask_grande, (x_crop, y_crop, w_crop, h_crop), lista_bboxes_transformados = \
            aplicar_augmentations_realistas(img_limpia_pil, bboxes_limpios, mask_pil=mask_limpia_pil)

        # crop y resize para imagen final
        try:
            img_recortada = img_aumentada_grande[y_crop : y_crop + h_crop, x_crop : x_crop + w_crop]
            img_final_cv = cv2.resize(img_recortada, (ANCHO_PLACA, ALTO_PLACA), interpolation=cv2.INTER_AREA)
        except Exception:
            return None

        # crop y resize para imagen "clean" (rotated but sin postprocesado)
        try:
            img_rotated_rec = img_rotated_grande[y_crop : y_crop + h_crop, x_crop : x_crop + w_crop]
            img_clean_cv = cv2.resize(img_rotated_rec, (ANCHO_PLACA, ALTO_PLACA), interpolation=cv2.INTER_AREA)
        except Exception:
            img_clean_cv = None

        # crop y resize para mascara (si existe)
        mask_final_resized = None
        if mask_grande is not None:
            try:
                mask_rec = mask_grande[y_crop : y_crop + h_crop, x_crop : x_crop + w_crop]
                mask_final_resized = cv2.resize(mask_rec, (ANCHO_PLACA, ALTO_PLACA), interpolation=cv2.INTER_NEAREST)
            except Exception:
                mask_final_resized = None

        # json de bboxes (mismo código tuyo)
        escala_x_final = ANCHO_PLACA / max(1, w_crop)
        escala_y_final = ALTO_PLACA / max(1, h_crop)

        datos_json_finales = []
        for item in lista_bboxes_transformados:
            char = item["char"]
            corners_en_grande = item["corners_tf"]
            corners_traducidos = corners_en_grande - [x_crop, y_crop]
            corners_finales = corners_traducidos * [escala_x_final, escala_y_final]
            (bx, by, bw, bh) = cv2.boundingRect(np.int32(corners_finales))
            datos_json_finales.append({
                "char": char,
                "bbox_xywh": [int(bx), int(by), int(bw), int(bh)]
            })

        # encode imagen final (jpg)
        calidad_jpeg = random.randint(75, 95)
        _, img_encoded = cv2.imencode('.jpg', img_final_cv, [int(cv2.IMWRITE_JPEG_QUALITY), calidad_jpeg])
        img_bytes = img_encoded.tobytes()

        # encode imagen clean (png si existe)
        clean_bytes = None
        if img_clean_cv is not None:
            # img_clean_cv está en BGR, usar png para preservar sin compresión con pérdida
            _, clean_encoded = cv2.imencode('.png', img_clean_cv)
            clean_bytes = clean_encoded.tobytes()

        # encode mask (png)
        mask_bytes = None
        if mask_final_resized is not None:
            # mask_final_resized es single-channel, 0..255 (0 texto, 255 fondo)
            _, mask_encoded = cv2.imencode('.png', mask_final_resized)
            mask_bytes = mask_encoded.tobytes()

        return {
            'nombre': texto_placa,
            'img_bytes': img_bytes,
            'clean_bytes': clean_bytes,
            'mask_bytes': mask_bytes,
            'json_data': datos_json_finales
        }

    except Exception as e:
        # devolver info de error para debug en main (no romper pool)
        return {'_error': True, 'exception': repr(e)}

# --- AJUSTE: generar_dataset_paralelo ---
def generar_dataset_paralelo(cantidad, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    print(f"Usando {num_workers} workers (cores: {cpu_count()})")

    lista_plantillas = glob.glob(os.path.join(DIR_PLANTILLAS, "*.jpg"))
    lista_plantillas.extend(glob.glob(os.path.join(DIR_PLANTILLAS, "*.png")))

    if not lista_plantillas or not FUENTES_DISPONIBLES:
        print("ERROR: Faltan plantillas o fuentes")
        return

    print(f"Cargando {len(lista_plantillas)} plantillas (main) para inicializar workers...")
    # PASAMOS SOLO LAS RUTAS a los workers; ellos cargarán sus propias mini-caches en init_worker
    plantilla_paths = lista_plantillas

    print(f"Iniciando generación de {cantidad} imágenes...")
    print(f"Usando {len(FUENTES_DISPONIBLES)} fuentes y {len(plantilla_paths)} plantillas.")

    # preparar args: (idx, ruta_plantilla, seed)  <-- no pasamos la lista completa de fuentes por tarea
    args_list = []
    for i in range(cantidad):
        ruta_plantilla = random.choice(plantilla_paths)
        seed = random.randint(0, 2**32 - 1)
        args_list.append((i, ruta_plantilla, seed))

    generados_count = 0
    # elegir chunksize pequeño para feedback rápido (puedes subirlo si hay overhead)
    chunksize = 1

    with Pool(processes=num_workers, initializer=init_worker, initargs=(plantilla_paths, FUENTES_DISPONIBLES)) as pool:
        for resultado in tqdm(pool.imap_unordered(generar_imagen_worker, args_list, chunksize=chunksize),
                             total=cantidad, desc="Generando imágenes"):
            if resultado is None:
                continue
            if isinstance(resultado, dict) and resultado.get('_error'):
                # opcional: imprimir trace para debug
                print("Worker error:", resultado['exception'])
                continue

            nombre_base = resultado['nombre']
            ruta_img = os.path.join(DIR_DATASET, f"{nombre_base}.jpg")
            with open(ruta_img, 'wb') as f:
                f.write(resultado['img_bytes'])

            # guardar imagen clean (si existe)
            if resultado.get('clean_bytes'):
                ruta_clean = os.path.join(DIR_DATASET, f"{nombre_base}_clean.png")
                with open(ruta_clean, 'wb') as f:
                    f.write(resultado['clean_bytes'])

            # guardar máscara binaria (si existe)
            if resultado.get('mask_bytes'):
                ruta_mask = os.path.join(DIR_DATASET, f"{nombre_base}_mask.png")
                with open(ruta_mask, 'wb') as f:
                    f.write(resultado['mask_bytes'])

            ruta_json = os.path.join(DIR_DATASET, f"{nombre_base}.json")
            with open(ruta_json, 'w') as f:
                json.dump(resultado['json_data'], f, indent=2)

            generados_count += 1

    print(f"\n¡Proceso completado! Se generaron {generados_count} imágenes en '{DIR_DATASET}'.")


# --- SCRIPT PRINCIPAL ---
if __name__ == '__main__':
    # Para usar multiprocessing en Windows
    import multiprocessing
    multiprocessing.freeze_support()
    
    CANTIDAD_DATASET = 1000
    
    # Generar dataset con todos los cores disponibles
    generar_dataset_paralelo(CANTIDAD_DATASET, num_workers=None)  # None = automático
    
    # O especificar manualmente:
    # generar_dataset_paralelo(CANTIDAD_DATASET, num_workers=6)  # Usar 6 cores
