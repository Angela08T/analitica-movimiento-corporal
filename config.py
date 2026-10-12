"""Configuración del piloto. Ajusta los umbrales según la cámara y el lugar."""
import os

# Fuente de video: 0 = webcam, "video.mp4" = archivo, "rtsp://usuario:clave@IP:554/stream" = cámara IP
FUENTE_VIDEO = 0

# Modelo de pose (se descarga solo la primera vez). Opciones: yolo11n-pose (rápido) ... yolo11x-pose (preciso)
MODELO_POSE = "yolo11n-pose.pt"
CONFIANZA_MIN = 0.5      # confianza mínima para aceptar una persona
CONF_KEYPOINT = 0.5      # confianza mínima para usar un punto del cuerpo (muñeca, hombro, etc.)

# --- Umbrales de comportamiento ---
# Las velocidades se miden en "alturas corporales por segundo" para no depender de la distancia a la cámara.
MANOS_ARRIBA_SEG = 1.5          # ambas manos sobre la cabeza durante este tiempo
CAIDA_ANGULO = 60               # inclinación del torso (grados desde la vertical) para considerarlo en el suelo
CAIDA_RATIO = 1.3               # o caja más ancha que alta en esta proporción
CAIDA_SEG = 2.0
VEL_CARRERA = 1.5
CARRERA_SEG = 1.0
VEL_GOLPE = 3.0                 # velocidad de muñeca considerada "brusca" (golpe, empujón)
AGRESION_VENTANA = 2.0          # segundos que se revisan hacia atrás
AGRESION_TIEMPO_BRUSCO = 0.4    # segundos acumulados de movimiento brusco dentro de la ventana
DISTANCIA_CERCANIA = 1.2        # dos personas están "cerca" si sus centros distan menos de esto (en alturas)
MERODEO_SEG = 60                # tiempo en escena para considerar merodeo

# --- Alertas ---
COOLDOWN_SEG = 30               # no repetir la misma alerta para la(s) misma(s) persona(s) antes de este tiempo
CARPETA_ALERTAS = "alertas"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# tipo: (descripción, severidad)
TIPOS_EVENTO = {
    "MANOS_ARRIBA": ("Persona con manos arriba (posible asalto o amenaza)", "ALTA"),
    "POSIBLE_AGRESION": ("Movimientos bruscos entre personas cercanas (posible pelea)", "ALTA"),
    "CAIDA": ("Persona en el suelo (posible caída o agresión)", "MEDIA"),
    "CARRERA": ("Persona corriendo (posible huida)", "BAJA"),
    "MERODEO": ("Persona permanece mucho tiempo en la zona", "BAJA"),
}
