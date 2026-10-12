# Piloto: analítica de movimiento corporal

Analiza el video de una cámara, estima la pose (esqueleto) de cada persona y genera alertas
cuando detecta comportamientos que podrían indicar un incidente.

## Cómo funciona

```
Cámara (webcam / IP RTSP / video)
   → YOLO11-pose: detecta personas + 17 puntos del cuerpo + ID de seguimiento
   → Reglas de comportamiento (comportamiento.py)
   → Alertas (alertas.py): foto JPG + registro CSV + consola + Telegram (opcional)
```

| Comportamiento     | Regla                                                         | Severidad |
|--------------------|---------------------------------------------------------------|-----------|
| MANOS_ARRIBA       | Ambas muñecas sobre la cabeza ≥ 1.5 s                          | Alta      |
| POSIBLE_AGRESION   | Movimientos bruscos de brazos con otra persona muy cerca       | Alta      |
| CAIDA              | Torso horizontal o cuerpo tendido ≥ 2 s                        | Media     |
| CARRERA            | Desplazamiento rápido ≥ 1 s                                    | Baja      |
| MERODEO            | Persona en escena ≥ 60 s                                       | Baja      |

Todos los umbrales están en `config.py`.

## Instalación (Windows)

1. Instalar Python 3.11 desde https://www.python.org/downloads/ (marcar **"Add python.exe to PATH"**),
   o en PowerShell: `winget install Python.Python.3.11`
2. En la carpeta del proyecto:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso

```powershell
python main.py                       # webcam
python main.py --fuente prueba.mp4   # archivo de video
python main.py --fuente "rtsp://usuario:clave@192.168.1.50:554/stream1"   # cámara IP
```

Las alertas se guardan en `alertas/` (imágenes + `registro.csv`).

### Alertas por Telegram (opcional)

1. Crear un bot con @BotFather y copiar el token.
2. Escribirle al bot y obtener el chat id en `https://api.telegram.org/bot<TOKEN>/getUpdates`.
3. Antes de ejecutar:

```powershell
$env:TELEGRAM_TOKEN = "123456:ABC..."
$env:TELEGRAM_CHAT_ID = "987654321"
python main.py
```

## Limitaciones del piloto

- Detecta **comportamientos observables**, no "delitos". Toda alerta debe revisarla una persona.
- Habrá falsos positivos (alguien estirándose = manos arriba, agacharse = caída) y falsos negativos.
- Funciona mejor con cámara a la altura de 2–3 m, buena iluminación y personas de cuerpo entero.
- Sin GPU NVIDIA, espera ~5–15 FPS en CPU con `yolo11n-pose.pt`.

## Siguientes pasos sugeridos

- Grabar videos de prueba (actuados) de cada situación y medir aciertos / falsos positivos.
- Detección de armas: entrenar un modelo YOLO con un dataset de armas (p. ej. de Roboflow).
- Reconocimiento de acciones con modelos temporales (p. ej. clasificador sobre secuencias de pose)
  entrenado con datos propios, en lugar de reglas fijas.
- Zonas de interés (solo alertar en ciertas áreas) y panel web para operadores.
