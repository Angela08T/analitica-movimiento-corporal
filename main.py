"""Piloto: detección de comportamientos sospechosos por lenguaje corporal desde una cámara."""
import argparse
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

import config
from alertas import GestorAlertas
from comportamiento import AnalizadorComportamiento, Persona

COLOR_SEVERIDAD = {"ALTA": (0, 0, 255), "MEDIA": (0, 165, 255), "BAJA": (0, 255, 255)}


def extraer_personas(resultado):
    cajas, kps = resultado.boxes, resultado.keypoints
    if cajas is None or cajas.id is None or kps is None:
        return []
    ids = cajas.id.int().cpu().tolist()
    xyxy = cajas.xyxy.cpu().numpy()
    puntos = kps.xy.cpu().numpy()
    confs = kps.conf.cpu().numpy() if kps.conf is not None else np.ones(puntos.shape[:2])
    return [Persona(pid, xyxy[n], puntos[n], confs[n]) for n, pid in enumerate(ids)]


def dibujar_etiquetas(vista, personas, etiquetas):
    for p in personas:
        x1, y1 = int(p.caja[0]), int(p.caja[1])
        for k, tipo in enumerate(sorted(etiquetas.get(p.id, ()))):
            color = COLOR_SEVERIDAD[config.TIPOS_EVENTO[tipo][1]]
            cv2.putText(vista, tipo, (x1, max(20, y1 - 30 - 22 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def abrir_fuente(fuente):
    """En Windows las cámaras locales se abren con DirectShow o Media Foundation; el backend por defecto falla."""
    if isinstance(fuente, int) and sys.platform == "win32":
        for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            cap = cv2.VideoCapture(fuente, api)
            if cap.isOpened():
                return cap
            cap.release()
    return cv2.VideoCapture(fuente)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fuente", default=str(config.FUENTE_VIDEO), help="0 = webcam, ruta de video o URL RTSP")
    parser.add_argument("--sin-ventana", action="store_true", help="no mostrar el video (uso en servidor)")
    args = parser.parse_args()

    fuente = int(args.fuente) if args.fuente.isdigit() else args.fuente
    es_archivo = isinstance(fuente, str) and not fuente.lower().startswith(("rtsp://", "http://", "https://"))

    cap = abrir_fuente(fuente)
    if not cap.isOpened():
        if isinstance(fuente, str) and fuente.lower().startswith("http"):
            pista = "Si usas la app IP Webcam, la dirección debe terminar en /video (ej. http://IP:8080/video)."
        else:
            pista = "Revisa que la cámara esté conectada y que ninguna otra app (Zoom, Teams...) la esté usando."
        raise SystemExit(f"No se pudo abrir la fuente de video: {fuente}\n{pista}")

    modelo = YOLO(config.MODELO_POSE)
    analizador = AnalizadorComportamiento()
    alertas = GestorAlertas()
    print("Analizando... presiona 'q' en la ventana para salir.")

    t_anterior = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            if es_archivo:
                break
            print("Se perdió la señal de la cámara, reintentando...")
            time.sleep(1)
            cap.release()
            cap = abrir_fuente(fuente)
            continue

        # En archivos se usa el tiempo del video para que los umbrales en segundos sean reales
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if es_archivo else time.time()

        resultado = modelo.track(frame, persist=True, classes=[0], conf=config.CONFIANZA_MIN, verbose=False)[0]
        personas = extraer_personas(resultado)
        for evento in analizador.actualizar(personas, t):
            alertas.emitir(evento, frame)

        if not args.sin_ventana:
            vista = resultado.plot()
            dibujar_etiquetas(vista, personas, analizador.etiquetas)
            ahora = time.time()
            fps = 1 / max(ahora - t_anterior, 1e-6)
            t_anterior = ahora
            cv2.putText(vista, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Analitica de movimiento corporal", vista)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
