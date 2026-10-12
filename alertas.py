"""Registro y envío de alertas: captura JPG, registro CSV, consola y (opcional) Telegram."""
import csv
import os
import threading
from datetime import datetime

import cv2
import requests

import config


class GestorAlertas:
    def __init__(self):
        os.makedirs(config.CARPETA_ALERTAS, exist_ok=True)
        self.registro = os.path.join(config.CARPETA_ALERTAS, "registro.csv")
        self._ultima = {}
        if not os.path.exists(self.registro):
            with open(self.registro, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["fecha_hora", "tipo", "severidad", "personas", "descripcion", "imagen"])

    def emitir(self, evento, frame):
        clave = (evento.tipo, evento.ids)
        if evento.t - self._ultima.get(clave, float("-inf")) < config.COOLDOWN_SEG:
            return
        self._ultima[clave] = evento.t

        descripcion, severidad = config.TIPOS_EVENTO[evento.tipo]
        ahora = datetime.now()
        ids = "-".join(map(str, evento.ids))
        imagen = os.path.join(config.CARPETA_ALERTAS, f"{ahora:%Y%m%d_%H%M%S}_{evento.tipo}_{ids}.jpg")

        captura = frame.copy()
        x1, y1, x2, y2 = map(int, evento.caja)
        cv2.rectangle(captura, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(captura, evento.tipo, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(imagen, captura)

        with open(self.registro, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([f"{ahora:%Y-%m-%d %H:%M:%S}", evento.tipo, severidad, ids, descripcion, imagen])

        print(f"[ALERTA {severidad}] {ahora:%H:%M:%S} {descripcion} (persona(s) {ids})")

        if config.TELEGRAM_TOKEN and config.TELEGRAM_CHAT_ID:
            texto = f"⚠️ Alerta {severidad}\n{descripcion}\n{ahora:%Y-%m-%d %H:%M:%S}"
            threading.Thread(target=self._telegram, args=(texto, imagen), daemon=True).start()

    @staticmethod
    def _telegram(texto, imagen):
        url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendPhoto"
        try:
            with open(imagen, "rb") as foto:
                respuesta = requests.post(url, data={"chat_id": config.TELEGRAM_CHAT_ID, "caption": texto},
                                          files={"photo": foto}, timeout=10)
            respuesta.raise_for_status()
        except Exception as e:
            print(f"No se pudo enviar la alerta a Telegram: {e}")
