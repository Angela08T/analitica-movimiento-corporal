"""Reglas que traducen la pose de cada persona en comportamientos sospechosos."""
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config

# Índices de los 17 puntos del cuerpo (formato COCO) que entrega YOLO pose
NARIZ = 0
HOMBRO_I, HOMBRO_D = 5, 6
MUNECA_I, MUNECA_D = 9, 10
CADERA_I, CADERA_D = 11, 12

HISTORIA_SEG = 3.0       # cuánto historial de movimiento se guarda por persona
OLVIDAR_SEG = 5.0        # se borra a una persona que no se ve hace este tiempo


@dataclass
class Persona:
    id: int
    caja: np.ndarray      # x1, y1, x2, y2
    kps: np.ndarray       # (17, 2) coordenadas de los puntos
    conf: np.ndarray      # (17,) confianza de cada punto

    @property
    def altura(self):
        return max(float(self.caja[3] - self.caja[1]), 1.0)

    @property
    def ancho(self):
        return max(float(self.caja[2] - self.caja[0]), 1.0)

    @property
    def centro(self):
        return np.array([(self.caja[0] + self.caja[2]) / 2, (self.caja[1] + self.caja[3]) / 2])

    def punto(self, idx):
        return self.kps[idx] if self.conf[idx] >= config.CONF_KEYPOINT else None

    def medio(self, a, b):
        pa, pb = self.punto(a), self.punto(b)
        return None if pa is None or pb is None else (pa + pb) / 2


@dataclass
class Evento:
    tipo: str
    ids: tuple
    caja: np.ndarray      # zona a resaltar en la captura
    t: float


@dataclass
class Historial:
    primera_vez: float
    ultima_vez: float = 0.0
    centros: deque = field(default_factory=deque)   # (t, centro, altura)
    munecas: deque = field(default_factory=deque)   # (t, muñeca izq, muñeca der, altura)
    desde: dict = field(default_factory=dict)       # condición -> instante en que empezó


class AnalizadorComportamiento:
    def __init__(self):
        self.historial = {}
        self.etiquetas = {}   # id -> comportamientos activos en el cuadro actual (para dibujar)

    def actualizar(self, personas, t):
        eventos = []
        self.etiquetas = {p.id: set() for p in personas}

        for p in personas:
            h = self.historial.setdefault(p.id, Historial(primera_vez=t))
            h.ultima_vez = t
            h.centros.append((t, p.centro, p.altura))
            h.munecas.append((t, p.punto(MUNECA_I), p.punto(MUNECA_D), p.altura))
            for cola in (h.centros, h.munecas):
                while cola and t - cola[0][0] > HISTORIA_SEG:
                    cola.popleft()

            if self._sostenido(h, "MANOS_ARRIBA", self._manos_arriba(p), t, config.MANOS_ARRIBA_SEG):
                eventos.append(self._evento("MANOS_ARRIBA", [p], t))
            if self._sostenido(h, "CAIDA", self._en_suelo(p), t, config.CAIDA_SEG):
                eventos.append(self._evento("CAIDA", [p], t))
            corre = self._velocidad_centro(h, t) > config.VEL_CARRERA
            if self._sostenido(h, "CARRERA", corre, t, config.CARRERA_SEG):
                eventos.append(self._evento("CARRERA", [p], t))
            if t - h.primera_vez >= config.MERODEO_SEG:
                eventos.append(self._evento("MERODEO", [p], t))

        eventos += self._agresion(personas, t)
        self._limpiar(t)
        return eventos

    # ---------- reglas individuales ----------

    def _manos_arriba(self, p):
        izq, der = p.punto(MUNECA_I), p.punto(MUNECA_D)
        if izq is None or der is None:
            return False
        nariz = p.punto(NARIZ)
        if nariz is not None:
            limite = nariz[1]
        else:
            hombros = p.medio(HOMBRO_I, HOMBRO_D)
            if hombros is None:
                return False
            limite = hombros[1] - 0.15 * p.altura
        return izq[1] < limite and der[1] < limite   # en imagen, y menor = más arriba

    def _en_suelo(self, p):
        hombros, caderas = p.medio(HOMBRO_I, HOMBRO_D), p.medio(CADERA_I, CADERA_D)
        if hombros is not None and caderas is not None:
            dx, dy = caderas - hombros
            if math.degrees(math.atan2(abs(dx), abs(dy))) > config.CAIDA_ANGULO:
                return True
        return p.ancho / p.altura > config.CAIDA_RATIO

    @staticmethod
    def _velocidad_centro(h, t, ventana=0.5):
        anteriores = [m for m in h.centros if t - m[0] >= ventana]
        if not anteriores:
            return 0.0
        t0, c0, a0 = anteriores[-1]
        _, c1, a1 = h.centros[-1]
        return float(np.linalg.norm(c1 - c0)) / (t - t0) / ((a0 + a1) / 2)

    @staticmethod
    def _tiempo_brusco(h, t):
        """Segundos, dentro de la ventana, en que alguna muñeca se movió a velocidad de golpe."""
        total = 0.0
        muestras = [m for m in h.munecas if t - m[0] <= config.AGRESION_VENTANA]
        for (t0, i0, d0, _), (t1, i1, d1, altura) in zip(muestras, muestras[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            for p0, p1 in ((i0, i1), (d0, d1)):
                if p0 is not None and p1 is not None and np.linalg.norm(p1 - p0) / dt / altura > config.VEL_GOLPE:
                    total += dt
                    break
        return total

    # ---------- reglas entre personas ----------

    def _agresion(self, personas, t):
        eventos = []
        bruscos = {p.id: self._tiempo_brusco(self.historial[p.id], t) >= config.AGRESION_TIEMPO_BRUSCO
                   for p in personas}
        for i, a in enumerate(personas):
            for b in personas[i + 1:]:
                if not (bruscos[a.id] or bruscos[b.id]):
                    continue
                distancia = np.linalg.norm(a.centro - b.centro) / ((a.altura + b.altura) / 2)
                if distancia < config.DISTANCIA_CERCANIA:
                    eventos.append(self._evento("POSIBLE_AGRESION", [a, b], t))
        return eventos

    # ---------- utilidades ----------

    @staticmethod
    def _sostenido(h, clave, condicion, t, segundos):
        if not condicion:
            h.desde.pop(clave, None)
            return False
        h.desde.setdefault(clave, t)
        return t - h.desde[clave] >= segundos

    def _evento(self, tipo, personas, t):
        for p in personas:
            self.etiquetas[p.id].add(tipo)
        cajas = np.array([p.caja for p in personas])
        union = np.array([cajas[:, 0].min(), cajas[:, 1].min(), cajas[:, 2].max(), cajas[:, 3].max()])
        return Evento(tipo, tuple(sorted(p.id for p in personas)), union, t)

    def _limpiar(self, t):
        for pid in [pid for pid, h in self.historial.items() if t - h.ultima_vez > OLVIDAR_SEG]:
            del self.historial[pid]
