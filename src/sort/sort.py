"""
sort.py
-------

- Mantiene una lista de "tracks" (carros).
- En cada frame:
  - Hace matching de detecciones actuales con tracks anteriores usando IoU.
  - Actualiza la bbox de los tracks matcheados.
  - Crea nuevos tracks para detecciones sin match.
  - Elimina tracks que han estado muchos frames sin ser vistos (max_age).
- Devuelve un np.ndarray de shape (N, 5) con:
      [x1, y1, x2, y2, track_id]
"""

from __future__ import annotations

import numpy as np


def iou(boxA, boxB) -> float:
    """
    Docstring for iou
    
    :param boxA: Primera caja delimitadora.
    :param boxB: Segunda caja delimitadora.
    :return: Description
    :rtype: Valor IoU entre 0.0 y 1.0.
    Calcula IoU entre dos cajas [x1, y1, x2, y2].
    Las cajas deben estar en formato [x1, y1, x2, y2]
    """
    
    #Coordenadas del rectángulo de intersección
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Area de intersección
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    boxA_area = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxB_area = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    if boxA_area <= 0 or boxB_area <= 0:
        return 0.0

    # Firmula: Area de Interseccion / (Area A + Area B - Area de Interseccion)
    return inter_area / float(boxA_area + boxB_area - inter_area)


class Track:
    """
    Track sencillo que mantiene:
    - id único
    - bbox actual
    - cuántos frames lleva sin ser visto (misses)
    - cuántos hits ha tenido (hits)
    """
    # Contador de ID estático para asegurar IDs unicos
    _next_id = 0

    def __init__(self, bbox):
        # bbox: [x1, y1, x2, y2]
        self.id = Track._next_id
        Track._next_id += 1

        # Almacena la bbox como un array de NumPy
        self.bbox = np.array(bbox, dtype=float)
        self.misses = 0
        self.hits = 1

    def update(self, bbox):
        """
        Docstring for update
        
        :param self: Description
        :param bbox: Description
        Actualiza la posición del track con una nueva detección emparejada (hit)
        """
        self.bbox = np.array(bbox, dtype=float)
        self.misses = 0
        self.hits += 1

    def mark_missed(self):
        self.misses += 1


class Sort:
    """
    Tracker tipo SORT (muy simplificado, sin Kalman).

    Parameters
    ----------
    max_age : int
        Número máximo de frames que se permite a un track estar "perdido"
        (sin detecciones) antes de eliminarlo.
    iou_threshold : float
        IoU mínima para considerar que una detección pertenece a un track.
    """

    def __init__(self, max_age=5, iou_threshold=0.3):
        """
        Docstring for __init__
        
        :param self: D
        :param max_age: Número máximo de fotogramas que un track puede estar
                           "perdido" antes de ser eliminado.
        :param iou_threshold: mbral IoU mínimo requerido para considerar
                                   que una detección coincide con un track.
        """
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []

    def update(self, dets):
        """
        Actualiza el conjunto de tracks dada una lista/np.ndarray de detecciones.

        Parameters
        ----------
        dets : array-like, shape (N, 5)
            Detecciones del frame actual: [x1, y1, x2, y2, score]

        Returns
        -------
        np.ndarray, shape (M, 5)
            Tracks activos: [x1, y1, x2, y2, track_id]
        """
        # Convertir a np.ndarray
        dets = np.asarray(dets, dtype=float)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        if dets.size == 0:
            # No hay detecciones: solo aumentamos misses de tracks existentes
            for trk in self.tracks:
                trk.mark_missed()
            # Eliminar tracks que exceden max_age
            self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
            # Devolver lo que quede
            if not self.tracks:
                return np.empty((0, 5))
            out = []
            for trk in self.tracks:
                x1, y1, x2, y2 = trk.bbox.tolist()
                out.append([x1, y1, x2, y2, trk.id])
            return np.asarray(out, dtype=float)

        # Si no hay tracks, crear uno por cada detección
        if len(self.tracks) == 0:
            self.tracks = [Track(det[:4]) for det in dets]
        else:
            # Matching detecciones <-> tracks via IoU greedy
            num_trks = len(self.tracks)
            num_dets = dets.shape[0]

            iou_matrix = np.zeros((num_trks, num_dets), dtype=float)
            for t_idx, trk in enumerate(self.tracks):
                for d_idx in range(num_dets):
                    iou_matrix[t_idx, d_idx] = iou(trk.bbox, dets[d_idx, :4])

            # Conjuntos de índices aún no emparejados
            unmatched_tracks = set(range(num_trks))
            unmatched_dets = set(range(num_dets))
            matches = []

            # Greedy: tomamos el par (t, d) de mayor IoU mientras supere el umbral
            while True:
                if len(unmatched_tracks) == 0 or len(unmatched_dets) == 0:
                    break

                best_iou = 0.0
                best_t = None
                best_d = None
                for t_idx in unmatched_tracks:
                    for d_idx in unmatched_dets:
                        iou_val = iou_matrix[t_idx, d_idx]
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_t = t_idx
                            best_d = d_idx

                if best_t is None or best_d is None or best_iou < self.iou_threshold:
                    break

                # Aceptar match
                matches.append((best_t, best_d))
                unmatched_tracks.remove(best_t)
                unmatched_dets.remove(best_d)

            # Actualizar tracks emparejados
            for t_idx, d_idx in matches:
                self.tracks[t_idx].update(dets[d_idx, :4])

            # Marcar tracks no emparejados como perdidos
            for t_idx in unmatched_tracks:
                self.tracks[t_idx].mark_missed()

            # Crear nuevos tracks para detecciones no emparejadas
            for d_idx in unmatched_dets:
                self.tracks.append(Track(dets[d_idx, :4]))

            # Eliminar tracks muy viejos
            self.tracks = [t for t in self.tracks if t.misses <= self.max_age]

        # Construir salida: [x1, y1, x2, y2, track_id]
        if not self.tracks:
            return np.empty((0, 5))

        out = []
        for trk in self.tracks:
            x1, y1, x2, y2 = trk.bbox.tolist()
            out.append([x1, y1, x2, y2, trk.id])

        return np.asarray(out, dtype=float)
