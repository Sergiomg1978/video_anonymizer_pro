# FASE 4 — ANOTACIÓN MANUAL DE FOTOGRAMAS ANCLA

## Contexto

Estamos construyendo "video_anonymizer_pro". Las fases 0-3 están implementadas. Ahora implementa la GUI de anotación manual que permite al usuario marcar la cabeza de la mujer en fotogramas de referencia antes del procesamiento automático.

## Tarea 4A: Implementar annotation/manual_annotator.py

```python
class ManualAnnotator:
    """
    GUI para que el usuario anote manualmente fotogramas de referencia.
    
    FLUJO:
    1. Extrae N fotogramas representativos del vídeo (1 cada 2 segundos +
       fotogramas de cambios de escena).
    2. Pre-ejecuta los detectores de cara/persona en cada fotograma extraído.
    3. Muestra al usuario cada fotograma con las detecciones superpuestas.
    4. El usuario puede:
       a) Hacer clic en la detección correcta (la de la mujer) para confirmarla.
       b) Dibujar manualmente un rectángulo si el detector falló.
       c) Marcar "mujer no presente" si no está en ese fotograma.
       d) Saltar al siguiente fotograma.
    5. Al terminar, genera una lista de AnchorFrames con embeddings.
    
    INTERFAZ (PyQt6):
    - Ventana principal con el fotograma actual mostrado a tamaño adecuado.
    - Bounding boxes de las detecciones dibujados sobre la imagen:
      - Azul: detecciones de cara.
      - Verde: detecciones de persona.
      - El usuario hace clic dentro de un bbox para seleccionarlo como "la mujer".
    - Modo dibujo: botón para activar y luego arrastrar un rectángulo sobre la cabeza.
    - Barra de navegación inferior:
      - Slider para ir a cualquier fotograma.
      - Botones: "Anterior", "Siguiente", "Auto-detectar todo", "Mujer no presente".
    - Panel lateral con info:
      - Frame actual / Total de frames extraídos.
      - Frames ya anotados / Total.
      - Timestamp del frame actual.
    - Botón "Guardar y Salir" que genera los AnchorFrames.
    - Botón "Anotación Rápida" que:
      1. Auto-detecta en todos los frames.
      2. Asume que la persona más grande es la mujer.
      3. Muestra solo los frames dudosos para confirmación manual.
    
    ATAJOS DE TECLADO:
    - Flecha derecha: siguiente frame.
    - Flecha izquierda: anterior frame.
    - Enter: confirmar selección actual.
    - N: marcar "mujer no presente".
    - D: activar modo dibujo.
    - Q: guardar y salir.
    """

    def __init__(self, video_path: str, auto_extract_interval: float = 2.0):
        """
        Recibe la ruta del vídeo y el intervalo de extracción.
        Inicializa VideoReader y FrameExtractor.
        Carga detectores (MultiFaceDetector, PersonDetector, HeadDetector).
        """

    def auto_extract_keyframes(self) -> list[int]:
        """
        Extrae frames representativos:
        - 1 cada auto_extract_interval segundos.
        - Frames adicionales en cambios de escena.
        - Primer y último frame del vídeo.
        Devuelve lista de frame_numbers.
        """

    def _pre_detect(self, frames: list[tuple[int, np.ndarray]]) -> dict:
        """
        Ejecuta detección en todos los frames extraídos.
        Devuelve dict: {frame_number: {faces: [...], persons: [...], heads: [...]}}.
        Muestra barra de progreso durante la detección.
        """

    def launch_gui(self) -> list['AnchorFrame']:
        """
        Lanza la ventana PyQt6.
        Devuelve la lista de AnchorFrames cuando el usuario cierra.
        """

    def _quick_annotate(self, frames: list, detections: dict) -> list['AnchorFrame']:
        """
        Anotación rápida automática:
        1. En cada frame, identifica la persona más grande como la mujer.
        2. Extrae su embedding facial.
        3. Para frames donde la confianza de clasificación es baja,
           los marca para revisión manual.
        """
```

## Tarea 4B: Implementar annotation/anchor_frames.py

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np
import json
from pathlib import Path

@dataclass
class AnchorFrame:
    frame_number: int
    timestamp: float
    head_bbox: Optional[tuple[int, int, int, int]]  # None si mujer no presente
    woman_present: bool
    face_embedding: Optional[np.ndarray] = None
    body_embedding: Optional[np.ndarray] = None
    annotated_by: str = "manual"  # "manual" o "auto_confirmed"


class AnchorFrameManager:
    """
    Gestiona los fotogramas ancla anotados manualmente.
    
    Funciones:
    - Almacenar y recuperar AnchorFrames.
    - Guardar/cargar desde archivo JSON (para reutilizar entre ejecuciones).
    - Interpolar posiciones entre anclas para frames intermedios.
    - Proporcionar embeddings de referencia para el IdentityManager.
    """

    def __init__(self):
        """Inicializa la lista de anclas vacía."""

    def add_anchor(self, anchor: AnchorFrame):
        """Añade un ancla a la lista, manteniendo orden por frame_number."""

    def get_anchors(self) -> list[AnchorFrame]:
        """Devuelve todas las anclas ordenadas."""

    def get_reference_embeddings(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Calcula embedding facial de referencia de la mujer:
        - Promedia todos los embeddings faciales de las anclas donde woman_present=True.
        - Devuelve (adult_embedding, child_embedding_or_none).
        """

    def get_nearest_anchors(self, frame_number: int) -> tuple[Optional[AnchorFrame], Optional[AnchorFrame]]:
        """
        Devuelve las anclas más cercanas antes y después del frame dado.
        Útil para interpolación cuando el detector automático falla.
        """

    def interpolate_bbox(self, frame_number: int) -> Optional[tuple[int, int, int, int]]:
        """
        Interpola linealmente el bbox de la cabeza entre las dos anclas más cercanas.
        Solo si ambas anclas tienen woman_present=True y la distancia es razonable.
        """

    def save_to_file(self, filepath: str):
        """Guarda anclas a JSON. Convierte numpy arrays a listas."""

    def load_from_file(self, filepath: str):
        """Carga anclas desde JSON previamente guardado."""

    def get_woman_present_ratio(self) -> float:
        """Porcentaje de anclas donde la mujer está presente."""

    def get_annotated_count(self) -> int:
        """Número de anclas anotadas."""
```

## Verificación

1. La GUI se abre correctamente y muestra fotogramas del vídeo.
2. Se pueden navegar fotogramas con el slider y botones.
3. Se puede hacer clic en un bbox para seleccionarlo.
4. Se puede dibujar un rectángulo manualmente.
5. La anotación rápida funciona y pre-selecciona a la persona más grande.
6. Los AnchorFrames se guardan y cargan correctamente desde JSON.
7. NO implementes análisis de escena, multipass ni anonimización todavía.
