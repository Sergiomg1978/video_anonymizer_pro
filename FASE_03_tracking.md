# FASE 3 — TRACKING Y RE-IDENTIFICACIÓN

## Contexto

Estamos construyendo "video_anonymizer_pro". Las fases 0-2 ya están implementadas (estructura, video_io, detectores). Ahora implementa el sistema de tracking.

## Tarea 3A: Implementar tracking/deep_sort_tracker.py

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class TrackedPerson:
    track_id: int
    person_bbox: tuple[int, int, int, int]
    head_bbox: Optional[tuple[int, int, int, int]] = None
    face_bbox: Optional[tuple[int, int, int, int]] = None
    classification: Optional['PersonClassification'] = None
    is_target: bool = False          # True si es la mujer a anonimizar
    confidence: float = 0.0
    frames_since_seen: int = 0
    velocity: Optional[tuple[float, float]] = None  # dx, dy por frame

@dataclass
class TrackState:
    frame_number: int
    bbox: tuple[int, int, int, int]
    confidence: float
    is_confirmed: bool


class PersonTracker:
    """
    Tracking multi-objeto usando Deep SORT con features de apariencia.
    
    Configuración:
    - max_age = 90 frames (3 seg a 30fps) para tolerar oclusiones largas
      y desapariciones temporales del encuadre.
    - n_init = 3 (confirma tracks rápidamente).
    - max_cosine_distance = 0.3 (matching por apariencia estricto).
    - nn_budget = 150 (historial largo de apariencia).
    
    El tracker recibe bounding boxes de PERSONAS (no de caras), porque
    los bboxes de personas son más estables y grandes.
    
    Features de apariencia:
    - Extrae features con un modelo Re-ID (OSNet o similar).
    - Usa deep_sort_realtime como librería de tracking.
    """

    def __init__(self, config: 'TrackingConfig'):
        """
        Inicializa Deep SORT con los parámetros de config.
        Carga el modelo Re-ID para extracción de features.
        
        Usa deep_sort_realtime.deepsort_tracker.DeepSort con:
        - max_age=config.max_age
        - n_init=config.n_init  
        - max_cosine_distance=config.max_cosine_distance
        - nn_budget=config.nn_budget
        - embedder='mobilenet'  (o 'clip_RN50' para mejor calidad)
        """

    def update(self, frame: np.ndarray, detections: list) -> list[TrackedPerson]:
        """
        Actualiza el tracker con las detecciones del frame actual.
        
        1. Convierte detecciones a formato Deep SORT: ([x,y,w,h], confidence, feature).
        2. Llama a tracker.update_tracks().
        3. Convierte tracks confirmados a TrackedPerson.
        4. Calcula velocity basándose en posiciones anteriores.
        5. Devuelve lista de TrackedPerson con track_ids estables.
        """

    def get_track_history(self, track_id: int) -> list[TrackState]:
        """Devuelve historial completo de un track."""

    def is_track_active(self, track_id: int) -> bool:
        """True si el track tiene detecciones recientes."""

    def get_all_active_tracks(self) -> list[TrackedPerson]:
        """Devuelve todos los tracks activos (confirmados y no eliminados)."""

    def reset(self):
        """Reinicia el tracker por completo (para backward pass)."""
```

## Tarea 3B: Implementar tracking/identity_manager.py

```python
@dataclass
class PersonFeatures:
    """Todas las características disponibles de una persona."""
    face_embedding: Optional[np.ndarray] = None    # 512-d de InsightFace
    body_embedding: Optional[np.ndarray] = None    # Feature de Re-ID
    torso_color_hist: Optional[np.ndarray] = None  # Histograma de color del torso
    height_pixels: Optional[float] = None
    classification: Optional['PersonClassification'] = None
    last_bbox: Optional[tuple[int, int, int, int]] = None
    last_velocity: Optional[tuple[float, float]] = None

@dataclass
class LostIdentity:
    identity_id: int
    last_position: tuple[int, int, int, int]
    exit_side: str                    # "left", "right", "top", "bottom"
    frames_since_lost: int
    features: PersonFeatures


class IdentityManager:
    """
    Gestiona identidades persistentes a lo largo del vídeo.
    
    Diferencia entre track_id (del tracker, puede cambiar si se pierde y reaparece)
    e identity_id (persistente, sobrevive a la re-identificación).
    
    Para cada identidad almacena:
    - Banco de embeddings faciales (hasta 20, de diferentes ángulos).
    - Banco de embeddings corporales (hasta 10).
    - Histograma de color del torso (media de múltiples observaciones).
    - Historial de clasificaciones adulto/niño.
    - Historial de tamaños.
    - Última posición y velocidad.
    """

    def __init__(self):
        """Inicializa el registro vacío de identidades."""

    def register_identity(self, track_id: int, features: PersonFeatures) -> int:
        """
        Registra una nueva identidad. Devuelve identity_id único.
        Asocia el track_id actual con esta identidad.
        """

    def match_identity(self, features: PersonFeatures) -> tuple[int, float]:
        """
        Busca la identidad más parecida a las features dadas.
        
        Señales de matching (votación ponderada):
        1. Similitud coseno de embedding facial (peso 0.4, threshold 0.5).
        2. Similitud coseno de embedding corporal (peso 0.3, threshold 0.6).
        3. Similitud de histograma de color del torso (peso 0.15, Bhattacharyya).
        4. Coherencia de tamaño (peso 0.15).
        
        Devuelve (identity_id, confidence). Si confidence < 0.4, devuelve (-1, 0).
        """

    def update_identity(self, identity_id: int, features: PersonFeatures):
        """Actualiza los bancos de features de una identidad existente."""

    def get_target_identity(self) -> int:
        """Devuelve el identity_id de la mujer a anonimizar."""

    def set_target_identity(self, identity_id: int):
        """Marca una identidad como la mujer objetivo."""

    def mark_identity_lost(self, identity_id: int, last_position: tuple, exit_side: str):
        """Marca una identidad como perdida (salió del encuadre)."""

    def get_lost_identities(self) -> list[LostIdentity]:
        """Devuelve identidades perdidas que podrían reaparecer."""

    def _compute_torso_histogram(self, frame: np.ndarray, person_bbox: tuple, keypoints: np.ndarray = None) -> np.ndarray:
        """
        Extrae histograma de color HSV de la región del torso.
        Si hay keypoints: usa la zona entre hombros y caderas.
        Si no: usa el tercio medio del bbox de persona.
        """

    def _determine_exit_side(self, bbox: tuple, frame_shape: tuple) -> str:
        """Determina por qué lado del frame salió la persona."""
```

## Tarea 3C: Implementar tracking/reidentification.py

```python
class ReIdentifier:
    """
    Re-identifica personas cuando reaparecen en el encuadre.
    
    CONTEXTO: La mujer sale y entra del encuadre frecuentemente. Cuando vuelve,
    el tracker le asigna un nuevo track_id. Este módulo determina si esa nueva
    persona es la misma mujer de antes.
    
    Señales de re-identificación:
    1. Embedding facial (si hay cara visible al reaparecer).
    2. Embedding corporal (Re-ID features).
    3. Lado de entrada vs lado de salida (coherencia espacial).
    4. Tiempo desde la última vez vista (menos tiempo = más probable).
    5. Color de ropa (debería ser el mismo).
    6. Tamaño relativo (debería ser similar).
    """

    def __init__(self, identity_manager: 'IdentityManager'):
        """Recibe el IdentityManager para acceder a identidades perdidas."""

    def attempt_reidentification(self, new_track: TrackedPerson, entry_side: str,
                                  current_frame: int, frame: np.ndarray) -> Optional[int]:
        """
        Intenta re-identificar un nuevo track con una identidad perdida.
        
        1. Obtiene todas las identidades perdidas del IdentityManager.
        2. Para cada identidad perdida, calcula similitud multi-señal.
        3. Si la mejor coincidencia tiene confianza > 0.6, asigna esa identidad.
        4. Devuelve identity_id si hay match, None si no.
        
        Bonus de confianza:
        - +0.1 si entry_side coincide con exit_side de la identidad perdida.
        - +0.05 si el tiempo desde la pérdida es < 3 segundos.
        - -0.1 si han pasado > 30 segundos desde la pérdida.
        """

    def _determine_entry_side(self, bbox: tuple, frame_shape: tuple) -> str:
        """Determina por qué lado del frame entra la persona."""

    def check_new_tracks(self, frame: np.ndarray, active_tracks: list[TrackedPerson],
                          current_frame: int) -> dict[int, int]:
        """
        Revisa todos los tracks activos recientes y busca re-identificaciones.
        Devuelve dict: {track_id: identity_id} para los que se re-identificaron.
        """
```

## Verificación

1. Prueba que PersonTracker mantiene IDs estables cuando una persona se mueve.
2. Prueba que IdentityManager distingue correctamente dos personas.
3. Prueba que ReIdentifier recupera la identidad cuando simulamos una salida y re-entrada.
4. NO implementes análisis de escena ni anonimización todavía.
