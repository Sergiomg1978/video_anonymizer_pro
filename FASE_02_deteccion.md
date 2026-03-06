# FASE 2 — DETECCIÓN MULTIFACÉTICA (CARA + CABEZA + PERSONA)

## Contexto

Estamos construyendo "video_anonymizer_pro". Las fases 0-1 ya están implementadas (estructura, video_io, codec_manager). Ahora implementa el sistema de detección.

## Tarea 2A: Implementar detection/face_detector.py

Crea un detector de caras multi-modelo que combine 3 detectores para máxima robustez.

```python
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]          # x1, y1, x2, y2
    confidence: float
    landmarks: Optional[np.ndarray] = None   # 5 puntos faciales si disponibles
    embedding: Optional[np.ndarray] = None   # 512-d vector de InsightFace
    source_models: list[str] = field(default_factory=list)  # qué modelos la detectaron


class MultiFaceDetector:
    """
    Fusiona resultados de 3 detectores faciales:
    
    1. YOLOv8-Face (ultralytics con modelo de detección facial)
    2. MediaPipe Face Detection (model_selection=1, largo alcance)
    3. RetinaFace via InsightFace (buffalo_l, proporciona embeddings)
    
    Fusión por NMS ponderado:
    - Ejecuta los 3 detectores.
    - Agrupa detecciones que se superponen (IoU > 0.4).
    - Pondera por confianza de cada modelo.
    - Si 2+ modelos detectan la misma cara, la confianza sube.
    """

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.3):
        """
        Inicializa los 3 modelos.
        - YOLOv8: usa ultralytics.YOLO con un modelo face (busca 'yolov8n-face.pt'
          o similar). Si no existe modelo face específico, usa el modelo general y
          entrena/descarga uno.
        - MediaPipe: usa mp.solutions.face_detection con model_selection=1.
        - InsightFace: usa insightface.app.FaceAnalysis(name='buffalo_l').
          Configura con ctx_id=0 para GPU.
        """

    def _detect_yolo(self, frame: np.ndarray) -> list[FaceDetection]:
        """Detecta con YOLOv8-Face. Convierte resultados a FaceDetection."""

    def _detect_mediapipe(self, frame: np.ndarray) -> list[FaceDetection]:
        """
        Detecta con MediaPipe.
        NOTA: MediaPipe espera RGB, no BGR. Convierte antes de detectar.
        Los bboxes de MediaPipe son relativos (0-1), conviértelos a píxeles.
        """

    def _detect_insightface(self, frame: np.ndarray) -> list[FaceDetection]:
        """
        Detecta con InsightFace.
        Extrae: bbox, landmarks (5 puntos), embedding (512-d).
        El embedding es CRÍTICO para re-identificación posterior.
        """

    def _weighted_nms(self, all_detections: list[FaceDetection], iou_threshold: float = 0.4) -> list[FaceDetection]:
        """
        Non-Maximum Suppression ponderado.
        1. Agrupa detecciones con IoU > threshold.
        2. Para cada grupo, calcula bbox promedio ponderado por confianza.
        3. La confianza final = max(confianzas) * (1 + 0.1 * (num_modelos - 1)).
        4. Preserva landmarks y embedding del modelo más confiable del grupo.
        5. Registra en source_models qué modelos contribuyeron.
        """

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        """
        Ejecuta los 3 detectores y fusiona resultados.
        Filtra por confidence_threshold al final.
        """

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[FaceDetection]]:
        """Versión batch para procesamiento eficiente de múltiples frames."""
```

## Tarea 2B: Implementar detection/head_detector.py

```python
@dataclass
class HeadDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    method: str  # "face_expanded", "head_model", "person_estimated"
    associated_person_id: Optional[int] = None


class HeadDetector:
    """
    Detecta la cabeza completa (no solo la cara) para cuando la persona
    está de espaldas, de perfil extremo, o la cara no es visible.
    
    Estrategia en cascada:
    1. Si hay detección facial → expande el bbox un 50% para cubrir toda la cabeza
       (cabello, orejas). Método: "face_expanded".
    2. Si no hay cara → usa modelo YOLO entrenado en cabezas.
       Descarga pesos pre-entrenados del dataset CrowdHuman.
       URL: https://github.com/deepakcrk/yolov5-crowdhuman
       O busca modelos YOLO-head en ultralytics hub.
       Método: "head_model".
    3. Si tampoco hay detección de cabeza → estima desde detección de persona.
       Usa YOLOv8-pose para keypoints. El keypoint 0 (nariz) indica la cabeza.
       Si no hay keypoints, toma el 12-15% superior del bbox de persona.
       Método: "person_estimated".
    """

    def __init__(self, device: str = "cuda"):
        """
        Carga:
        - Modelo YOLO-Head (CrowdHuman o similar).
        - YOLOv8x-pose para keypoints como fallback.
        """

    def _expand_face_to_head(self, face_bbox: tuple, frame_shape: tuple) -> tuple:
        """
        Expande bbox facial un 50% arriba (para cabello), 30% a los lados,
        20% abajo (para barbilla/cuello). Clampea a los límites del frame.
        """

    def _detect_head_model(self, frame: np.ndarray) -> list[HeadDetection]:
        """Detecta cabezas directamente con modelo YOLO-Head."""

    def _estimate_from_person(self, frame: np.ndarray, person_bbox: tuple, keypoints: np.ndarray = None) -> HeadDetection:
        """
        Estima posición de cabeza desde detección de persona.
        Si hay keypoints: usa punto 0 (nariz) como centro, radio proporcional.
        Si no: toma el 15% superior del bbox de persona.
        """

    def detect(self, frame: np.ndarray, face_detections: list = None, person_detections: list = None) -> list[HeadDetection]:
        """
        Detecta cabezas usando la cascada de estrategias.
        Prioridad: face_expanded > head_model > person_estimated.
        Asocia cada cabeza con su persona correspondiente si es posible.
        """
```

## Tarea 2C: Implementar detection/person_detector.py

```python
@dataclass
class PersonDetection:
    bbox: tuple[int, int, int, int]      # bounding box de persona completa
    confidence: float
    keypoints: Optional[np.ndarray] = None  # 17 keypoints COCO si disponibles
    person_id: Optional[int] = None


class PersonDetector:
    """
    Detecta personas completas usando YOLOv8x (modelo grande, máxima precisión).
    También extrae keypoints de pose con YOLOv8x-pose.
    """

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.5):
        """
        Carga:
        - YOLOv8x (modelo COCO, clase 0 = person).
        - YOLOv8x-pose (para keypoints del esqueleto, 17 puntos COCO).
        """

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """Detecta personas y extrae keypoints de pose."""

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[PersonDetection]]:
        """Versión batch."""
```

## Tarea 2D: Implementar detection/age_gender_classifier.py

```python
@dataclass
class PersonClassification:
    is_adult: bool
    is_female: bool
    estimated_age: Optional[float] = None
    confidence: float = 0.0
    method: str = ""  # "face_analysis", "body_proportions", "relative_size", "reidentification"


class AgeGenderClassifier:
    """
    Clasifica si una persona detectada es la mujer adulta (a anonimizar)
    o el niño (NO anonimizar).
    
    Usa 4 señales combinadas:
    
    1. ANÁLISIS FACIAL (si hay cara visible):
       - InsightFace estima edad y género.
       - Adulto si edad > 14, femenino si gender = 0.
    
    2. PROPORCIONES CORPORALES (si hay keypoints de pose):
       - Ratio cabeza/cuerpo: niños tienen cabezas más grandes relativas al cuerpo.
       - Usa distancia entre keypoints: head_size / shoulder_width.
       - Niños: ratio > 0.7. Adultos: ratio < 0.5.
    
    3. TAMAÑO RELATIVO:
       - Si hay 2 personas, la más alta es probablemente la adulta.
       - Compara alturas de bbox en píxeles.
       - Si una persona es >1.3x más alta que la otra, es la adulta.
    
    4. RE-IDENTIFICACIÓN (si hay embeddings de referencia):
       - Compara embedding facial con los de referencia guardados.
       - Similitud coseno > 0.5 = match con identidad conocida.
    
    Votación final ponderada:
    - Face analysis: peso 0.4 (muy fiable cuando disponible).
    - Body proportions: peso 0.2.
    - Relative size: peso 0.2.
    - Re-identification: peso 0.5 (el más fiable, override casi total).
    - Normaliza pesos según señales disponibles.
    """

    def __init__(self, device: str = "cuda"):
        """Carga el modelo InsightFace para análisis facial."""

    def classify(self, frame: np.ndarray, person_bbox: tuple,
                 face_detection=None, keypoints=None,
                 other_persons: list = None) -> PersonClassification:
        """Clasifica una persona. other_persons son las demás personas del frame."""

    def set_reference_embeddings(self, adult_embedding: np.ndarray, child_embedding: np.ndarray):
        """Establece embeddings de referencia desde anotación manual o auto-detección."""

    def classify_by_face(self, face_detection) -> Optional[PersonClassification]:
        """Clasifica usando análisis facial de InsightFace."""

    def classify_by_proportions(self, person_bbox: tuple, keypoints: np.ndarray) -> Optional[PersonClassification]:
        """Clasifica usando proporciones corporales."""

    def classify_by_relative_size(self, person_bbox: tuple, other_persons: list) -> Optional[PersonClassification]:
        """Clasifica comparando tamaños relativos."""

    def classify_by_reidentification(self, face_detection) -> Optional[PersonClassification]:
        """Clasifica por similitud con embeddings de referencia."""
```

## Verificación

Después de implementar los 4 archivos:
1. Comprueba que todos los modelos se descargan automáticamente al instanciar las clases.
2. Comprueba que MultiFaceDetector funciona con una imagen de prueba.
3. Comprueba que PersonDetector detecta personas y keypoints.
4. NO implementes tracking ni otros módulos todavía.
