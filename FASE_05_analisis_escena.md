# FASE 5 — ANÁLISIS DE ESCENA CON INTELIGENCIA ARTIFICIAL

## Contexto

Estamos construyendo "video_anonymizer_pro". Las fases 0-4 están implementadas. Ahora implementa el análisis de escena que mejora la detección con comprensión contextual.

## Tarea 5A: Implementar scene_analysis/shot_detector.py

```python
from dataclasses import dataclass

@dataclass
class ShotBoundary:
    frame_number: int
    type: str              # "cut" (corte duro) o "transition" (transición gradual)
    confidence: float
    histogram_diff: float  # magnitud de la diferencia


class ShotDetector:
    """
    Detecta cambios de plano/escena en el vídeo.
    
    IMPORTANTE: Tras un cambio de plano las posiciones de las personas cambian
    drásticamente. El tracker debe reinicializarse parcialmente y la
    re-identificación debe activarse.
    
    Métodos de detección:
    1. Diferencia de histograma (principal):
       - Calcula histograma HSV de cada frame.
       - Compara con el frame anterior usando correlación de Bhattacharyya.
       - Si la diferencia > threshold → corte detectado.
    
    2. Análisis de bordes (para transiciones graduales):
       - Calcula Canny edges en cada frame.
       - Si la diferencia de edge density cambia gradualmente durante N frames
         → transición detectada.
    
    3. Threshold adaptativo:
       - Calcula estadísticas de diferencias para todo el vídeo.
       - El threshold = media + K * desviación_estándar.
       - K es ajustable con sensitivity (0.5 = muy sensible, 1.5 = poco sensible).
    """

    def __init__(self, sensitivity: float = 0.7):
        """sensitivity: 0.0 (detecta todo) a 1.0 (solo cambios drásticos)."""

    def detect_shots(self, video_path: str, progress_callback=None) -> list[ShotBoundary]:
        """
        Analiza el vídeo completo y devuelve lista de cambios de plano.
        Esto se ejecuta antes del procesamiento principal.
        Muestra progreso si hay callback.
        
        Proceso:
        1. Primera pasada rápida: calcula histograma HSV de cada frame.
        2. Calcula diferencias entre frames consecutivos.
        3. Calcula threshold adaptativo.
        4. Identifica picos como cambios de plano.
        5. Clasifica cada cambio como "cut" o "transition".
        """

    def is_shot_change(self, frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[bool, float]:
        """Compara dos frames y determina si hay cambio. Devuelve (es_cambio, diferencia)."""
```

## Tarea 5B: Implementar scene_analysis/motion_estimator.py

```python
@dataclass
class MotionVector:
    track_id: int
    dx: float              # desplazamiento horizontal por frame
    dy: float              # desplazamiento vertical por frame
    speed: float           # velocidad total en píxeles/frame
    direction: float       # ángulo en radianes
    predicted_bbox: tuple[int, int, int, int]  # bbox predicho para el siguiente frame


class MotionEstimator:
    """
    Estima movimiento y predice posiciones futuras de las personas rastreadas.
    
    USOS:
    - Interpolar posición cuando hay frames sin detección.
    - Predecir dónde reaparecerá la mujer al entrar por el borde.
    - Suavizar la posición del blur (evitar "saltos").
    
    Métodos:
    1. Historial del tracker (Kalman filter de Deep SORT):
       - Usa las posiciones históricas del track.
       - Calcula velocidad media de los últimos N frames.
    
    2. Flujo óptico (para mayor precisión):
       - Calcula optical flow con cv2.calcOpticalFlowFarneback entre frames.
       - Extrae el flow medio en la región del bbox de persona.
       - Más preciso pero más lento que el historial simple.
    """

    def __init__(self, use_optical_flow: bool = False):
        """use_optical_flow: True para mayor precisión (más lento)."""

    def estimate_motion(self, track_history: list, current_bbox: tuple) -> MotionVector:
        """
        Estima movimiento basándose en el historial de posiciones.
        Calcula velocidad y dirección con media ponderada (más peso a frames recientes).
        """

    def predict_next_position(self, motion: MotionVector, steps: int = 1) -> tuple[int, int, int, int]:
        """
        Predice la posición del bbox dentro de N frames.
        Aplica el vector de movimiento N veces.
        Clampea a los límites del frame.
        """

    def smooth_positions(self, bbox_history: list[tuple], window: int = 5) -> list[tuple]:
        """
        Suaviza una secuencia de posiciones con media móvil ponderada.
        Peso exponencial decreciente hacia el pasado.
        Esto se usa para evitar jitter en el blur.
        """

    def estimate_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
                               bbox: tuple) -> tuple[float, float]:
        """Calcula flow medio dentro del bbox. Devuelve (dx, dy)."""
```

## Tarea 5C: Implementar scene_analysis/scene_interpreter.py

```python
@dataclass
class SceneContext:
    scene_type: str            # "both_visible", "woman_alone", "child_alone",
                               # "woman_entering", "woman_exiting", "close_up",
                               # "wide_shot", "occlusion", "empty"
    num_persons_visible: int
    woman_visible: bool
    woman_confidence: float
    child_visible: bool
    estimated_distance: str    # "close", "medium", "far"
    occlusion_detected: bool
    frame_region_hint: Optional[str] = None   # "left", "right", "center", etc.
    vlm_description: Optional[str] = None     # descripción del VLM si se usó


class SceneInterpreter:
    """
    Analiza la escena para mejorar la detección y tracking.
    
    NIVEL 1 — Análisis heurístico (siempre activo):
    - Evalúa cuántas personas hay, su posición, tamaño relativo.
    - Detecta oclusiones (overlap de bboxes > 30%).
    - Detecta si alguien está parcialmente fuera del encuadre (bbox tocando borde).
    - Clasifica el tipo de plano (close-up, medio, wide) según tamaño relativo.
    
    NIVEL 2 — Vision-Language Model (opcional, activable):
    - Usa Florence-2 (microsoft/Florence-2-base) de HuggingFace.
    - Florence-2 es open-source, ligero (~0.7GB) y se ejecuta localmente.
    - Capacidades: caption, grounding, detection.
    - Se invoca SOLO cuando:
      a) Se pierde el track de la mujer.
      b) La confianza de detección baja de un umbral.
      c) Hay un cambio de plano.
      d) Cada N segundos como verificación.
    - NO se ejecuta en cada frame (sería demasiado lento).
    
    Prompt para Florence-2:
    - Usa la tarea "<CAPTION_TO_PHRASE_GROUNDING>" con el texto:
      "adult woman" para localizar a la mujer.
    - O usa "<OD>" (Object Detection) para obtener bboxes de todas las personas.
    """

    def __init__(self, use_vlm: bool = False, device: str = "cuda",
                 vlm_confidence_threshold: float = 0.4):
        """
        Inicializa.
        Si use_vlm=True, carga Florence-2 desde HuggingFace (transformers).
        """

    def analyze_frame(self, frame: np.ndarray, detections: list,
                      tracks: list, frame_shape: tuple) -> SceneContext:
        """
        Análisis heurístico principal (Nivel 1).
        
        Lógica:
        1. Cuenta personas visibles.
        2. Identifica cuáles son mujer/niño basándose en tracks+clasificación.
        3. Calcula distancia estimada según tamaño relativo al frame.
        4. Detecta oclusiones entre bboxes.
        5. Detecta entradas/salidas (bbox tocando borde del frame).
        6. Clasifica tipo de escena.
        """

    def invoke_vlm(self, frame: np.ndarray) -> dict:
        """
        Invoca Florence-2 para análisis avanzado.
        
        Tareas ejecutadas:
        1. "<OD>": detección de objetos → lista de bboxes con labels.
        2. "<CAPTION>": descripción general de la escena.
        3. "<CAPTION_TO_PHRASE_GROUNDING>" con "adult woman": localiza a la mujer.
        
        Devuelve dict con: detections, caption, grounding_results.
        """

    def should_invoke_vlm(self, confidence_history: list[float],
                           woman_lost: bool, shot_change: bool) -> bool:
        """
        Decide si invocar el VLM en este frame.
        True si:
        - La confianza media de los últimos 10 frames < threshold.
        - La mujer lleva perdida más de 5 frames.
        - Acaba de haber un cambio de plano.
        """
```

## Verificación

1. ShotDetector detecta cortes de escena correctamente en un vídeo de prueba.
2. MotionEstimator predice posiciones razonables a 1-3 frames de distancia.
3. SceneInterpreter clasifica tipos de escena correctamente.
4. Si use_vlm=True, Florence-2 se descarga y ejecuta correctamente.
5. NO implementes multipass ni anonimización todavía.
