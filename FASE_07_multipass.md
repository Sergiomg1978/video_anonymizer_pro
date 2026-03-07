# FASE 7 — PROCESAMIENTO MULTI-PASADA (FORWARD + BACKWARD)

## Contexto

Estamos construyendo "video_anonymizer_pro". Las fases 0-6 están implementadas. Ahora implementa el sistema de doble pasada para maximizar la detección.

## Concepto

El vídeo se procesa DOS veces: de principio a fin (forward) y de fin a principio (backward). Luego se fusionan los resultados. Esto resuelve:
- Cuando la mujer entra en escena, el forward puede tardar en identificarla, pero el backward ya la conoce de frames posteriores.
- Oclusiones temporales: la pasada sin oclusión tiene mejor información.
- Maximiza la cobertura: lo que una pasada pierde, la otra lo cubre.

## Tarea 7A: Implementar multipass/forward_pass.py

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class FrameResult:
    frame_number: int
    woman_detected: bool
    head_bbox: Optional[tuple[int, int, int, int]] = None
    confidence: float = 0.0
    detection_method: str = ""   # "face", "head", "person_est", "interpolated", "vlm"
    track_id: Optional[int] = None
    identity_id: Optional[int] = None
    scene_context: Optional[dict] = None


class ForwardPass:
    """
    Procesa el vídeo del frame 0 al frame N (orden natural).
    
    Para cada frame ejecuta:
    1. Detección de personas (PersonDetector).
    2. Detección de caras (MultiFaceDetector).
    3. Detección de cabezas (HeadDetector con resultados de cara y persona).
    4. Clasificación adulto/niño (AgeGenderClassifier).
    5. Tracking (PersonTracker.update).
    6. Re-identificación (si hay tracks nuevos y identidades perdidas).
    7. Análisis de escena (periódicamente o cuando confianza baja).
    8. Registro en log de resultados por frame.
    
    Usa los AnchorFrames como puntos de referencia:
    - En frames cercanos a un ancla, la confianza se ajusta al alza.
    - Los embeddings de las anclas inicializan el IdentityManager.
    """

    def __init__(self, detector_face, detector_head, detector_person,
                 classifier, tracker, identity_manager, scene_interpreter,
                 anchor_manager, shot_boundaries, config):
        """
        Recibe todos los componentes ya inicializados.
        shot_boundaries: lista de ShotBoundary del ShotDetector.
        """

    def process(self, video_reader, progress_callback=None) -> list[FrameResult]:
        """
        Procesa todos los frames en orden forward.
        
        Pseudocódigo:
        
        results = []
        for frame_num, frame in video_reader.iterate_frames():
            # 1. Comprobar cambio de plano
            if frame_num in shot_boundary_frames:
                self._handle_shot_change()
            
            # 2. Detectar personas
            persons = person_detector.detect(frame)
            
            # 3. Detectar caras
            faces = face_detector.detect(frame)
            
            # 4. Detectar cabezas
            heads = head_detector.detect(frame, faces, persons)
            
            # 5. Clasificar cada persona
            for person in persons:
                classification = classifier.classify(frame, person.bbox, ...)
            
            # 6. Actualizar tracker
            tracked = tracker.update(frame, persons)
            
            # 7. Re-identificar tracks nuevos
            reidentifier.check_new_tracks(frame, tracked, frame_num)
            
            # 8. Encontrar cabeza de la mujer objetivo
            target_head = self._find_target_head(tracked, heads, faces)
            
            # 9. Análisis de escena (cada N frames o si confianza baja)
            if should_analyze_scene:
                scene = scene_interpreter.analyze_frame(frame, ...)
            
            # 10. Registrar resultado
            results.append(FrameResult(
                frame_number=frame_num,
                woman_detected=target_head is not None,
                head_bbox=target_head,
                confidence=...,
                ...
            ))
            
            if progress_callback:
                progress_callback(frame_num)
        
        return results
        """

    def _find_target_head(self, tracked_persons, heads, faces) -> Optional[tuple]:
        """
        Encuentra el bbox de la cabeza de la mujer objetivo.
        1. Busca entre los tracked_persons el que tenga identity_id == target.
        2. Asocia con la mejor detección de cabeza/cara.
        3. Devuelve el bbox o None si no se encuentra.
        """

    def _handle_shot_change(self):
        """
        Maneja un cambio de plano:
        - No reinicia el tracker por completo (los tracks se pierden solos).
        - Activa re-identificación agresiva para los próximos frames.
        - Si use_vlm, invoca el VLM en el primer frame del nuevo plano.
        """
```

## Tarea 7B: Implementar multipass/backward_pass.py

```python
class BackwardPass:
    """
    Procesa el vídeo del frame N al frame 0 (orden inverso).
    
    Ejecuta EXACTAMENTE la misma lógica que ForwardPass, pero:
    - Lee frames en orden inverso (usa video_reader.iterate_frames_reverse).
    - Reinicializa el tracker y el identity manager.
    - PERO usa los mismos embeddings de referencia de la mujer
      (obtenidos de las anclas manuales o del forward pass).
    - Los shot_boundaries se aplican al revés.
    """

    def __init__(self, detector_face, detector_head, detector_person,
                 classifier, tracker, identity_manager, scene_interpreter,
                 anchor_manager, shot_boundaries, config,
                 reference_embeddings: dict = None):
        """
        reference_embeddings: embeddings faciales/corporales de la mujer,
        obtenidos de las anclas o del forward pass.
        El tracker y identity_manager deben ser NUEVAS INSTANCIAS.
        """

    def process(self, video_reader, progress_callback=None) -> list[FrameResult]:
        """
        Igual que ForwardPass.process() pero itera en reversa.
        Los resultados se devuelven en orden de frame_number (0 a N),
        no en orden de procesamiento.
        """
```

## Tarea 7C: Implementar multipass/confidence_merger.py

```python
@dataclass
class MergedFrameResult:
    frame_number: int
    woman_detected: bool
    head_bbox: Optional[tuple[int, int, int, int]] = None
    confidence: float = 0.0
    source: str = ""          # "forward", "backward", "merged", "none"
    is_gap: bool = False      # True si ninguna pasada la detectó


class ConfidenceMerger:
    """
    Fusiona resultados del forward pass y backward pass para cada frame.
    
    Reglas de fusión:
    1. Ambas detectan con confianza > 0.7 → usa la de mayor confianza.
    2. Solo una detecta → usa esa (con su confianza).
    3. Ambas detectan pero bboxes muy diferentes (IoU < 0.5):
       - Usa la de mayor confianza.
       - Si confianzas similares (diff < 0.1): promedia los bboxes ponderando.
    4. Ninguna detecta → marca como gap.
    """

    def __init__(self, iou_threshold: float = 0.5):
        pass

    def merge(self, forward_results: list[FrameResult],
              backward_results: list[FrameResult]) -> list[MergedFrameResult]:
        """
        Fusiona frame a frame. Ambas listas deben tener el mismo tamaño
        e ir indexadas por frame_number.
        """

    def _compute_iou(self, bbox1: tuple, bbox2: tuple) -> float:
        """Calcula Intersection over Union entre dos bboxes."""

    def _merge_bboxes(self, bbox1: tuple, conf1: float,
                       bbox2: tuple, conf2: float) -> tuple:
        """Promedia dos bboxes ponderando por confianza."""
```

## Tarea 7D: Implementar multipass/gap_filler.py

```python
@dataclass
class FinalFrameResult:
    frame_number: int
    woman_detected: bool
    head_bbox: Optional[tuple[int, int, int, int]] = None
    confidence: float = 0.0
    source: str = ""
    needs_review: bool = False   # True si el hueco es grande y no está clara la razón

@dataclass
class GapReport:
    total_frames: int
    frames_with_detection: int
    frames_interpolated: int
    frames_woman_absent: int       # se estima que salió del encuadre
    frames_needing_review: int
    gaps: list[dict]               # detalle de cada hueco


class GapFiller:
    """
    Rellena los huecos donde ninguna pasada detectó la cabeza.
    
    Tipos de hueco:
    1. Corto (< max_interpolation_gap frames):
       - Interpola linealmente bbox entre el último conocido antes y el primero después.
       - Interpola también el tamaño del bbox.
       - Marca como source="interpolated".
    
    2. Largo (>= max_interpolation_gap frames):
       - Verifica si la mujer salió del encuadre:
         Si el último bbox conocido estaba cerca del borde → source="woman_absent".
       - Si no está claro → marca como needs_review=True.
    """

    def __init__(self, max_interpolation_gap: int = 10, frame_shape: tuple = None):
        pass

    def fill_gaps(self, merged_results: list[MergedFrameResult]) -> list[FinalFrameResult]:
        """
        Procesa todos los merged_results y rellena huecos.
        Devuelve lista final con todos los frames resueltos.
        """

    def _interpolate_bbox(self, bbox_before: tuple, bbox_after: tuple,
                           t: float) -> tuple:
        """
        Interpola linealmente entre dos bboxes.
        t=0.0 → bbox_before, t=1.0 → bbox_after.
        Interpola x1, y1, x2, y2 independientemente.
        """

    def _is_near_edge(self, bbox: tuple, threshold_px: int = 50) -> tuple[bool, str]:
        """
        Determina si un bbox está cerca del borde del frame.
        Devuelve (True/False, lado: "left"/"right"/"top"/"bottom").
        """

    def get_gap_report(self) -> GapReport:
        """Genera reporte detallado de huecos y su resolución."""
```

## Verificación

1. ForwardPass procesa un vídeo de prueba sin errores.
2. BackwardPass procesa en reversa correctamente.
3. ConfidenceMerger fusiona resultados coherentemente.
4. GapFiller interpola huecos cortos y detecta salidas del encuadre.
5. El GapReport muestra estadísticas correctas.
6. NO implementes el pipeline final ni el CLI todavía.
