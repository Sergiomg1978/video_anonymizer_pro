# FASE 8 — PIPELINE ORQUESTADOR PRINCIPAL

## Contexto

Estamos construyendo "video_anonymizer_pro". Las fases 0-7 están implementadas. Ahora implementa el pipeline que orquesta todo el flujo de principio a fin.

## Tarea: Implementar core/pipeline.py

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class PipelineProgress:
    current_step: str
    step_number: int
    total_steps: int
    current_frame: int
    total_frames: int
    fps_processing: float
    eta_seconds: float
    gpu_usage_percent: Optional[float] = None

@dataclass
class PipelineReport:
    total_frames: int
    frames_with_blur: int
    frames_interpolated: int
    frames_woman_absent: int
    frames_needing_review: int
    average_confidence: float
    processing_time_seconds: float
    quality_psnr_mean: float
    quality_ssim_mean: float
    input_metadata: dict
    output_metadata: dict
    gap_report: 'GapReport'


class AnonymizationPipeline:
    """
    Orquestador principal. Ejecuta todo el flujo de anonimización.
    
    MODOS:
    - "full": Anotación manual + doble pasada + SAM + VLM.
    - "auto": Sin anotación manual (auto-detecta en los primeros frames) + doble pasada + SAM.
    - "quick": Sin anotación + una sola pasada + sin SAM (bbox rectangular).
    - "debug": Como "full" pero genera vídeo extra con visualizaciones de debug.
    
    FLUJO COMPLETO (modo "full"):
    
    PASO 1 — Inicialización:
      a. Abrir vídeo con VideoReader y extraer metadatos.
      b. Verificar FFmpeg disponible.
      c. Detectar GPU y VRAM disponible con gpu_manager.
      d. Mostrar configuración con Rich console.
    
    PASO 2 — Pre-análisis del vídeo:
      a. Ejecutar ShotDetector para encontrar cambios de plano.
      b. Mostrar resumen: duración, resolución, fps, número de planos.
    
    PASO 3 — Carga de modelos de IA:
      a. Cargar todos los modelos según la configuración.
      b. Si hay poca VRAM, cargar solo los necesarios para el modo seleccionado.
      c. Mostrar barra de progreso durante la carga.
    
    PASO 4 — Anotación manual (si mode="full" o use_manual_annotation=True):
      a. Extraer fotogramas representativos con FrameExtractor.
      b. Lanzar GUI ManualAnnotator.
      c. Obtener AnchorFrames del usuario.
      d. Extraer embeddings de referencia de la mujer.
      e. Inicializar IdentityManager con embeddings.
      f. Si hay archivo de anotación previo (--annotation-file), cargarlo en su lugar.
    
    PASO 4-ALT — Auto-detección (si mode="auto" o "quick"):
      a. Tomar los primeros 30 frames del vídeo.
      b. Detectar personas y caras en cada uno.
      c. Asumir que la persona más grande/alta es la mujer adulta.
      d. Extraer embeddings y confirmar con clasificador de edad/género.
      e. Inicializar IdentityManager automáticamente.
    
    PASO 5 — Pasada Forward:
      a. Crear instancia de ForwardPass con todos los componentes.
      b. Procesar frame a frame con barra de progreso.
      c. Mostrar: ETA, FPS de procesamiento, frame actual/total, confianza media.
    
    PASO 6 — Pasada Backward (si multipass=True):
      a. Crear NUEVAS instancias de tracker e identity_manager.
      b. Pasar los embeddings de referencia de la mujer (del paso 4/5).
      c. Procesar en reversa con barra de progreso.
    
    PASO 7 — Fusión y relleno de huecos:
      a. ConfidenceMerger fusiona forward + backward.
      b. GapFiller rellena huecos cortos con interpolación.
      c. Generar GapReport.
      d. Mostrar resumen de huecos en consola.
    
    PASO 8 — Generación de máscaras:
      a. Para cada frame con detección, generar máscara con MaskGenerator.
      b. Si use_sam: usa SAM 2 con propagación de vídeo.
         - Inicializa propagación al inicio de cada plano.
         - Re-inicializa tras cambios de plano.
      c. Aplicar suavizado temporal a las máscaras.
    
    PASO 9 — Anonimización y escritura:
      a. Abrir VideoWriter con los metadatos del original.
      b. Para cada frame:
         - Leer frame original.
         - Si hay máscara: aplicar blur con BlurEngine.
         - Si no hay máscara: escribir frame sin modificar.
         - Escribir frame al VideoWriter.
      c. Cerrar VideoWriter.
    
    PASO 10 — Verificación y reporte:
      a. Verificar calidad: comparar PSNR/SSIM en frames SIN blur.
         - Estos frames no deberían tener ninguna degradación.
      b. Si debug=True: generar vídeo de debug con bboxes, IDs, confianzas visibles.
      c. Generar PipelineReport con todas las estadísticas.
      d. Si --report: guardar reporte en JSON.
      e. Mostrar resumen final con Rich panel.
    """

    def __init__(self, config: 'PipelineConfig'):
        """
        Recibe la configuración completa.
        NO carga modelos todavía (se cargan en run()).
        """

    def run(self, input_video: str, output_video: str, mode: str = "full"):
        """
        Ejecuta el pipeline completo.
        
        Usa Rich console para mostrar:
        - Banner con nombre, versión, configuración.
        - Tabla con metadatos del vídeo.
        - Barras de progreso para cada paso.
        - Panel final con estadísticas.
        """

    def _step_init(self, input_video: str):
        """PASO 1: Inicialización y verificación."""

    def _step_preanalysis(self):
        """PASO 2: Pre-análisis (shot detection)."""

    def _step_load_models(self, mode: str):
        """PASO 3: Carga de modelos."""

    def _step_manual_annotation(self):
        """PASO 4: Anotación manual."""

    def _step_auto_detect(self):
        """PASO 4-ALT: Auto-detección."""

    def _step_forward_pass(self) -> list:
        """PASO 5: Pasada forward."""

    def _step_backward_pass(self) -> list:
        """PASO 6: Pasada backward."""

    def _step_merge(self, forward_results, backward_results) -> list:
        """PASO 7: Fusión y relleno."""

    def _step_generate_masks(self, final_results) -> dict:
        """PASO 8: Generación de máscaras. Devuelve dict {frame_num: mask}."""

    def _step_anonymize(self, output_video: str, final_results, masks: dict):
        """PASO 9: Anonimización y escritura."""

    def _step_verify(self, input_video: str, output_video: str) -> PipelineReport:
        """PASO 10: Verificación y reporte."""

    def _generate_debug_video(self, output_path: str, final_results):
        """
        Genera vídeo de debug donde cada frame muestra:
        - Bboxes de personas (verde), caras (azul), cabezas (rojo).
        - ID de track y de identidad sobre cada persona.
        - Confianza de detección.
        - Tipo de escena en la esquina.
        - Indicador "TARGET" sobre la mujer objetivo.
        - Frame number y timestamp.
        """

    def get_progress(self) -> PipelineProgress:
        """Devuelve el progreso actual del pipeline."""

    def get_report(self) -> PipelineReport:
        """Devuelve el reporte final (disponible tras completar)."""
```

Implementa también utils/logger.py, utils/gpu_manager.py, utils/progress_tracker.py y utils/visualization.py:

### utils/logger.py
```python
# Logger profesional con Rich.
# Niveles: DEBUG, INFO, WARNING, ERROR.
# Formato: [timestamp] [LEVEL] [module] message
# Opción de escribir a archivo además de consola.
```

### utils/gpu_manager.py
```python
# Detecta GPU NVIDIA con torch.cuda.
# Reporta: nombre GPU, VRAM total, VRAM libre, CUDA version.
# Decide device automáticamente ("cuda" si disponible, "cpu" si no).
# Monitoriza uso de VRAM durante el procesamiento.
```

### utils/progress_tracker.py
```python
# Barra de progreso con Rich.
# Muestra: paso actual, frame actual/total, FPS, ETA, uso GPU.
# Actualizable en tiempo real.
```

### utils/visualization.py
```python
# Funciones para dibujar bboxes, labels, info sobre frames.
# Usado para el vídeo de debug.
# cv2.rectangle, cv2.putText con colores configurables.
```

## Verificación

1. El pipeline ejecuta todos los pasos en orden sin errores.
2. La barra de progreso se actualiza correctamente.
3. El vídeo de salida tiene la misma resolución y FPS que el original.
4. Los frames sin blur son idénticos al original.
5. El reporte final muestra estadísticas coherentes.
