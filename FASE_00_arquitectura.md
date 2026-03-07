# FASE 0 — ARQUITECTURA DEL PROYECTO Y ENTORNO

## Contexto del proyecto

Estoy construyendo un programa Python profesional llamado "video_anonymizer_pro" para anonimizar automáticamente el rostro/cabeza de una mujer adulta en vídeos donde aparece junto a un niño. El niño NO debe ser anonimizado. El programa debe preservar la calidad original del vídeo al 100%.

## Tarea: Crear la estructura de carpetas, dependencias y configuración

Crea SOLO estos archivos iniciales (NO implementes la lógica aún, solo la estructura):

### 1. Estructura de carpetas

```
video_anonymizer_pro/
├── main.py                          # Punto de entrada CLI (vacío, solo docstring por ahora)
├── config.py                        # Configuración global y constantes
├── requirements.txt                 # Dependencias
├── README.md                        # Documentación
├── setup.py                         # Instalación del paquete
│
├── core/
│   ├── __init__.py
│   ├── pipeline.py                  # Orquestador principal
│   ├── video_io.py                  # Lectura/escritura de vídeo sin pérdida
│   └── frame_extractor.py           # Extracción de fotogramas
│
├── detection/
│   ├── __init__.py
│   ├── face_detector.py             # Detección facial multi-modelo
│   ├── head_detector.py             # Detección de cabeza completa
│   ├── person_detector.py           # Detección de personas (YOLO/COCO)
│   └── age_gender_classifier.py     # Clasificador adulto vs niño
│
├── tracking/
│   ├── __init__.py
│   ├── deep_sort_tracker.py         # Tracking con Deep SORT
│   ├── identity_manager.py          # Gestión de identidades persistentes
│   └── reidentification.py          # Re-identificación tras salir/entrar del encuadre
│
├── annotation/
│   ├── __init__.py
│   ├── manual_annotator.py          # GUI para anotación manual
│   └── anchor_frames.py             # Gestión de fotogramas ancla
│
├── scene_analysis/
│   ├── __init__.py
│   ├── scene_interpreter.py         # Análisis de escena con IA
│   ├── shot_detector.py             # Detección de cambios de plano
│   └── motion_estimator.py          # Estimación de movimiento
│
├── anonymization/
│   ├── __init__.py
│   ├── blur_engine.py               # Motor de difuminado
│   ├── mask_generator.py            # Generación de máscaras con SAM 2
│   └── inpainting_engine.py         # Inpainting opcional
│
├── multipass/
│   ├── __init__.py
│   ├── forward_pass.py              # Procesamiento hacia adelante
│   ├── backward_pass.py             # Procesamiento hacia atrás
│   ├── confidence_merger.py         # Fusión de resultados
│   └── gap_filler.py                # Relleno de huecos
│
├── quality/
│   ├── __init__.py
│   ├── codec_manager.py             # Gestión de códecs
│   ├── frame_validator.py           # Validación de calidad
│   └── metadata_preserver.py        # Preservación de metadatos
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                    # Logging profesional
│   ├── gpu_manager.py               # Gestión de GPU/CUDA
│   ├── progress_tracker.py          # Barra de progreso
│   └── visualization.py             # Visualización de debug
│
└── tests/
    ├── __init__.py
    ├── test_detection.py
    ├── test_tracking.py
    ├── test_pipeline.py
    └── test_quality.py
```

### 2. requirements.txt

```
ultralytics>=8.1.0
mediapipe>=0.10.9
insightface>=0.7.3
deep-sort-realtime>=1.3.2
segment-anything-2>=0.1.0
opencv-python-headless>=4.9.0
ffmpeg-python>=0.2.0
torch>=2.1.0
torchvision>=0.16.0
onnxruntime-gpu>=1.17.0
numpy>=1.24.0
scipy>=1.11.0
rich>=13.7.0
PyQt6>=6.6.0
pytest>=8.0.0
Pillow>=10.2.0
scikit-image>=0.22.0
```

### 3. config.py

Crea config.py con dataclasses para toda la configuración:

```python
"""Configuración global del Video Anonymizer Pro."""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

@dataclass
class DetectionConfig:
    face_confidence_threshold: float = 0.3
    head_confidence_threshold: float = 0.4
    person_confidence_threshold: float = 0.5
    nms_iou_threshold: float = 0.4
    use_mediapipe: bool = True
    use_retinaface: bool = True
    use_yolo_face: bool = True
    face_expansion_ratio: float = 0.5  # expandir bbox facial para cubrir cabeza

@dataclass
class TrackingConfig:
    max_age: int = 90          # frames sin ver antes de eliminar track
    n_init: int = 3            # frames para confirmar track
    max_cosine_distance: float = 0.3
    nn_budget: int = 150
    reid_threshold: float = 0.6

@dataclass
class AnonymizationConfig:
    blur_mode: str = "gaussian"  # gaussian, pixelate, solid, mosaic
    temporal_smoothing: int = 5
    mask_dilation_px: int = 5
    mask_feather_px: int = 3
    use_sam: bool = True

@dataclass
class QualityConfig:
    mode: str = "lossless"  # lossless, high, medium
    crf_lossless: int = 0
    crf_high: int = 4
    crf_medium: int = 18
    min_psnr: float = 50.0
    min_ssim: float = 0.99

@dataclass
class SceneAnalysisConfig:
    use_vlm: bool = False
    vlm_model: str = "microsoft/Florence-2-base"
    shot_detection_sensitivity: float = 0.7
    vlm_invoke_confidence_threshold: float = 0.4

@dataclass
class MultipassConfig:
    enabled: bool = True
    interpolation_max_gap: int = 10
    merge_iou_threshold: float = 0.5

@dataclass
class PipelineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    anonymization: AnonymizationConfig = field(default_factory=AnonymizationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    scene_analysis: SceneAnalysisConfig = field(default_factory=SceneAnalysisConfig)
    multipass: MultipassConfig = field(default_factory=MultipassConfig)
    device: str = "auto"           # auto, cuda, cpu
    use_manual_annotation: bool = True
    debug_output: bool = False
    log_level: str = "INFO"
    models_dir: Path = Path("./models")
    chunk_size: int = 1000         # frames por bloque para gestión de RAM

# Constantes globales
VERSION = "1.0.0"
APP_NAME = "Video Anonymizer Pro"
```

### 4. README.md

Genera un README.md profesional con:
- Nombre del proyecto y descripción.
- Requisitos del sistema (Python 3.10+, GPU NVIDIA recomendada, FFmpeg instalado).
- Instrucciones de instalación paso a paso.
- Ejemplos de uso (modo básico, modo completo con anotación manual, modo rápido).
- Descripción de la arquitectura del sistema.
- Licencia (MIT).

### 5. Archivos __init__.py

Cada __init__.py debe estar vacío o tener solo un docstring describiendo el módulo.

### 6. Archivos placeholder

Cada archivo .py dentro de los módulos (pipeline.py, video_io.py, etc.) debe tener SOLO:
- Un docstring descriptivo explicando su propósito.
- Los imports que va a necesitar (comentados con # TODO).
- Las clases/funciones principales definidas con `pass` como placeholder.

IMPORTANTE: Crea los archivos uno por uno. Empieza por requirements.txt, luego config.py, luego los __init__.py, y finalmente los placeholders.
