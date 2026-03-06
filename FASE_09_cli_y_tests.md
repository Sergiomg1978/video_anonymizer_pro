# FASE 9 — CLI PRINCIPAL Y TESTING

## Contexto

Estamos construyendo "video_anonymizer_pro". Las fases 0-8 están implementadas. Ahora implementa el punto de entrada CLI y la suite de tests.

## Tarea 9A: Implementar main.py

```python
"""
Video Anonymizer Pro — CLI principal.

Uso básico:
    python main.py input_video.mp4 output_video.mp4

Uso completo:
    python main.py input_video.mp4 output_video.mp4 \
        --mode full \
        --blur gaussian \
        --quality lossless \
        --device cuda \
        --temporal-smoothing 5 \
        --confidence-threshold 0.4 \
        --use-sam \
        --use-vlm \
        --annotate \
        --debug \
        --debug-video debug_output.mp4 \
        --report report.json \
        --log-level INFO
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Argumentos de argparse:
#
# Posicionales:
#   input               Ruta al vídeo de entrada
#   output              Ruta al vídeo de salida
#
# Opcionales:
#   --mode              Modo de ejecución: full, auto, quick, debug (default: auto)
#   --blur              Tipo de blur: gaussian, pixelate, solid, mosaic (default: gaussian)
#   --quality           Calidad de salida: lossless, high, medium (default: lossless)
#   --device            Dispositivo: cuda, cpu, auto (default: auto)
#   --temporal-smoothing  Frames de suavizado temporal (default: 5)
#   --confidence-threshold  Umbral mínimo de confianza (default: 0.4)
#   --use-sam           Activar SAM 2 para segmentación precisa
#   --use-vlm           Activar Florence-2 para análisis de escena
#   --annotate          Lanzar GUI de anotación manual antes de procesar
#   --annotation-file   Cargar anotaciones previas desde archivo JSON
#   --debug             Activar modo debug con logs detallados
#   --debug-video       Ruta para vídeo de debug con visualizaciones
#   --report            Ruta para guardar reporte JSON
#   --log-level         Nivel de logging: DEBUG, INFO, WARNING, ERROR (default: INFO)
#   --batch             Procesar múltiples vídeos (acepta glob pattern)
#   --version           Mostrar versión y salir
#
# FLUJO del main:
#
# 1. Parsear argumentos.
# 2. Mostrar banner con Rich:
#    ╔══════════════════════════════════════╗
#    ║     VIDEO ANONYMIZER PRO v1.0.0     ║
#    ╚══════════════════════════════════════╝
#
# 3. Validar que el input existe y es un formato de vídeo soportado.
# 4. Validar que la carpeta del output existe.
# 5. Verificar que FFmpeg está instalado y en el PATH.
#    Si no: error con instrucciones de instalación.
#
# 6. Construir PipelineConfig desde los argumentos.
# 7. Mostrar tabla Rich con la configuración:
#    ┌─────────────────┬───────────────┐
#    │ Parámetro       │ Valor         │
#    ├─────────────────┼───────────────┤
#    │ Modo            │ full          │
#    │ Blur            │ gaussian      │
#    │ Calidad         │ lossless      │
#    │ Dispositivo     │ CUDA (RTX...) │
#    │ SAM             │ Activado      │
#    │ VLM             │ Desactivado   │
#    │ Doble pasada    │ Sí            │
#    └─────────────────┴───────────────┘
#
# 8. Crear AnonymizationPipeline(config).
# 9. Ejecutar pipeline.run(input, output, mode).
# 10. Mostrar reporte final con Rich panel:
#     ┌─────────────── RESULTADO ───────────────┐
#     │ Frames procesados: 4500                  │
#     │ Frames con blur:   3200 (71.1%)          │
#     │ Frames interpolados: 45                  │
#     │ Huecos sin resolver: 2                   │
#     │ Confianza media: 0.87                    │
#     │ Calidad (PSNR): 52.3 dB                 │
#     │ Tiempo total: 4m 23s                     │
#     │ Archivo de salida: output_video.mp4      │
#     └──────────────────────────────────────────┘
#
# 11. Si --batch: iterar sobre todos los vídeos que coincidan con el pattern.
#
# Manejo de errores:
# - try/except global con mensajes claros.
# - Si falla un modelo: sugerir instalación de dependencias.
# - Si falta VRAM: sugerir modo "quick" o "--device cpu".
# - Ctrl+C: cierre limpio, mostrar progreso parcial.


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        prog="video_anonymizer_pro",
        description="Anonimiza automáticamente el rostro de una mujer adulta en vídeos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  Modo rápido (una pasada, sin SAM):
    python main.py video.mp4 output.mp4 --mode quick
  
  Modo completo con anotación manual:
    python main.py video.mp4 output.mp4 --mode full --annotate --use-sam
  
  Procesar carpeta de vídeos:
    python main.py "videos/*.mp4" outputs/ --batch --mode auto
        """
    )
    # ... definir todos los argumentos ...
    return parser.parse_args()


def show_banner(console):
    """Muestra banner del programa."""


def show_config_table(console, config):
    """Muestra tabla con la configuración."""


def show_report(console, report):
    """Muestra panel final con resultados."""


def main():
    """Punto de entrada principal."""
    args = parse_args()
    console = Console()
    
    show_banner(console)
    
    # ... flujo completo ...


if __name__ == "__main__":
    main()
```

## Tarea 9B: Implementar tests/

### tests/test_detection.py

```python
"""
Tests para el sistema de detección.

Tests a implementar:

1. test_face_detector_loads:
   - Verifica que MultiFaceDetector se instancia sin errores.
   
2. test_face_detector_finds_faces:
   - Crea una imagen de prueba con cv2 (dibuja un óvalo como "cara").
   - O mejor: descarga una imagen de prueba con cara real (usar una foto de dominio público).
   - Ejecuta detect() y verifica que devuelve al menos 1 detección.
   
3. test_head_detector_fallback:
   - Sin cara visible, verifica que HeadDetector estima desde bbox de persona.
   
4. test_person_detector_finds_persons:
   - Verifica detección de personas en imagen de prueba.
   
5. test_classifier_adult_vs_child:
   - Con dos detecciones de diferente tamaño, verifica clasificación.
   
6. test_nms_fusion:
   - Crea detecciones simuladas de 3 modelos.
   - Verifica que la fusión NMS produce resultado coherente.
"""
```

### tests/test_tracking.py

```python
"""
Tests para el sistema de tracking.

Tests a implementar:

1. test_tracker_stable_ids:
   - Simula 10 frames con un bbox que se mueve linealmente.
   - Verifica que el track_id es el mismo en todos los frames.
   
2. test_tracker_two_persons:
   - Simula dos personas moviéndose en paralelo.
   - Verifica que cada una mantiene su propio track_id.
   
3. test_identity_manager_register:
   - Registra dos identidades.
   - Verifica que cada una tiene un identity_id diferente.
   
4. test_identity_manager_match:
   - Registra una identidad con embedding facial.
   - Busca con un embedding similar.
   - Verifica que encuentra la identidad correcta.
   
5. test_reidentification_after_exit:
   - Simula secuencia: persona visible 10 frames, ausente 5, visible 10.
   - Verifica que la re-identificación asigna la misma identidad.
"""
```

### tests/test_pipeline.py

```python
"""
Tests de integración del pipeline.

Tests a implementar:

1. test_pipeline_creates_output:
   - Genera vídeo sintético de prueba (10 frames, 640x480, 30fps).
     El vídeo tiene dos rectángulos de colores como "personas".
   - Ejecuta pipeline en modo "quick".
   - Verifica que el archivo de salida existe.
   
2. test_output_metadata_matches:
   - Verifica que resolución, FPS y duración del output coinciden con el input.
   
3. test_pipeline_modes:
   - Verifica que los modos "quick", "auto" no generan errores.
   
4. test_pipeline_report:
   - Verifica que el pipeline genera un PipelineReport con campos válidos.
"""
```

### tests/test_quality.py

```python
"""
Tests de calidad de vídeo.

Tests a implementar:

1. test_codec_manager_params:
   - Verifica que CodecManager genera parámetros correctos para h264 y h265.
   
2. test_lossless_encoding:
   - Crea un frame aleatorio (numpy random).
   - Escribe con VideoWriter en modo lossless.
   - Lee de vuelta con VideoReader.
   - Verifica PSNR > 50 dB y SSIM > 0.99.
   
3. test_frame_dimensions_preserved:
   - Frame de entrada 1920x1080.
   - Después de write+read: sigue siendo 1920x1080.
   
4. test_fps_preserved:
   - Crea vídeo con FPS = 29.97 (30000/1001).
   - Lee de vuelta y verifica FPS ≈ 29.97.
   
5. test_no_quality_loss_outside_blur:
   - Simula un frame con blur solo en una región.
   - Compara los píxeles FUERA de la región con el original.
   - Deben ser idénticos (o PSNR > 50dB en modo lossless).
"""
```

## Verificación final

1. `python main.py --help` muestra la ayuda correctamente.
2. `python main.py --version` muestra la versión.
3. `python main.py test_video.mp4 output.mp4 --mode quick` ejecuta sin errores.
4. `pytest tests/ -v` pasa todos los tests.
5. El vídeo de salida se reproduce correctamente en cualquier reproductor.
6. La calidad visual del vídeo (fuera del blur) es idéntica al original.

## Tabla de referencia de modelos

Para que Gemini sepa qué descargar, aquí están los modelos y su procedencia:

| Componente | Modelo | Paquete/URL |
|---|---|---|
| Detección facial | YOLOv8n-face | ultralytics (buscar modelo face) |
| Detección facial | RetinaFace | insightface pip (buffalo_l) |
| Detección facial | MediaPipe | mediapipe pip (model_selection=1) |
| Detección cabeza | YOLO-Head | github.com/deepakcrk/yolov5-crowdhuman |
| Detección persona | YOLOv8x | ultralytics (modelo COCO) |
| Pose estimation | YOLOv8x-pose | ultralytics |
| Tracking | Deep SORT | deep-sort-realtime pip |
| Segmentación | SAM 2 | segment-anything-2 pip |
| Vision-Language | Florence-2 | HuggingFace: microsoft/Florence-2-base |
| Vídeo I/O | FFmpeg | ffmpeg-python pip + ffmpeg en PATH |
