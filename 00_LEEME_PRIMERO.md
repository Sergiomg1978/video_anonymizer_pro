# GUÍA DE USO — Prompts por Fases para VS Code

## Cómo usar estos archivos con Gemini Code Assist (o cualquier IA en VS Code)

### El problema
Si pegas el prompt completo de una vez, Gemini intenta crear todos los archivos simultáneamente y falla con "The code change cannot be automatically applied".

### La solución
Alimenta a Gemini **un archivo a la vez**, en orden. Espera a que termine cada fase antes de pasar a la siguiente.

---

## Orden de ejecución

| # | Archivo | Qué implementa | Archivos que crea |
|---|---------|----------------|-------------------|
| 0 | `FASE_00_arquitectura.md` | Estructura, config, requirements | ~30 archivos placeholder |
| 1 | `FASE_01_video_io.md` | Lectura/escritura de vídeo | video_io.py, codec_manager.py, frame_extractor.py |
| 2 | `FASE_02_deteccion.md` | Detectores de cara, cabeza, persona | face_detector.py, head_detector.py, person_detector.py, age_gender_classifier.py |
| 3 | `FASE_03_tracking.md` | Tracking y re-identificación | deep_sort_tracker.py, identity_manager.py, reidentification.py |
| 4 | `FASE_04_anotacion_manual.md` | GUI de anotación | manual_annotator.py, anchor_frames.py |
| 5 | `FASE_05_analisis_escena.md` | Análisis de escena con IA | shot_detector.py, motion_estimator.py, scene_interpreter.py |
| 6 | `FASE_06_anonimizacion.md` | Blur y segmentación SAM | mask_generator.py, blur_engine.py, inpainting_engine.py |
| 7 | `FASE_07_multipass.md` | Doble pasada forward/backward | forward_pass.py, backward_pass.py, confidence_merger.py, gap_filler.py |
| 8 | `FASE_08_pipeline.md` | Pipeline orquestador + utils | pipeline.py, logger.py, gpu_manager.py, progress_tracker.py, visualization.py |
| 9 | `FASE_09_cli_y_tests.md` | CLI y tests | main.py, test_*.py |

## Instrucciones paso a paso

1. Abre VS Code en la carpeta donde quieres crear el proyecto.
2. Abre el chat de Gemini Code Assist (o Cursor, Cline, etc.).
3. Copia el contenido completo de `FASE_00_arquitectura.md` y pégalo en el chat.
4. Espera a que Gemini cree todos los archivos de la Fase 0.
5. **Verifica** que la estructura de carpetas es correcta.
6. Copia el contenido de `FASE_01_video_io.md` y pégalo.
7. Repite para cada fase en orden.

### Tips importantes

- Si Gemini falla al aplicar un cambio, haz clic en "Show full code block" y cópialo manualmente al archivo.
- Usa el modo **Agent** (no Preview) para que Gemini cree archivos directamente.
- Si un prompt es demasiado largo para Gemini, puedes dividirlo: primero la Tarea A, luego la B, etc.
- Después de cada fase, ejecuta el proyecto para verificar que no hay errores de import.
- Si usas Cursor en vez de Gemini: funciona igual, pega cada fase como prompt.
