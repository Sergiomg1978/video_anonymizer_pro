# FASE 1 — ENTRADA/SALIDA DE VÍDEO SIN PÉRDIDA DE CALIDAD

## Contexto

Estamos construyendo "video_anonymizer_pro". La estructura de carpetas y config.py ya están creados (Fase 0). Ahora implementa la lectura y escritura de vídeo.

## Tarea: Implementar core/video_io.py y quality/codec_manager.py

### core/video_io.py

Implementa dos clases: VideoReader y VideoWriter.

**CLASE VideoReader:**

```python
class VideoReader:
    """Lee vídeos usando FFmpeg para máxima compatibilidad y calidad."""

    def __init__(self, video_path: str):
        """
        Abre el vídeo y extrae todos los metadatos.
        Usa ffprobe (via ffmpeg-python) para obtener:
        - Resolución (width, height)
        - FPS exactos como fracción (ej: 30000/1001 para 29.97fps)
        - Códec (h264, hevc, etc.)
        - Bitrate
        - Pixel format (yuv420p, yuv444p, etc.)
        - Perfil de color
        - Número total de frames
        - Duración
        """

    def get_metadata(self) -> dict:
        """Devuelve todos los metadatos del vídeo."""

    def read_frame(self, frame_number: int) -> np.ndarray:
        """Acceso aleatorio a un frame específico. Devuelve BGR numpy array."""

    def iterate_frames(self) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        Itera todos los frames secuencialmente.
        Yield: (frame_number, frame_bgr)
        Usa un pipe de FFmpeg para máximo rendimiento.
        """

    def iterate_frames_reverse(self) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        Itera todos los frames en orden inverso (para backward pass).
        Estrategia: lee el vídeo en bloques de ~500 frames, invierte cada bloque.
        Esto es más eficiente que hacer seek aleatorio para cada frame.
        """

    def get_total_frames(self) -> int: ...
    def get_fps(self) -> Fraction: ...
    def get_resolution(self) -> tuple[int, int]: ...
    def close(self): ...
    def __enter__(self): ...
    def __exit__(self, *args): ...
```

**CLASE VideoWriter:**

```python
class VideoWriter:
    """Escribe vídeos preservando la calidad original usando FFmpeg."""

    def __init__(self, output_path: str, source_metadata: dict, quality_mode: str = "lossless"):
        """
        Configura el writer con los parámetros del vídeo original.
        
        quality_mode:
        - "lossless": CRF 0 con el mismo códec que el original. Sin pérdida alguna.
        - "high": CRF 1-4. Visualmente idéntico, archivo más pequeño.
        - "medium": CRF 18. Alta calidad pero con compresión notable.
        
        CRÍTICO: 
        - Usa los mismos FPS exactos que el original (como fracción).
        - Usa el mismo pixel format.
        - NO incluye audio (el audio se descarta).
        - Usa FFmpeg como subproceso con pipe de entrada.
        """

    def write_frame(self, frame: np.ndarray):
        """Escribe un frame al pipe de FFmpeg."""

    def finalize(self):
        """Cierra el pipe y espera a que FFmpeg termine."""

    def __enter__(self): ...
    def __exit__(self, *args): ...
```

**Implementación del pipe FFmpeg para escribir:**

El writer debe crear un subproceso FFmpeg con este comando (ejemplo para H.264 lossless):
```
ffmpeg -y -f rawvideo -vcodec rawvideo -s {width}x{height} -pix_fmt bgr24 
  -r {fps_fraction} -i pipe: -an -vcodec libx264 -crf 0 -preset veryslow 
  -pix_fmt yuv420p {output_path}
```

Para H.265:
```
ffmpeg -y -f rawvideo -vcodec rawvideo -s {width}x{height} -pix_fmt bgr24 
  -r {fps_fraction} -i pipe: -an -vcodec libx265 -crf 0 -preset veryslow 
  -pix_fmt yuv420p {output_path}
```

### quality/codec_manager.py

```python
class CodecManager:
    """Analiza el vídeo original y determina parámetros óptimos de codificación."""

    def __init__(self, source_metadata: dict):
        """Recibe los metadatos extraídos por VideoReader."""

    def get_encoding_params(self, quality_mode: str) -> dict:
        """
        Devuelve los parámetros de FFmpeg para la codificación.
        Retorna dict con: codec, crf, preset, pix_fmt, extra_params.
        
        Lógica:
        - Si original es H.264 → usa libx264
        - Si original es H.265/HEVC → usa libx265
        - Si original es otro códec → usa libx264 como fallback seguro
        - En modo lossless: CRF=0, preset=veryslow
        - En modo high: CRF=4, preset=slow
        - En modo medium: CRF=18, preset=medium
        """

    def verify_quality(self, original_frame: np.ndarray, encoded_frame: np.ndarray) -> dict:
        """
        Compara calidad entre frame original y codificado.
        Calcula y devuelve: PSNR, SSIM, diferencia máxima de píxel.
        Usa skimage.metrics.structural_similarity y peak_signal_noise_ratio.
        """
```

### core/frame_extractor.py

```python
class FrameExtractor:
    """Extrae fotogramas representativos del vídeo para anotación manual."""

    def __init__(self, video_reader: VideoReader):
        pass

    def extract_uniform(self, interval_seconds: float = 2.0) -> list[tuple[int, np.ndarray]]:
        """Extrae 1 frame cada N segundos. Devuelve lista de (frame_number, frame)."""

    def extract_at_shot_boundaries(self, shot_boundaries: list[int]) -> list[tuple[int, np.ndarray]]:
        """Extrae frames justo después de cada cambio de plano."""

    def extract_keyframes(self, num_frames: int = 20) -> list[tuple[int, np.ndarray]]:
        """
        Extrae N frames representativos combinando:
        - Frames uniformes a lo largo del vídeo.
        - Frames con máxima diferencia visual entre sí (diversidad).
        """

    def save_frames(self, frames: list[tuple[int, np.ndarray]], output_dir: str):
        """Guarda los frames extraídos como imágenes PNG sin compresión."""
```

### Verificación

Después de implementar, asegúrate de que:
1. FFmpeg está disponible en el PATH del sistema.
2. Un vídeo de prueba se puede abrir, leer frame por frame, y reescribir sin pérdida.
3. Los metadatos del vídeo de salida coinciden con los del original (resolución, FPS, códec).

NO implementes ningún otro módulo todavía. Solo video_io.py, codec_manager.py y frame_extractor.py.
