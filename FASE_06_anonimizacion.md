# FASE 6 — SEGMENTACIÓN PRECISA Y ANONIMIZACIÓN

## Contexto

Estamos construyendo "video_anonymizer_pro". Las fases 0-5 están implementadas. Ahora implementa la segmentación precisa de la cabeza y el motor de anonimización.

## Tarea 6A: Implementar anonymization/mask_generator.py

```python
class MaskGenerator:
    """
    Genera máscaras de segmentación precisas para la cabeza de la mujer.
    
    Usa SAM 2 (Segment Anything Model 2) de Meta para segmentación precisa.
    SAM 2 recibe un "prompt" (bbox o punto) y genera una máscara binaria
    exacta del objeto, mucho mejor que un simple rectángulo.
    
    MODOS:
    1. SAM 2 (preferido): Segmentación de alta calidad.
       - Usa sam2 o mobile_sam según disponibilidad de VRAM.
       - Recibe bbox de la cabeza como prompt.
       - Genera máscara que sigue el contorno exacto de la cabeza+cabello.
    
    2. SAM 2 Video Propagation (óptimo):
       - Segmenta la cabeza en un frame de referencia.
       - Propaga la máscara automáticamente a frames siguientes.
       - Mucho más eficiente y temporalmente consistente.
       - Re-inicializa tras cambios de plano.
    
    3. Fallback (sin SAM):
       - Usa el bbox expandido 15% como máscara rectangular.
       - Inscribe una elipse en el bbox para forma más natural.
       - Aplica feathering gaussiano en los bordes.
    """

    def __init__(self, model: str = "sam2", device: str = "cuda"):
        """
        Carga el modelo SAM.
        
        model="sam2": Carga SAM 2 (segment-anything-2).
          - Intenta cargar sam2_hiera_large.
          - Si no hay suficiente VRAM, carga sam2_hiera_small.
          - Si falla, intenta MobileSAM como último recurso.
        
        model="fallback": No carga ningún modelo, usa bbox+elipse.
        
        Configuración de SAM:
        - Crea SamPredictor para segmentación frame a frame.
        - Crea SamVideoPredictor para propagación de vídeo (si SAM 2 disponible).
        """

    def generate_mask(self, frame: np.ndarray, head_bbox: tuple) -> np.ndarray:
        """
        Genera máscara binaria de la cabeza.
        
        Con SAM:
        1. Alimenta la imagen a SAM con set_image().
        2. Usa head_bbox como box prompt.
        3. Opcionalmente añade punto central como point prompt adicional.
        4. SAM devuelve 3 máscaras candidatas con scores.
        5. Selecciona la de mayor score.
        6. Devuelve máscara binaria (uint8, 0 o 255) del tamaño del frame.
        
        Sin SAM:
        1. Crea máscara vacía del tamaño del frame.
        2. Dibuja elipse rellena inscrita en head_bbox expandido 15%.
        3. Devuelve la máscara.
        """

    def initialize_video_propagation(self, frame: np.ndarray, head_bbox: tuple, frame_idx: int):
        """
        Inicializa el modo de propagación de vídeo de SAM 2.
        Segmenta la cabeza en el frame dado y lo usa como referencia.
        Los siguientes frames pueden usar propagate_mask().
        """

    def propagate_mask(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Propaga la máscara al frame actual usando SAM 2 video mode.
        Si la confianza de propagación es baja, devuelve None.
        """

    def reset_propagation(self):
        """Reinicia la propagación (usar tras cambio de plano)."""

    def refine_mask(self, mask: np.ndarray, dilation_px: int = 5,
                    feather_px: int = 3) -> np.ndarray:
        """
        Refina la máscara:
        1. Dilatación morfológica para asegurar cobertura completa.
        2. Gaussian blur en los bordes para transición suave.
        3. Devuelve máscara float32 (0.0 a 1.0) con bordes suaves.
        
        La máscara refinada permite blend suave entre el frame original
        y el frame anonimizado, evitando bordes duros.
        """
```

## Tarea 6B: Implementar anonymization/blur_engine.py

```python
class BlurEngine:
    """
    Aplica anonimización sobre la máscara generada.
    
    MODOS:
    1. gaussian: Blur gaussiano intenso. Kernel adaptativo al tamaño de cabeza.
    2. pixelate: Reduce a bloques grandes y escala de vuelta (efecto mosaico).
    3. solid: Rellena con color sólido (negro o promedio de la región).
    4. mosaic: Similar a pixelado con patrón irregular.
    
    SUAVIZADO TEMPORAL:
    - Las posiciones de las máscaras se suavizan temporalmente para evitar jitter.
    - Media móvil ponderada de las últimas N posiciones.
    - Interpolación suave de las máscaras entre fotogramas.
    - CRÍTICO para que el blur no "salte" de un frame a otro.
    """

    def __init__(self, mode: str = "gaussian", temporal_smoothing: int = 5):
        """
        mode: tipo de blur.
        temporal_smoothing: número de frames para media móvil.
        """

    def anonymize_frame(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Aplica anonimización al frame completo usando la máscara.
        
        1. Genera la versión anonimizada de TODO el frame (o solo la región de interés).
        2. Usa la máscara float (0-1) para blend suave:
           output = frame * (1 - mask) + blurred_frame * mask
        3. Devuelve el frame con la cabeza anonimizada.
        
        IMPORTANTE: No modificar NADA fuera de la máscara.
        """

    def _apply_gaussian(self, region: np.ndarray, bbox_width: int) -> np.ndarray:
        """
        Blur gaussiano con kernel adaptativo.
        kernel_size = max(51, int(bbox_width * 0.8)) | 1 (asegurar impar).
        sigma = kernel_size / 3.
        """

    def _apply_pixelate(self, region: np.ndarray, bbox_width: int) -> np.ndarray:
        """
        Efecto pixelado:
        1. Calcula block_size = max(10, int(bbox_width / 6)).
        2. Reduce la región a (width/block_size, height/block_size) con INTER_LINEAR.
        3. Escala de vuelta al tamaño original con INTER_NEAREST.
        """

    def _apply_solid(self, region: np.ndarray) -> np.ndarray:
        """Rellena con color promedio de la región o negro."""

    def _apply_mosaic(self, region: np.ndarray, bbox_width: int) -> np.ndarray:
        """Pixelado con bloques de tamaño variable para patrón irregular."""

    def smooth_masks_temporal(self, masks: list[np.ndarray]) -> list[np.ndarray]:
        """
        Suaviza una secuencia de máscaras temporalmente.
        Usa media ponderada exponencial:
        smoothed[t] = alpha * mask[t] + (1-alpha) * smoothed[t-1]
        alpha = 2 / (temporal_smoothing + 1)
        """

    def set_mode(self, mode: str):
        """Cambia el modo de blur."""
```

## Tarea 6C (OPCIONAL): Implementar anonymization/inpainting_engine.py

```python
class InpaintingEngine:
    """
    OPCIONAL: Alternativa avanzada al blur.
    Reemplaza la cabeza con fondo generado usando inpainting.
    
    Más natural visualmente pero más lento.
    
    Opciones:
    1. OpenCV inpainting (rápido, calidad media):
       - cv2.inpaint con INPAINT_TELEA o INPAINT_NS.
    
    2. LaMa (lento, alta calidad):
       - Modelo de inpainting basado en deep learning.
       - Mejor para áreas grandes.
    """

    def __init__(self, method: str = "opencv", device: str = "cuda"):
        """Carga el modelo de inpainting."""

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Aplica inpainting en la región de la máscara."""
```

## Verificación

1. MaskGenerator con SAM genera máscaras que siguen el contorno real de la cabeza.
2. MaskGenerator fallback genera elipses suaves razonables.
3. BlurEngine produce blur convincente en los 4 modos.
4. El blend con máscara suavizada produce transiciones naturales sin bordes duros.
5. El suavizado temporal elimina el jitter al ver la secuencia animada.
6. NO implementes multipass ni pipeline todavía.
