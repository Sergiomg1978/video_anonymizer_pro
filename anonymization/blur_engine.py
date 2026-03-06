import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple

class BlurEngine:
    """
    Motor de difuminado para anonimización de vídeo.
    Aplica diferentes modos de blur sobre máscaras precisas con suavizado temporal.
    """

    def __init__(self, mode: str = "gaussian", temporal_smoothing: int = 5):
        """
        Inicializa el motor de blur.

        Args:
            mode: Modo de blur ('gaussian', 'pixelate', 'solid', 'mosaic')
            temporal_smoothing: Número de frames para suavizado temporal (default: 5)
        """
        self.mode = mode
        self.temporal_smoothing = temporal_smoothing
        self.mask_history = deque(maxlen=temporal_smoothing)

    def set_mode(self, mode: str):
        """
        Cambia el modo de blur.

        Args:
            mode: Nuevo modo ('gaussian', 'pixelate', 'solid', 'mosaic')
        """
        self.mode = mode

    def anonymize_frame(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Anonimiza un frame aplicando blur sobre la máscara con suavizado temporal.

        Args:
            frame: Frame de vídeo (BGR, uint8)
            mask: Máscara float 0-1 del mismo tamaño que el frame

        Returns:
            Frame anonimizado
        """
        # Agregar máscara al historial para suavizado temporal
        self.mask_history.append(mask.copy())

        # Si no hay suficiente historial, usar máscara actual
        if len(self.mask_history) < self.temporal_smoothing:
            smoothed_mask = mask
        else:
            # Promedio ponderado de las últimas N máscaras
            weights = np.linspace(0.5, 1.0, self.temporal_smoothing)
            weights = weights / weights.sum()
            smoothed_mask = np.average(np.stack(self.mask_history), axis=0, weights=weights)

        # Aplicar blur y blending
        return self._apply_blur_and_blend(frame, smoothed_mask)

    def anonymize_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Anonimiza una región específica del frame.

        Args:
            frame: Frame de vídeo
            bbox: Bounding box (x1, y1, x2, y2)
            mask: Máscara opcional, si None se crea una máscara rectangular con feathering

        Returns:
            Frame anonimizado
        """
        if mask is None:
            mask = self._create_bbox_mask(frame.shape[:2], bbox)

        # Para esta versión, no aplicamos suavizado temporal en anonymize_region
        # El suavizado se maneja en anonymize_frame para secuencias completas
        return self._apply_blur_and_blend(frame, mask)

    def _create_bbox_mask(self, frame_shape: Tuple[int, int], bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crea una máscara rectangular con feathering gaussiano desde un bbox.

        Args:
            frame_shape: (height, width) del frame
            bbox: (x1, y1, x2, y2)

        Returns:
            Máscara float 0-1
        """
        mask = np.zeros(frame_shape, dtype=np.float32)
        x1, y1, x2, y2 = bbox

        # Región central sólida
        inner_margin = 5
        mask[max(0, y1+inner_margin):min(frame_shape[0], y2-inner_margin),
             max(0, x1+inner_margin):min(frame_shape[1], x2-inner_margin)] = 1.0

        # Feathering en los bordes
        feather_size = 10
        # Crear máscara con bordes suavizados usando distance transform
        temp_mask = np.zeros(frame_shape, dtype=np.uint8)
        temp_mask[y1:y2, x1:x2] = 255
        dist = cv2.distanceTransform(temp_mask, cv2.DIST_L2, 5)
        mask = np.clip(dist / feather_size, 0, 1).astype(np.float32)

        return mask

    def _apply_blur_and_blend(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Aplica el blur sobre la región enmascarada y hace el blending.

        Args:
            frame: Frame original
            mask: Máscara 0-1

        Returns:
            Frame con blur aplicado
        """
        # Encontrar el bounding box de la máscara
        coords = np.where(mask > 0.1)
        if len(coords[0]) == 0:
            return frame.copy()

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # Expandir un poco para evitar artefactos en bordes
        margin = 20
        y_min = max(0, y_min - margin)
        y_max = min(frame.shape[0], y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(frame.shape[1], x_max + margin)

        region = frame[y_min:y_max, x_min:x_max]
        mask_region = mask[y_min:y_max, x_min:x_max]

        # Aplicar el modo de blur seleccionado
        blurred_region = self._apply_blur_mode(region, mask_region)

        # Blending suave
        output = frame.copy()
        mask_3d = mask_region[:, :, np.newaxis]  # Para broadcasting con canales BGR
        output[y_min:y_max, x_min:x_max] = (
            region * (1 - mask_3d) + blurred_region * mask_3d
        ).astype(np.uint8)

        return output

    def _apply_blur_mode(self, region: np.ndarray, mask_region: np.ndarray) -> np.ndarray:
        """
        Aplica el modo de blur específico a la región.

        Args:
            region: Región del frame a anonimizar
            mask_region: Máscara correspondiente

        Returns:
            Región anonimizada
        """
        height, width = region.shape[:2]

        if self.mode == "gaussian":
            # Blur gaussiano adaptativo al tamaño de la cabeza
            head_width = width
            kernel_size = max(51, int(head_width * 0.8))
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma = kernel_size / 6.0  # Sigma proporcional al kernel
            return cv2.GaussianBlur(region, (kernel_size, kernel_size), sigma)

        elif self.mode == "pixelate":
            # Pixelado con bloques grandes
            head_width = width
            block_size = max(10, int(head_width / 6))
            # Downsampling
            small_h = max(1, height // block_size)
            small_w = max(1, width // block_size)
            small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            # Upsampling
            return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

        elif self.mode == "solid":
            # Relleno con color promedio de la región
            # Solo considerar píxeles donde la máscara es significativa
            valid_pixels = region[mask_region > 0.5]
            if len(valid_pixels) > 0:
                avg_color = np.mean(valid_pixels, axis=0)
            else:
                avg_color = np.mean(region, axis=(0, 1))
            return np.full_like(region, avg_color, dtype=np.uint8)

        elif self.mode == "mosaic":
            # Mosaico similar a pixelado pero con patrón más irregular
            head_width = width
            block_size = max(8, int(head_width / 8))  # Bloques más pequeños para irregularidad

            # Crear patrón irregular dividiendo en bloques y aplicando offsets aleatorios
            mosaic = region.copy()
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    y_end = min(y + block_size, height)
                    x_end = min(x + block_size, width)
                    block = region[y:y_end, x:x_end]

                    if block.size > 0:
                        # Tomar el color promedio del bloque
                        avg_color = np.mean(block, axis=(0, 1))
                        mosaic[y:y_end, x:x_end] = avg_color

            return mosaic

        else:
            # Modo desconocido, devolver región original
            return region.copy()