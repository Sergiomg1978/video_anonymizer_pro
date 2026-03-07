import numpy as np
import cv2
from typing import Optional

class InpaintingEngine:
    """
    Motor de inpainting para anonimización avanzada.
    Reemplaza la región anonimizada con contenido generado.
    """

    def __init__(self, method: str = "opencv"):
        """
        Inicializa el motor de inpainting.

        Args:
            method: Método de inpainting ('opencv', 'lama' - futuro)
        """
        self.method = method

    def inpaint_region(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Aplica inpainting a la región enmascarada.

        Args:
            frame: Frame original
            mask: Máscara binaria (0-1) de la región a inpaintar

        Returns:
            Frame con inpainting aplicado
        """
        if self.method == "opencv":
            return self._opencv_inpaint(frame, mask)
        else:
            # Fallback: devolver frame original
            return frame.copy()

    def _opencv_inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Usa OpenCV inpainting para rellenar la región.
        """
        # Convertir máscara a uint8
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Aplicar inpainting
        # INPAINT_TELEA es más rápido, INPAINT_NS es mejor calidad
        inpainted = cv2.inpaint(frame, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return inpainted

    def anonymize_frame(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Método de conveniencia para anonimizar un frame completo.
        """
        return self.inpaint_region(frame, mask)