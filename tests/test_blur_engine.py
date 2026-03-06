import numpy as np
import cv2
import pytest
from anonymization.blur_engine import BlurEngine

class TestBlurEngine:
    """Tests para el motor de difuminado BlurEngine."""

    @pytest.fixture
    def sample_frame(self):
        """Frame de prueba 100x100 con un cuadrado rojo en el centro."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[40:60, 40:60] = [0, 0, 255]  # Cuadrado rojo
        return frame

    @pytest.fixture
    def sample_mask(self):
        """Máscara circular en el centro."""
        mask = np.zeros((100, 100), dtype=np.float32)
        center = (50, 50)
        radius = 15
        y, x = np.ogrid[:100, :100]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        mask[dist_from_center <= radius] = 1.0
        # Suavizar bordes
        mask[dist_from_center <= radius] = np.clip((radius - dist_from_center) / 5, 0, 1)[dist_from_center <= radius]
        return mask

    def test_init(self):
        """Test inicialización con parámetros por defecto."""
        engine = BlurEngine()
        assert engine.mode == "gaussian"
        assert engine.temporal_smoothing == 5
        assert len(engine.mask_history) == 0

    def test_set_mode(self):
        """Test cambio de modo."""
        engine = BlurEngine()
        engine.set_mode("pixelate")
        assert engine.mode == "pixelate"

    @pytest.mark.parametrize("mode", ["gaussian", "pixelate", "solid", "mosaic"])
    def test_anonymize_frame_modes(self, sample_frame, sample_mask, mode):
        """Test anonimización con diferentes modos."""
        engine = BlurEngine(mode=mode, temporal_smoothing=1)  # Sin suavizado temporal para test
        result = engine.anonymize_frame(sample_frame, sample_mask)

        # Verificar que el resultado tiene el mismo tamaño
        assert result.shape == sample_frame.shape
        assert result.dtype == np.uint8

        # Verificar que hay cambios en la región enmascarada
        # (difícil verificar exactamente sin análisis visual, pero al menos que no sea idéntico)
        assert not np.array_equal(result, sample_frame)

    def test_anonymize_frame_temporal_smoothing(self, sample_frame, sample_mask):
        """Test suavizado temporal."""
        engine = BlurEngine(temporal_smoothing=3)

        # Primera llamada
        result1 = engine.anonymize_frame(sample_frame, sample_mask)
        assert len(engine.mask_history) == 1

        # Crear máscara ligeramente diferente
        mask2 = sample_mask.copy()
        mask2[45:55, 45:55] += 0.1  # Pequeño cambio
        mask2 = np.clip(mask2, 0, 1)

        result2 = engine.anonymize_frame(sample_frame, mask2)
        assert len(engine.mask_history) == 2

        # Tercera llamada
        result3 = engine.anonymize_frame(sample_frame, sample_mask)
        assert len(engine.mask_history) == 3

        # Verificar que los resultados son diferentes (debido al suavizado)
        assert not np.array_equal(result1, result2)
        assert not np.array_equal(result2, result3)

    def test_anonymize_region_with_mask(self, sample_frame):
        """Test anonimización de región con máscara proporcionada."""
        engine = BlurEngine(mode="gaussian")
        bbox = (30, 30, 70, 70)
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1.0

        result = engine.anonymize_region(sample_frame, bbox, mask)

        assert result.shape == sample_frame.shape
        assert not np.array_equal(result, sample_frame)

    def test_anonymize_region_without_mask(self, sample_frame):
        """Test anonimización de región creando máscara automáticamente."""
        engine = BlurEngine(mode="solid")
        bbox = (20, 20, 80, 80)

        result = engine.anonymize_region(sample_frame, bbox)

        assert result.shape == sample_frame.shape
        assert not np.array_equal(result, sample_frame)

    def test_gaussian_blur_mode(self, sample_frame, sample_mask):
        """Test específico del modo gaussian."""
        engine = BlurEngine(mode="gaussian", temporal_smoothing=1)

        # La región enmascarada debe estar borrosa
        result = engine.anonymize_frame(sample_frame, sample_mask)

        # Verificar que en la región central hay blur (colores mezclados)
        center_region = result[40:60, 40:60]
        # En un blur gaussiano, los colores deberían estar más uniformes
        std_before = np.std(sample_frame[40:60, 40:60])
        std_after = np.std(center_region)
        # El blur debería reducir la varianza de color
        assert std_after < std_before

    def test_pixelate_mode(self, sample_frame, sample_mask):
        """Test específico del modo pixelate."""
        engine = BlurEngine(mode="pixelate", temporal_smoothing=1)
        result = engine.anonymize_frame(sample_frame, sample_mask)

        # En modo pixelate, debería haber bloques de color uniforme
        center_region = result[40:60, 40:60]
        # Verificar que hay menos variación de color
        assert np.std(center_region) < np.std(sample_frame[40:60, 40:60])

    def test_solid_mode(self, sample_frame, sample_mask):
        """Test específico del modo solid."""
        engine = BlurEngine(mode="solid", temporal_smoothing=1)
        result = engine.anonymize_frame(sample_frame, sample_mask)

        # En modo solid, la región debería tener color uniforme
        center_region = result[45:55, 45:55]  # Región donde la máscara es 1.0
        # Verificar que todos los píxeles son casi iguales
        std_r = np.std(center_region[:, :, 0])
        std_g = np.std(center_region[:, :, 1])
        std_b = np.std(center_region[:, :, 2])
        assert std_r < 5 and std_g < 5 and std_b < 5  # Muy baja varianza

    def test_mosaic_mode(self, sample_frame, sample_mask):
        """Test específico del modo mosaic."""
        engine = BlurEngine(mode="mosaic", temporal_smoothing=1)
        result = engine.anonymize_frame(sample_frame, sample_mask)

        # Similar al pixelate pero con patrón irregular
        center_region = result[40:60, 40:60]
        assert np.std(center_region) < np.std(sample_frame[40:60, 40:60])

    def test_empty_mask(self, sample_frame):
        """Test con máscara vacía."""
        engine = BlurEngine()
        empty_mask = np.zeros((100, 100), dtype=np.float32)

        result = engine.anonymize_frame(sample_frame, empty_mask)

        # Debería devolver el frame original
        assert np.array_equal(result, sample_frame)

    def test_invalid_mode(self, sample_frame, sample_mask):
        """Test con modo inválido."""
        engine = BlurEngine(mode="invalid", temporal_smoothing=1)
        result = engine.anonymize_frame(sample_frame, sample_mask)

        # Debería devolver el frame sin cambios (fallback)
        assert np.array_equal(result, sample_frame)

if __name__ == "__main__":
    pytest.main([__file__])