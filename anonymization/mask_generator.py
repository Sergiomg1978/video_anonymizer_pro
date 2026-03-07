"""
Mask generation for precise head segmentation using SAM 2
Generates high-quality masks for anonymization with fallback options
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try to import SAM 2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    logger.warning("SAM 2 not available. Using fallback mask generation.")

# Try to import MobileSAM as alternative
try:
    from mobile_sam import SamPredictor as MobileSamPredictor
    MOBILE_SAM_AVAILABLE = True
except ImportError:
    MOBILE_SAM_AVAILABLE = False
    logger.warning("MobileSAM not available. Using basic fallback.")


class MaskGenerator:
    """
    Generates precise masks for head segmentation using SAM 2 or fallback methods.

    Supports both single-frame and video propagation modes for efficiency.
    """

    def __init__(self, model: str = "sam2", device: str = "cuda"):
        """
        Initialize the mask generator

        Args:
            model: Model to use ("sam2", "mobile_sam", or "fallback")
            device: Device for inference ("cuda" or "cpu")
        """
        self.model_type = model
        self.device = device
        self.sam_predictor = None
        self.video_predictor = None
        self.propagation_initialized = False
        self.frame_count = 0

        # Initialize the selected model
        if model == "sam2" and SAM2_AVAILABLE:
            self._init_sam2()
        elif model == "mobile_sam" and MOBILE_SAM_AVAILABLE:
            self._init_mobile_sam()
        else:
            logger.info(f"Using fallback mask generation (model: {model})")
            self.model_type = "fallback"

    def _init_sam2(self):
        """Initialize SAM 2 model"""
        try:
            # SAM 2 model configuration
            sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"  # Path to SAM 2 checkpoint
            model_cfg = "sam2_hiera_l.yaml"

            # Build predictors
            sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
            self.sam_predictor = SAM2ImagePredictor(sam2_model)
            self.video_predictor = SAM2VideoPredictor(sam2_model)

            logger.info("SAM 2 initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize SAM 2: {e}")
            self.model_type = "fallback"

    def _init_mobile_sam(self):
        """Initialize MobileSAM model"""
        try:
            # MobileSAM uses a different initialization
            self.sam_predictor = MobileSamPredictor()
            logger.info("MobileSAM initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MobileSAM: {e}")
            self.model_type = "fallback"

    def generate_mask(self, frame: np.ndarray, head_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Generate a binary mask for the head region

        Args:
            frame: Input frame (BGR format)
            head_bbox: Head bounding box (x1, y1, x2, y2)

        Returns:
            Binary mask (0-1 float array) of same size as frame
        """
        if self.model_type in ["sam2", "mobile_sam"] and self.sam_predictor:
            return self._generate_sam_mask(frame, head_bbox)
        else:
            return self._generate_fallback_mask(frame, head_bbox)

    def _generate_sam_mask(self, frame: np.ndarray, head_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Generate mask using SAM models"""
        try:
            # Convert BGR to RGB for SAM
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Set image for prediction
            self.sam_predictor.set_image(rgb_frame)

            # Convert bbox to SAM input format
            x1, y1, x2, y2 = head_bbox
            bbox = np.array([[x1, y1, x2, y2]])

            # Optional: Add point prompt at center of bbox for better precision
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            point_coords = np.array([[center_x, center_y]])
            point_labels = np.array([1])  # 1 = foreground point

            # Generate mask
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=bbox,
                multimask_output=True
            )

            # Select the best mask (highest score)
            if len(scores) > 0:
                best_mask_idx = np.argmax(scores)
                mask = masks[best_mask_idx].astype(np.uint8)
            else:
                mask = masks[0].astype(np.uint8)

            # Ensure mask is binary
            mask = (mask > 0).astype(np.uint8)

            # Apply morphological operations to clean up the mask
            mask = self.refine_mask(mask, dilation_px=3, feather_px=2)

            return mask

        except Exception as e:
            logger.warning(f"SAM mask generation failed: {e}. Using fallback.")
            return self._generate_fallback_mask(frame, head_bbox)

    def _generate_fallback_mask(self, frame: np.ndarray, head_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Generate mask using fallback method (expanded bbox with rounded corners)"""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        x1, y1, x2, y2 = head_bbox

        # Expand bbox by 15%
        width = x2 - x1
        height = y2 - y1
        expand_x = int(width * 0.15)
        expand_y = int(height * 0.15)

        x1 = max(0, x1 - expand_x)
        y1 = max(0, y1 - expand_y)
        x2 = min(w, x2 + expand_x)
        y2 = min(h, y2 + expand_y)

        # Create elliptical mask for rounded corners
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        axis1 = (x2 - x1) // 2
        axis2 = (y2 - y1) // 2

        # Create ellipse mask
        cv2.ellipse(mask, (center_x, center_y), (axis1, axis2), 0, 0, 360, 1, -1)

        # Apply feathering
        mask = self.refine_mask(mask.astype(np.float32), dilation_px=0, feather_px=5)

        return mask

    def initialize_video_propagation(self, frame: np.ndarray, head_bbox: Tuple[int, int, int, int]):
        """
        Initialize video propagation mode for SAM 2

        Args:
            frame: Reference frame
            head_bbox: Head bbox in reference frame
        """
        if self.model_type == "sam2" and self.video_predictor and SAM2_AVAILABLE:
            try:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Initialize video predictor
                self.video_predictor.load_first_frame(rgb_frame)

                # Convert bbox to SAM input format
                x1, y1, x2, y2 = head_bbox
                bbox = np.array([[x1, y1, x2, y2]])

                # Add point prompt
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                point_coords = np.array([[center_x, center_y]])
                point_labels = np.array([1])

                # Initialize with first frame
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=None,
                    frame_idx=0,
                    obj_id=1,
                    points=point_coords,
                    labels=point_labels,
                    box=bbox
                )

                self.propagation_initialized = True
                self.frame_count = 1
                logger.info("Video propagation initialized")

            except Exception as e:
                logger.warning(f"Video propagation initialization failed: {e}")
                self.propagation_initialized = False
        else:
            self.propagation_initialized = False

    def propagate_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Propagate mask to next frame using video predictor

        Args:
            frame: Next frame

        Returns:
            Propagated mask
        """
        if not self.propagation_initialized or not self.video_predictor:
            # Fallback to single-frame generation
            return self.generate_mask(frame, (0, 0, frame.shape[1], frame.shape[0]))  # Dummy bbox

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Propagate to next frame
            out_obj_ids, out_mask_logits = self.video_predictor.propagate_in_video(
                inference_state=None,
                frame_idx=self.frame_count
            )

            # Convert logits to mask
            mask = (out_mask_logits[0] > 0).astype(np.uint8)

            # Refine mask
            mask = self.refine_mask(mask, dilation_px=2, feather_px=2)

            self.frame_count += 1
            return mask

        except Exception as e:
            logger.warning(f"Mask propagation failed: {e}")
            # Fallback to single-frame generation
            return self.generate_mask(frame, (0, 0, frame.shape[1], frame.shape[0]))

    def reset_propagation(self):
        """Reset video propagation state"""
        self.propagation_initialized = False
        self.frame_count = 0
        if self.video_predictor:
            # Reset the video predictor state
            try:
                self.video_predictor.reset_state()
            except:
                pass

    def refine_mask(self, mask: np.ndarray, dilation_px: int = 5,
                   feather_px: int = 3) -> np.ndarray:
        """
        Refine mask with morphological operations and feathering

        Args:
            mask: Input mask (binary or float)
            dilation_px: Pixels to dilate (0 = no dilation)
            feather_px: Pixels for feathering/blurring

        Returns:
            Refined mask (float 0-1)
        """
        # Ensure mask is float
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)

        # Apply dilation if requested
        if dilation_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px, dilation_px))
            mask = cv2.dilate(mask, kernel, iterations=1)

        # Apply feathering (gaussian blur)
        if feather_px > 0:
            mask = cv2.GaussianBlur(mask, (feather_px*2+1, feather_px*2+1), 0)

        # Ensure values are in [0, 1] range
        mask = np.clip(mask, 0.0, 1.0)

        return mask

    def get_model_info(self) -> dict:
        """Get information about the current model configuration"""
        return {
            "model_type": self.model_type,
            "device": self.device,
            "sam2_available": SAM2_AVAILABLE,
            "mobile_sam_available": MOBILE_SAM_AVAILABLE,
            "video_propagation_active": self.propagation_initialized
        }