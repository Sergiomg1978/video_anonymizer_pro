"""
Scene interpretation for video analysis
Analyzes scene composition and provides contextual understanding using heuristics and optional VLM
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class SceneContext:
    """Contextual information about the current scene"""
    scene_type: str  # "both_visible", "woman_alone", "child_alone", "woman_entering", etc.
    num_persons_visible: int
    woman_visible: bool
    woman_confidence: float
    child_visible: bool
    estimated_distance: str  # "close", "medium", "far"
    occlusion_detected: bool
    frame_region_hint: Optional[str]  # "left", "right", "center", "entering_left", etc.

class SceneInterpreter:
    """
    Interprets scene content using two levels:
    1. Heuristic analysis based on detections and tracks
    2. Optional vision-language model (Florence-2) for advanced understanding
    """

    def __init__(self, use_vlm: bool = False, device: str = "cuda"):
        """
        Initialize scene interpreter

        Args:
            use_vlm: Whether to use vision-language model for advanced analysis
            device: Device for VLM inference ("cuda" or "cpu")
        """
        self.use_vlm = use_vlm
        self.device = device
        self.vlm_model = None
        self.vlm_processor = None

        if use_vlm:
            self._load_vlm_model()

        # Scene analysis parameters
        self.frame_width = None
        self.frame_height = None
        self.distance_thresholds = {
            'close': 0.6,    # Person takes >60% of frame height
            'medium': 0.3,   # Person takes 30-60% of frame height
            'far': 0.3       # Person takes <30% of frame height
        }

    def analyze_frame(self, frame: np.ndarray, detections: List[Any],
                     tracks: List[Any]) -> SceneContext:
        """
        Analyze the current frame and return scene context

        Args:
            frame: Current video frame
            detections: List of person detections
            tracks: List of active tracks

        Returns:
            SceneContext with analysis results
        """
        self.frame_height, self.frame_width = frame.shape[:2]

        # Level 1: Heuristic analysis
        heuristic_context = self._heuristic_analysis(frame, detections, tracks)

        # Level 2: VLM analysis (if enabled and conditions met)
        if self.use_vlm and self._should_invoke_vlm([t.confidence for t in tracks if hasattr(t, 'confidence')]):
            vlm_context = self._vlm_analysis(frame, detections, tracks)
            # Merge VLM results with heuristic results
            final_context = self._merge_contexts(heuristic_context, vlm_context)
        else:
            final_context = heuristic_context

        return final_context

    def should_invoke_vlm(self, confidence_history: List[float]) -> bool:
        """
        Determine if VLM should be invoked based on confidence history

        Args:
            confidence_history: List of recent confidence scores

        Returns:
            True if VLM should be invoked
        """
        return self._should_invoke_vlm(confidence_history)

    def _heuristic_analysis(self, frame: np.ndarray, detections: List[Any],
                           tracks: List[Any]) -> SceneContext:
        """Perform heuristic scene analysis"""

        # Extract person information from tracks (prioritize tracks over raw detections)
        persons = []
        for track in tracks:
            if hasattr(track, 'person_bbox') and hasattr(track, 'classification'):
                person_info = {
                    'bbox': track.person_bbox,
                    'classification': track.classification,
                    'confidence': getattr(track, 'confidence', 0.5),
                    'is_target': getattr(track, 'is_target', False)
                }
                persons.append(person_info)

        # Fallback to detections if no tracks
        if not persons:
            for detection in detections:
                if hasattr(detection, 'bbox'):
                    person_info = {
                        'bbox': detection.bbox,
                        'classification': getattr(detection, 'classification', None),
                        'confidence': getattr(detection, 'confidence', 0.5),
                        'is_target': False
                    }
                    persons.append(person_info)

        num_persons = len(persons)

        # Analyze each person
        woman_visible = False
        woman_confidence = 0.0
        child_visible = False
        target_person = None

        for person in persons:
            classification = person['classification']
            if classification and hasattr(classification, 'is_adult') and hasattr(classification, 'is_female'):
                if classification.is_adult and classification.is_female:
                    woman_visible = True
                    woman_confidence = max(woman_confidence, classification.confidence)
                    if person['is_target']:
                        target_person = person
                elif not classification.is_adult:
                    child_visible = True

        # Determine scene type
        scene_type = self._classify_scene_type(persons, woman_visible, child_visible)

        # Estimate distance
        estimated_distance = self._estimate_distance(persons)

        # Check for occlusions
        occlusion_detected = self._detect_occlusions(persons, frame.shape[:2])

        # Determine frame region hint
        frame_region_hint = self._get_frame_region_hint(target_person, frame.shape[:2])

        return SceneContext(
            scene_type=scene_type,
            num_persons_visible=num_persons,
            woman_visible=woman_visible,
            woman_confidence=woman_confidence,
            child_visible=child_visible,
            estimated_distance=estimated_distance,
            occlusion_detected=occlusion_detected,
            frame_region_hint=frame_region_hint
        )

    def _classify_scene_type(self, persons: List[Dict], woman_visible: bool,
                           child_visible: bool) -> str:
        """Classify the type of scene based on visible persons"""

        if woman_visible and child_visible:
            # Check if persons are close to each other (interaction scene)
            if len(persons) >= 2:
                bboxes = [p['bbox'] for p in persons]
                distances = []
                for i in range(len(bboxes)):
                    for j in range(i+1, len(bboxes)):
                        dist = self._bbox_distance(bboxes[i], bboxes[j])
                        distances.append(dist)

                avg_distance = np.mean(distances) if distances else float('inf')
                frame_diag = np.sqrt(self.frame_width**2 + self.frame_height**2)

                if avg_distance < frame_diag * 0.3:  # Persons are close
                    return "both_visible"
                else:
                    return "both_separate"
            return "both_visible"

        elif woman_visible and not child_visible:
            # Check if woman is entering/exiting
            if len(persons) == 1:
                bbox = persons[0]['bbox']
                region = self._get_bbox_region(bbox)
                if region in ['left_edge', 'right_edge', 'top_edge', 'bottom_edge']:
                    return "woman_entering"
            return "woman_alone"

        elif not woman_visible and child_visible:
            return "child_alone"

        else:  # No persons visible
            return "empty"

    def _estimate_distance(self, persons: List[Dict]) -> str:
        """Estimate distance of persons from camera"""

        if not persons:
            return "unknown"

        # Calculate average relative size
        relative_sizes = []
        for person in persons:
            bbox = person['bbox']
            bbox_height = bbox[3] - bbox[1]
            relative_size = bbox_height / self.frame_height
            relative_sizes.append(relative_size)

        avg_size = np.mean(relative_sizes)

        if avg_size > self.distance_thresholds['close']:
            return "close"
        elif avg_size > self.distance_thresholds['medium']:
            return "medium"
        else:
            return "far"

    def _detect_occlusions(self, persons: List[Dict], frame_shape: Tuple[int, int]) -> bool:
        """Detect if persons are occluding each other"""

        if len(persons) < 2:
            return False

        bboxes = [p['bbox'] for p in persons]

        # Check for overlapping bboxes
        for i in range(len(bboxes)):
            for j in range(i+1, len(bboxes)):
                if self._bboxes_overlap(bboxes[i], bboxes[j]):
                    # Check if one person is significantly in front of another
                    # This is a simple heuristic based on bbox size
                    size_i = (bboxes[i][2] - bboxes[i][0]) * (bboxes[i][3] - bboxes[i][1])
                    size_j = (bboxes[j][2] - bboxes[j][0]) * (bboxes[j][3] - bboxes[j][1])

                    size_ratio = max(size_i, size_j) / min(size_i, size_j)
                    if size_ratio > 1.5:  # Significant size difference suggests occlusion
                        return True

        return False

    def _get_frame_region_hint(self, target_person: Optional[Dict],
                              frame_shape: Tuple[int, int]) -> Optional[str]:
        """Get hint about where the target person is located in the frame"""

        if not target_person:
            return None

        bbox = target_person['bbox']
        return self._get_bbox_region(bbox)

    def _get_bbox_region(self, bbox: Tuple[int, int, int, int]) -> str:
        """Determine which region of the frame the bbox occupies"""

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Define regions
        left_threshold = self.frame_width * 0.3
        right_threshold = self.frame_width * 0.7
        top_threshold = self.frame_height * 0.3
        bottom_threshold = self.frame_height * 0.7

        if center_x < left_threshold:
            if center_y < top_threshold:
                return "top_left"
            elif center_y > bottom_threshold:
                return "bottom_left"
            else:
                return "left"
        elif center_x > right_threshold:
            if center_y < top_threshold:
                return "top_right"
            elif center_y > bottom_threshold:
                return "bottom_right"
            else:
                return "right"
        else:
            if center_y < top_threshold:
                return "top"
            elif center_y > bottom_threshold:
                return "bottom"
            else:
                return "center"

    def _bbox_distance(self, bbox1: Tuple[int, int, int, int],
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate distance between centers of two bboxes"""

        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)

        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def _bboxes_overlap(self, bbox1: Tuple[int, int, int, int],
                       bbox2: Tuple[int, int, int, int]) -> bool:
        """Check if two bboxes overlap"""

        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

    def _should_invoke_vlm(self, confidence_history: List[float]) -> bool:
        """Determine if VLM should be invoked"""

        if not self.use_vlm or not self.vlm_model:
            return False

        # Invoke VLM if:
        # 1. Recent confidence is low (< 0.6)
        # 2. Confidence is decreasing
        # 3. No recent high-confidence detections

        if not confidence_history:
            return False

        recent_conf = np.mean(confidence_history[-5:])  # Last 5 frames
        if recent_conf < 0.6:
            return True

        # Check for decreasing trend
        if len(confidence_history) >= 3:
            trend = np.polyfit(range(len(confidence_history[-5:])), confidence_history[-5:], 1)[0]
            if trend < -0.01:  # Confidence decreasing
                return True

        return False

    def _load_vlm_model(self):
        """Load Florence-2 vision-language model"""
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            import torch

            logger.info("Loading Florence-2 model for scene interpretation...")

            # Load Florence-2 model
            model_id = "microsoft/Florence-2-base"
            self.vlm_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.vlm_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self.vlm_model = self.vlm_model.to("cuda")
            else:
                self.device = "cpu"

            logger.info("Florence-2 model loaded successfully")

        except ImportError as e:
            logger.warning(f"Could not load Florence-2 model: {e}. VLM features disabled.")
            self.use_vlm = False
        except Exception as e:
            logger.error(f"Error loading VLM model: {e}")
            self.use_vlm = False

    def _vlm_analysis(self, frame: np.ndarray, detections: List[Any],
                     tracks: List[Any]) -> Dict[str, Any]:
        """Perform vision-language model analysis"""

        if not self.vlm_model or not self.vlm_processor:
            return {}

        try:
            import torch

            # Prepare image for model
            # Florence-2 expects RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create task prompt for scene understanding
            task = "<OD>"  # Object Detection task
            inputs = self.vlm_processor(text=task, images=rgb_frame, return_tensors="pt")

            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.vlm_model.generate(**inputs, max_length=100)

            # Decode response
            response = self.vlm_processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Parse response to extract scene information
            vlm_context = self._parse_vlm_response(response)

            return vlm_context

        except Exception as e:
            logger.warning(f"VLM analysis failed: {e}")
            return {}

    def _parse_vlm_response(self, response: str) -> Dict[str, Any]:
        """Parse VLM response to extract scene context"""

        # This is a simplified parser - in practice, you'd need more sophisticated
        # parsing based on Florence-2's output format

        context = {}

        # Look for person detections
        if "person" in response.lower():
            context['vlm_person_detected'] = True
        else:
            context['vlm_person_detected'] = False

        # Look for location hints
        location_keywords = {
            'left': 'left',
            'right': 'right',
            'center': 'center',
            'top': 'top',
            'bottom': 'bottom'
        }

        for keyword, region in location_keywords.items():
            if keyword in response.lower():
                context['vlm_location_hint'] = region
                break

        return context

    def _merge_contexts(self, heuristic_context: SceneContext,
                       vlm_context: Dict[str, Any]) -> SceneContext:
        """Merge heuristic and VLM results"""

        # For now, prefer heuristic results but boost confidence if VLM agrees
        merged = SceneContext(**heuristic_context.__dict__)

        if vlm_context.get('vlm_person_detected') and heuristic_context.woman_visible:
            # VLM confirms person detection, boost confidence
            merged.woman_confidence = min(1.0, merged.woman_confidence + 0.1)

        if 'vlm_location_hint' in vlm_context:
            # Use VLM location hint if available
            merged.frame_region_hint = vlm_context['vlm_location_hint']
        return merged