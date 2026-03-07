"""
Codec manager for determining optimal video encoding parameters
Analyzes source video and recommends best settings for quality preservation
"""

from typing import Dict, Tuple, Optional
from fractions import Fraction
import logging

logger = logging.getLogger(__name__)


class CodecManager:
    """Analyzes video codec and determines optimal encoding parameters"""

    def __init__(self):
        # Codec compatibility mappings
        self._codec_mappings = {
            'h264': 'libx264',
            'avc1': 'libx264',
            'h265': 'libx265',
            'hevc': 'libx265',
            'vp8': 'libvpx',
            'vp9': 'libvpx-vp9',
            'av1': 'libaom-av1',
        }

        # Quality mode parameters
        self._quality_presets = {
            'lossless': {
                'libx264': {'crf': 0, 'preset': 'veryslow'},
                'libx265': {'crf': 0, 'preset': 'veryslow'},
                'ffv1': {'level': 3},
                'huffyuv': {},
            },
            'high': {
                'libx264': {'crf': 18, 'preset': 'slow'},
                'libx265': {'crf': 20, 'preset': 'slow'},
            },
            'medium': {
                'libx264': {'crf': 23, 'preset': 'medium'},
                'libx265': {'crf': 25, 'preset': 'medium'},
            }
        }

    def analyze_video(self, metadata: Dict) -> Dict:
        """Analyze video metadata and return codec analysis"""
        analysis = {
            'source_codec': metadata.get('codec_name', ''),
            'profile': metadata.get('profile', ''),
            'level': metadata.get('level', ''),
            'pix_fmt': metadata.get('pix_fmt', 'yuv420p'),
            'bitrate': metadata.get('bitrate', 0),
            'fps': metadata.get('fps', '30/1'),
            'resolution': (metadata.get('width', 1920), metadata.get('height', 1080)),
            'duration': metadata.get('duration', 0),
        }

        # Determine if source is high quality
        analysis['is_high_quality'] = self._is_high_quality_source(analysis)

        # Determine recommended target codec
        analysis['recommended_codec'] = self._get_recommended_codec(analysis)

        return analysis

    def _is_high_quality_source(self, analysis: Dict) -> bool:
        """Determine if source video is high quality"""
        # Check bitrate (rough heuristic)
        width, height = analysis['resolution']
        expected_bitrate = width * height * 30 * 0.15  # rough bytes per frame

        if analysis['bitrate'] > expected_bitrate * 2:
            return True

        # Check codec profile/level
        if analysis['profile'] in ['High', 'Main', 'Main 10']:
            return True

        # Check resolution
        if width >= 1920 and height >= 1080:
            return True

        return False

    def _get_recommended_codec(self, analysis: Dict) -> str:
        """Get recommended target codec"""
        source_codec = analysis['source_codec']

        # Try to match source codec
        if source_codec in self._codec_mappings:
            return self._codec_mappings[source_codec]

        # Default to H.264 for compatibility
        return 'libx264'

    def get_encoding_params(self, analysis: Dict, quality_mode: str = "lossless") -> Dict:
        """Get optimal encoding parameters for the given analysis and quality mode"""
        codec = analysis['recommended_codec']

        # Base parameters
        params = {
            'c:v': codec,
            'pix_fmt': analysis['pix_fmt'],
            'r': analysis['fps'],
            # Remove audio
            'an': None,
        }

        # Get quality-specific parameters
        if quality_mode in self._quality_presets:
            quality_params = self._quality_presets[quality_mode]
            if codec in quality_params:
                params.update(quality_params[codec])

        # Add codec-specific optimizations
        if codec == 'libx264':
            params.update(self._get_h264_params(analysis, quality_mode))
        elif codec == 'libx265':
            params.update(self._get_h265_params(analysis, quality_mode))
        elif codec == 'ffv1':
            params.update(self._get_ffv1_params(analysis))

        return params

    def _get_h264_params(self, analysis: Dict, quality_mode: str) -> Dict:
        """Get H.264 specific parameters"""
        params = {}

        # Profile and level matching
        if analysis['profile'] == 'High':
            params['profile:v'] = 'high'
        elif analysis['profile'] == 'Main':
            params['profile:v'] = 'main'
        else:
            params['profile:v'] = 'high'  # Default to high for quality

        # Level
        if analysis['level']:
            params['level'] = analysis['level']

        # Tune for quality
        if quality_mode == 'lossless':
            params.update({
                'preset': 'veryslow',
                'crf': '0',
                'tune': 'film',
            })
        elif quality_mode == 'high':
            params.update({
                'preset': 'slow',
                'crf': '18',
                'tune': 'film',
            })

        return params

    def _get_h265_params(self, analysis: Dict, quality_mode: str) -> Dict:
        """Get H.265/HEVC specific parameters"""
        params = {}

        if analysis['profile'] == 'Main 10':
            params['profile:v'] = 'main10'
        else:
            params['profile:v'] = 'main'

        # Level
        if analysis['level']:
            params['level'] = analysis['level']

        # Tune for quality
        if quality_mode == 'lossless':
            params.update({
                'preset': 'veryslow',
                'crf': '0',
                'x265-params': 'lossless=1',
            })
        elif quality_mode == 'high':
            params.update({
                'preset': 'slow',
                'crf': '20',
            })

        return params

    def _get_ffv1_params(self, analysis: Dict) -> Dict:
        """Get FFV1 lossless parameters"""
        return {
            'level': '3',  # Version 3 for best compression
            'slicecrc': '1',  # Add CRC for error detection
        }

    def validate_quality_preservation(self, source_analysis: Dict, target_params: Dict) -> Dict:
        """Validate that encoding parameters will preserve quality"""
        validation = {
            'quality_preserved': True,
            'warnings': [],
            'recommendations': [],
        }

        # Check if we're using lossless encoding
        if target_params.get('crf') == '0' or target_params.get('lossless') == '1':
            validation['quality_preserved'] = True
            return validation

        # Check CRF values
        crf = target_params.get('crf')
        if crf and isinstance(crf, str):
            crf_val = int(crf)
            if crf_val > 18:
                validation['warnings'].append(f"CRF value {crf_val} may introduce visible quality loss")
                validation['quality_preserved'] = False

        # Check preset
        preset = target_params.get('preset')
        if preset in ['ultrafast', 'superfast', 'veryfast', 'faster']:
            validation['warnings'].append(f"Preset '{preset}' prioritizes speed over quality")
            validation['recommendations'].append("Consider using 'slow' or 'veryslow' preset for better quality")

        # Check codec compatibility
        source_codec = source_analysis.get('source_codec', '')
        target_codec = target_params.get('c:v', '')

        if source_codec in ['h264', 'avc1'] and target_codec != 'libx264':
            validation['warnings'].append("Changing codec from H.264 may affect compatibility")

        return validation