"""Transfer text layer from original video to generated variants using OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..util.logging import get_logger

logger = get_logger(__name__)


class VideoTextTransfer:
    """Extract text layer from original video and composite it onto variants."""

    def __init__(self, input_video_path: Path):
        """Initialize with input video path.

        Args:
            input_video_path: Path to the original video with text overlay
        """
        self.input_video = str(input_video_path)
        self.cap = cv2.VideoCapture(self.input_video)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_video_path}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"Video loaded: {self.width}x{self.height} @ {self.fps}fps, {self.total_frames} frames",
            extra={"event": "text_transfer.init"}
        )

    def __del__(self):
        """Release video capture on cleanup."""
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()

    def extract_text_mask(self, sample_frames: int = 30) -> tuple[np.ndarray, np.ndarray]:
        """Extract static text layer as a mask using temporal variance.

        Static elements (like text overlays) have low variance across frames,
        while dynamic content (people, objects) has high variance.

        Args:
            sample_frames: Number of frames to sample for analysis

        Returns:
            Tuple of (text_layer_rgb, alpha_mask)
            - text_layer_rgb: RGB image of the text layer (H, W, 3)
            - alpha_mask: Normalized alpha channel (H, W) in range [0, 1]
        """
        logger.info(f"Extracting text layer from {sample_frames} sample frames...")

        # Read sample frames evenly distributed across video
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        step = max(1, self.total_frames // sample_frames)
        for i in range(0, self.total_frames, step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame.astype(np.float32))
            if len(frames) >= sample_frames:
                break

        if not frames:
            raise RuntimeError("Failed to read any frames from video")

        frames_array = np.array(frames)
        logger.debug(f"Loaded {len(frames)} frames for analysis")

        # Calculate temporal standard deviation
        # Static pixels (text) have low std, dynamic pixels have high std
        std_frame = np.std(frames_array, axis=0)
        std_gray = cv2.cvtColor(std_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        std_gray = cv2.medianBlur(std_gray, 3)

        # Create mask: low std = static element (text)
        # Threshold determined empirically - adjust if needed
        _, static_mask = cv2.threshold(std_gray, 3, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Calculate mean frame for text layer
        mean_frame = np.mean(frames_array, axis=0).astype(np.uint8)

        # Extract only text region
        text_layer = mean_frame.copy()
        text_layer[static_mask == 0] = 0

        # Create alpha channel (0-1 range)
        alpha = static_mask.astype(np.float32) / 255.0

        # Smooth edges for better blending
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

        text_pixel_count = np.sum(static_mask > 0)
        text_percentage = (text_pixel_count / (self.width * self.height)) * 100

        logger.info(
            f"Text layer extracted: {text_pixel_count} pixels ({text_percentage:.2f}% of frame)",
            extra={"event": "text_transfer.extract_complete"}
        )

        return text_layer, alpha
    # from typing import Tuple
    # import numpy as np, cv2

    # def extract_text_mask(
    #     self,
    #     sample_frames: int = 30,
    #     variance_pct: float = 20.0,   # чуть строже, чем 18
    #     edge_mode: str = "canny",     # "canny" | "sobel"
    #     stabilize: bool = False,
    #     feather_px: float = 0.8      # внутреннее перо в пикселях (0 = без пера)
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     logger = getattr(self, "logger", None)
    #     if logger: logger.info(f"Extracting text layer (crisp) from {sample_frames} frames")

    #     # --- выборка кадров ---
    #     frames = []
    #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #     step = max(1, int(self.total_frames // max(1, sample_frames)))
    #     for i in range(0, self.total_frames, step):
    #         self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    #         ok, f = self.cap.read()
    #         if ok: frames.append(f.astype(np.float32))
    #         if len(frames) >= sample_frames: break
    #     if not frames: raise RuntimeError("Failed to read any frames from video")
    #     H, W = frames[0].shape[:2]

    #     # --- (опц.) стабилизация без сглаживания ---
    #     if stabilize and len(frames) > 1:
    #         ref_gray = cv2.cvtColor(frames[0].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    #         crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
    #         for i in range(1, len(frames)):
    #             try:
    #                 g = cv2.cvtColor(frames[i].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    #                 M = np.eye(2, 3, dtype=np.float32)
    #                 cv2.findTransformECC(ref_gray, g, M, cv2.MOTION_EUCLIDEAN, crit)
    #                 frames[i] = cv2.warpAffine(
    #                     frames[i], M, (W, H),
    #                     flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,  # без линейного сглаживания
    #                     borderMode=cv2.BORDER_REPLICATE
    #                 )
    #             except cv2.error:
    #                 pass

    #     F = np.stack(frames, 0)  # (N,H,W,3) float32

    #     # --- дисперсия по яркости ---
    #     luma = 0.114*F[...,0] + 0.587*F[...,1] + 0.299*F[...,2]
    #     std_luma = np.std(luma, axis=0).astype(np.float32)
    #     thr = float(np.percentile(std_luma, variance_pct))
    #     low_var_mask = (std_luma <= thr).astype(np.uint8) * 255

    #     # --- медианный кадр для устойчивых контуров ---
    #     median_frame = np.median(F, axis=0).astype(np.uint8)
    #     median_gray  = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)

    #     # --- узкие края: Canny вместо Sobel + лёгкая дилатация ---
    #     if edge_mode == "canny":
    #         v = float(np.median(median_gray))
    #         lo = int(max(0, 0.66 * v))
    #         hi = int(min(255, 1.33 * v))
    #         edges = cv2.Canny(median_gray, lo, hi)
    #         k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #         edge_mask = cv2.dilate(edges, k, iterations=1)
    #     else:
    #         dx = cv2.Sobel(median_gray, cv2.CV_32F, 1, 0, ksize=3)
    #         dy = cv2.Sobel(median_gray, cv2.CV_32F, 0, 1, ksize=3)
    #         mag = cv2.magnitude(dx, dy)
    #         edge_thr = float(np.percentile(mag, 70))
    #         edge_mask = (mag >= edge_thr).astype(np.uint8) * 255

    #     # --- пересечение + компактная морфология ---
    #     static_mask = cv2.bitwise_and(low_var_mask, edge_mask)
    #     k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #     static_mask = cv2.morphologyEx(static_mask, cv2.MORPH_CLOSE, k, iterations=1)
    #     static_mask = cv2.erode(static_mask, k, iterations=0)   # убираем «ореол»

    #     # --- жёсткая альфа + перо внутрь через distance transform ---
    #     hard = (static_mask > 0).astype(np.uint8) * 255
    #     if feather_px > 0:
    #         dist = cv2.distanceTransform(hard, cv2.DIST_L2, 3)  # расстояние до фона
    #         alpha = np.clip(dist / max(feather_px, 1e-6), 0, 1).astype(np.float32)
    #         alpha[hard == 0] = 0.0
    #     else:
    #         alpha = (hard / 255.0).astype(np.float32)

    #     # --- цвет: median-near (пер-пиксельно ближайший к медиане кадр) ---
    #     # уменьшает шум, но сохраняет резкие границы
    #     N = F.shape[0]
    #     median_rgb = median_frame.astype(np.float32)
    #     # L1 расстояние до медианы для каждого кадра
    #     diffs = [np.abs(F[i] - median_rgb).mean(axis=2) for i in range(N)]  # (H,W)
    #     diffs = np.stack(diffs, 0)  # (N,H,W)
    #     best_idx = np.argmin(diffs, axis=0)  # (H,W)
    #     # собираем цвет покадрово
    #     text_layer = np.zeros_like(median_frame)
    #     for i in range(N):
    #         sel = (best_idx == i)
    #         if np.any(sel):
    #             text_layer[sel] = F[i][sel].astype(np.uint8)

    #     # обнуляем фон
    #     text_layer[alpha < 1e-3] = 0

    #     if logger:
    #         px = int((alpha > 0.05).sum())
    #         logger.info(f"text px: {px} ({px/(H*W):.2%})")

    #     return text_layer, alpha




    def extract_background_video(self, output_path: Path, text_mask: np.ndarray) -> Path:
        """Create video without text using inpainting.

        This removes the text from the original video by inpainting the text regions.

        Args:
            output_path: Path to save the background video
            text_mask: Binary mask of text regions (from extract_text_mask alpha)

        Returns:
            Path to the saved background video
        """
        logger.info(f"Creating background video without text: {output_path}")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))

        # Expand mask for inpainting (to cover edges)
        kernel = np.ones((15, 15), np.uint8)
        inpaint_mask = cv2.dilate((text_mask > 0).astype(np.uint8) * 255, kernel, iterations=2)

        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Inpaint text region with surrounding content
            frame_inpainted = cv2.inpaint(frame, inpaint_mask, 3, cv2.INPAINT_TELEA)

            out.write(frame_inpainted)
            frame_count += 1

            if frame_count % 30 == 0:
                logger.debug(f"Inpainted {frame_count}/{self.total_frames} frames")

        out.release()
        logger.info(
            f"Background video saved: {frame_count} frames",
            extra={"event": "text_transfer.background_complete"}
        )

        return output_path

    def composite_text_on_video(
        self,
        background_video: Path,
        text_layer: np.ndarray,
        alpha: np.ndarray,
        output_path: Path
    ) -> Path:
        """Composite text layer onto a new video using alpha blending.

        Args:
            background_video: Path to the video to add text to (AI-generated variant)
            text_layer: RGB text layer from extract_text_mask
            alpha: Alpha mask from extract_text_mask (0-1 range)
            output_path: Path to save the final composited video

        Returns:
            Path to the composited video
        """
        logger.info(f"Compositing text onto video: {output_path}")

        cap_bg = cv2.VideoCapture(str(background_video))

        if not cap_bg.isOpened():
            raise RuntimeError(f"Failed to open background video: {background_video}")

        # Get target video properties
        target_fps = int(cap_bg.get(cv2.CAP_PROP_FPS))
        target_width = int(cap_bg.get(cv2.CAP_PROP_FRAME_WIDTH))
        target_height = int(cap_bg.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Target video: {target_width}x{target_height} @ {target_fps}fps")

        # Check if scaling is needed
        source_height, source_width = text_layer.shape[:2]
        if (source_height, source_width) != (target_height, target_width):
            # Check orientation compatibility
            source_is_vertical = source_height > source_width
            target_is_vertical = target_height > target_width

            if source_is_vertical != target_is_vertical:
                logger.error(
                    f"Orientation mismatch! Source is {'vertical' if source_is_vertical else 'horizontal'} "
                    f"({source_width}x{source_height}) but target is {'vertical' if target_is_vertical else 'horizontal'} "
                    f"({target_width}x{target_height}). This will distort the text!"
                )
                raise ValueError(
                    f"Cannot transfer text between videos with different orientations. "
                    f"Source: {source_width}x{source_height} ({'vertical' if source_is_vertical else 'horizontal'}), "
                    f"Target: {target_width}x{target_height} ({'vertical' if target_is_vertical else 'horizontal'})"
                )

            logger.warning(
                f"Size mismatch! Scaling text layer from {source_width}x{source_height} to {target_width}x{target_height}"
            )

            # Scale text layer and alpha to match target resolution
            text_layer = cv2.resize(text_layer, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            alpha = cv2.resize(alpha, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

            # Renormalize alpha after scaling
            alpha = np.clip(alpha, 0, 1)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, target_fps, (target_width, target_height))

        # Prepare 3-channel alpha for RGB composition
        alpha_3ch = np.stack([alpha] * 3, axis=-1)

        frame_count = 0
        while True:
            ret, bg_frame = cap_bg.read()
            if not ret:
                break

            # Alpha composition: result = text * alpha + background * (1 - alpha)
            composited = (
                text_layer.astype(np.float32) * alpha_3ch +
                bg_frame.astype(np.float32) * (1 - alpha_3ch)
            )
            composited = np.clip(composited, 0, 255).astype(np.uint8)

            out.write(composited)
            frame_count += 1

            if frame_count % 30 == 0:
                logger.debug(f"Composited {frame_count} frames")

        cap_bg.release()
        out.release()

        logger.info(
            f"Composition complete: {frame_count} frames",
            extra={"event": "text_transfer.composite_complete"}
        )

        return output_path

    def transfer_text_to_video(self, variant_video: Path, output_path: Path) -> Path:
        """Complete pipeline: extract text from original and composite onto variant.

        This is a convenience method that combines all steps:
        1. Extract text layer from original video
        2. Composite text onto variant video

        Args:
            variant_video: AI-generated variant video (without text)
            output_path: Path to save the final video with text

        Returns:
            Path to the final composited video
        """
        logger.info("Starting text transfer pipeline...")

        # Extract text layer
        text_layer, alpha = self.extract_text_mask()

        # Composite onto variant
        result = self.composite_text_on_video(variant_video, text_layer, alpha, output_path)

        logger.info(f"Text transfer complete: {result}")
        return result


def apply_text_overlay_opencv(
    original_video: Path,
    variant_video: Path,
    output_path: Path,
) -> Path:
    """Apply text overlay from original video to variant using OpenCV method.

    This is the main entry point for the OpenCV-based text transfer approach.
    It extracts the text layer from the original video and composites it onto
    the AI-generated variant.

    Args:
        original_video: Original video with text overlay
        variant_video: AI-generated variant without text
        output_path: Path to save the final result

    Returns:
        Path to the output video
    """
    transfer = VideoTextTransfer(original_video)
    return transfer.transfer_text_to_video(variant_video, output_path)
