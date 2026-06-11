"""Tests for solid image redaction helpers."""

from __future__ import annotations

import unittest

from PIL import Image
from text_redaction_benchmark.redaction import (
    apply_solid_redaction,
    clip_polygon_to_image,
    expand_polygon,
)
from text_redaction_benchmark.schemas import TextRegion


def make_text_region(polygon: tuple[tuple[float, float], ...]) -> TextRegion:
    """Create a minimal text region for redaction tests."""
    return TextRegion(
        polygon=polygon,
        bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
        confidence=1.0,
        source="test",
        detector="test",
    )


class SolidRedactionTest(unittest.TestCase):
    def test_apply_solid_redaction_masks_polygon_black(self) -> None:
        """Check that solid masking replaces pixels inside a polygon."""
        image = Image.new("RGB", (8, 8), "white")
        text_region = make_text_region(((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)))

        redacted_image = apply_solid_redaction(image, [text_region], mask_color="black")

        self.assertEqual(redacted_image.getpixel((3, 3)), (0, 0, 0))
        self.assertEqual(redacted_image.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(image.getpixel((3, 3)), (255, 255, 255))

    def test_apply_solid_redaction_masks_polygon_white(self) -> None:
        """Check that white masking is available for bright-document workflows."""
        image = Image.new("RGB", (8, 8), "black")
        text_region = make_text_region(((2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0)))

        redacted_image = apply_solid_redaction(image, [text_region], mask_color="white")

        self.assertEqual(redacted_image.getpixel((3, 3)), (255, 255, 255))
        self.assertEqual(redacted_image.getpixel((0, 0)), (0, 0, 0))

    def test_expand_polygon_applies_ratio_from_centroid(self) -> None:
        """Check that ratio expansion moves points away from the centroid."""
        expanded_polygon = expand_polygon(
            ((1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)),
            expand_ratio=0.5,
        )

        self.assertEqual(
            expanded_polygon,
            ((0.5, 0.5), (3.5, 0.5), (3.5, 3.5), (0.5, 3.5)),
        )

    def test_clip_polygon_to_image_clamps_points_to_pixel_bounds(self) -> None:
        """Check that polygon vertices are clipped to image dimensions."""
        clipped_polygon = clip_polygon_to_image(
            ((-10.0, 0.0), (5.0, 12.0), (100.0, -5.0)),
            image_width=10,
            image_height=10,
        )

        self.assertEqual(clipped_polygon, ((0.0, 0.0), (5.0, 9.0), (9.0, 0.0)))

    def test_negative_expansion_is_rejected(self) -> None:
        """Check that expansion settings cannot shrink privacy masks by accident."""
        with self.assertRaises(ValueError):
            expand_polygon(
                ((1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)),
                expand_pixels=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
