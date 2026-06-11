"""Image redaction helpers for masking detected text regions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from PIL import Image, ImageDraw

from text_redaction_benchmark.schemas import Point, Polygon, TextRegion

MaskColor = Literal["black", "white"]


def _validate_expansion(expand_pixels: float, expand_ratio: float) -> None:
    """Reject negative expansion settings before geometry is transformed."""
    if expand_pixels < 0:
        raise ValueError("expand_pixels must be greater than or equal to 0.")
    if expand_ratio < 0:
        raise ValueError("expand_ratio must be greater than or equal to 0.")


def _resolve_fill_color(
    image_mode: str, mask_color: MaskColor
) -> int | tuple[int, ...]:
    """Return a Pillow fill color for the target image mode."""
    if mask_color == "black":
        red_value, green_value, blue_value = 0, 0, 0
    elif mask_color == "white":
        red_value, green_value, blue_value = 255, 255, 255
    else:
        raise ValueError("mask_color must be either 'black' or 'white'.")

    if image_mode == "L":
        return red_value
    if image_mode == "RGBA":
        return (red_value, green_value, blue_value, 255)
    return (red_value, green_value, blue_value)


def _clip_coordinate(
    coordinate: float, lower_bound: float, upper_bound: float
) -> float:
    """Clamp a coordinate to an inclusive numeric range."""
    return min(max(coordinate, lower_bound), upper_bound)


def _polygon_centroid(polygon: Polygon) -> Point:
    """Return the arithmetic center of a polygon's vertices."""
    if not polygon:
        raise ValueError("Cannot calculate a centroid for an empty polygon.")
    x_total = sum(point[0] for point in polygon)
    y_total = sum(point[1] for point in polygon)
    point_count = len(polygon)
    return (x_total / point_count, y_total / point_count)


def _expand_point_from_centroid(
    point: Point,
    centroid: Point,
    *,
    expand_pixels: float,
    expand_ratio: float,
) -> Point:
    """Move one polygon point away from the centroid."""
    x_offset = point[0] - centroid[0]
    y_offset = point[1] - centroid[1]
    scaled_x_offset = x_offset * (1.0 + expand_ratio)
    scaled_y_offset = y_offset * (1.0 + expand_ratio)
    distance = (x_offset**2 + y_offset**2) ** 0.5
    if distance == 0:
        pixel_x_offset = 0.0
        pixel_y_offset = 0.0
    else:
        pixel_x_offset = (x_offset / distance) * expand_pixels
        pixel_y_offset = (y_offset / distance) * expand_pixels
    return (
        centroid[0] + scaled_x_offset + pixel_x_offset,
        centroid[1] + scaled_y_offset + pixel_y_offset,
    )


def _ensure_redactable_image(image: Image.Image) -> Image.Image:
    """Return an image mode that Pillow can fill with solid RGB masks."""
    if image.mode in {"RGB", "RGBA", "L"}:
        return image.copy()
    return image.convert("RGB")


def expand_polygon(
    polygon: Polygon,
    *,
    expand_pixels: float = 0.0,
    expand_ratio: float = 0.0,
) -> Polygon:
    """Expand a polygon outward from its vertex centroid."""
    _validate_expansion(expand_pixels, expand_ratio)
    centroid = _polygon_centroid(polygon)
    return tuple(
        _expand_point_from_centroid(
            point,
            centroid,
            expand_pixels=expand_pixels,
            expand_ratio=expand_ratio,
        )
        for point in polygon
    )


def clip_polygon_to_image(
    polygon: Polygon,
    *,
    image_width: int,
    image_height: int,
) -> Polygon:
    """Clip polygon vertices to image pixel bounds."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive.")
    max_x_coordinate = float(image_width - 1)
    max_y_coordinate = float(image_height - 1)
    return tuple(
        (
            _clip_coordinate(point[0], 0.0, max_x_coordinate),
            _clip_coordinate(point[1], 0.0, max_y_coordinate),
        )
        for point in polygon
    )


def apply_solid_redaction(
    image: Image.Image,
    regions: Sequence[TextRegion],
    *,
    mask_color: MaskColor = "black",
    expand_pixels: float = 0.0,
    expand_ratio: float = 0.0,
) -> Image.Image:
    """Return a copy of an image with text region polygons solid-masked."""
    redacted_image = _ensure_redactable_image(image)
    fill_color = _resolve_fill_color(redacted_image.mode, mask_color)
    image_width, image_height = redacted_image.size
    image_draw = ImageDraw.Draw(redacted_image)

    for text_region in regions:
        expanded_polygon = expand_polygon(
            text_region.polygon,
            expand_pixels=expand_pixels,
            expand_ratio=expand_ratio,
        )
        clipped_polygon = clip_polygon_to_image(
            expanded_polygon,
            image_width=image_width,
            image_height=image_height,
        )
        if len(clipped_polygon) >= 3:
            image_draw.polygon(clipped_polygon, fill=fill_color)

    return redacted_image
