"""Trapped-ball segmentation.

from: https://github.com/hepesu/LineFiller
modified by: Tomoya Sawada https://github.com/STomoya
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from skimage.color import label2rgb


def get_ball_structuring_element(radius: int) -> np.ndarray:
    """Get a ball shape structuring element with specific radius for morphology operation.

    The radius of ball usually equals to (leaking_gap_size / 2).

    Args:
        radius: int
            Radius of ball shape.

    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def get_unfilled_point(image: np.ndarray) -> np.ndarray:
    """Get points belong to unfilled(value==255) area.

    Args:
        image: np.ndarray
            an image.

    """
    y, x = np.where(image == 255)  # noqa: PLR2004
    return np.stack((x.astype(int), y.astype(int)), axis=-1)


def exclude_area(image: np.ndarray, radius: int) -> np.ndarray:
    """Perform erosion on image to exclude points near the boundary.

    We want to pick part using floodfill from the seed point after dilation.
    When the seed point is near boundary, it might not stay in the fill, and would
    not be a valid point for next floodfill operation. So we ignore these points with erosion.

    Args:
        image: np.ndarray
            an image.
        radius: int
            radius of ball shape.

    """
    return cv2.morphologyEx(image, cv2.MORPH_ERODE, get_ball_structuring_element(radius), anchor=(-1, -1), iterations=1)


def trapped_ball_fill_single(image: np.ndarray, seed_point: tuple[int, int], radius: int) -> np.ndarray:
    """Perform a single trapped ball fill operation.

    Args:
        image: np.ndarray
            an image. the image should consist of white background, black lines and black fills.
            the white area is unfilled area, and the black area is filled area.
        seed_point: tuple[int, int]
            seed point for trapped-ball fill, a tuple (integer, integer).
        radius: int
            radius of ball shape.

    """
    ball = get_ball_structuring_element(radius)

    pass1 = np.full(image.shape, 255, np.uint8)
    pass2 = np.full(image.shape, 255, np.uint8)

    im_inv = cv2.bitwise_not(image)

    # Floodfill the image
    mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    _, pass1, _, _ = cv2.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4)

    # Perform dilation on image. The fill areas between gaps became disconnected.
    pass1 = cv2.morphologyEx(pass1, cv2.MORPH_DILATE, ball, anchor=(-1, -1), iterations=1)
    mask2 = cv2.copyMakeBorder(pass1, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)

    # Floodfill with seed point again to select one fill area.
    _, pass2, _, rect = cv2.floodFill(pass2, mask2, seed_point, 0, 0, 0, 4)
    # Perform erosion on the fill result leaking-proof fill.
    pass2 = cv2.morphologyEx(pass2, cv2.MORPH_ERODE, ball, anchor=(-1, -1), iterations=1)

    return pass2


def trapped_ball_fill_multi(image: np.ndarray, radius: int, method: str = 'mean', max_iter: int = 1000) -> np.ndarray:
    """Perform multi trapped ball fill operations until all valid areas are filled.

    Args:
        image: np.ndarray
            an image. The image should consist of white background, black lines and black fills.
            the white area is unfilled area, and the black area is filled area.
        radius: int
            radius of ball shape.
        method: str (default: 'mean')
            method for filtering the fills.
            'max' is usually with large radius for select large area such as background.
        max_iter: int (default: 1000)
            max iteration number.

    """
    unfill_area = image
    filled_area, filled_area_size, result = [], [], []

    for _ in range(max_iter):
        points = get_unfilled_point(exclude_area(unfill_area, radius))

        if not len(points) > 0:
            break

        fill = trapped_ball_fill_single(unfill_area, (points[0][0], points[0][1]), radius)
        unfill_area = cv2.bitwise_and(unfill_area, fill)

        filled_area.append(np.where(fill == 0))
        filled_area_size.append(len(np.where(fill == 0)[0]))

    filled_area_size = np.asarray(filled_area_size)

    if method == 'max':
        area_size_filter = np.max(filled_area_size)
    elif method == 'median':
        area_size_filter = np.median(filled_area_size)
    elif method == 'mean':
        area_size_filter = np.mean(filled_area_size)
    else:
        area_size_filter = 0

    result_idx = np.where(filled_area_size >= area_size_filter)[0]

    for i in result_idx:
        result.append(filled_area[i])

    return result


def flood_fill_single(im: np.ndarray, seed_point: tuple[int, int]) -> np.ndarray:
    """Perform a single flood fill operation.

    Args:
        im: np.ndarray
            an image. the image should consist of white background, black lines and black fills.
            the white area is unfilled area, and the black area is filled area.
        seed_point: tuple[int, int]
            seed point for trapped-ball fill, a tuple (integer, integer).

    """
    pass1 = np.full(im.shape, 255, np.uint8)

    im_inv = cv2.bitwise_not(im)

    mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    _, pass1, _, _ = cv2.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4)

    return pass1


def flood_fill_multi(image, max_iter=20000) -> np.ndarray:
    """Perform multi flood fill operations until all valid areas are filled.

    This operation will fill all rest areas, which may result large amount of fills.

    Args:
        image: np.ndarray
            an image. the image should contain white background, black lines and black fills.
            the white area is unfilled area, and the black area is filled area.
        max_iter: int (default: 20000)
            max iteration number.

    """
    unfill_area = image
    filled_area = []

    for _ in range(max_iter):
        points = get_unfilled_point(unfill_area)

        if not len(points) > 0:
            break

        fill = flood_fill_single(unfill_area, (points[0][0], points[0][1]))
        unfill_area = cv2.bitwise_and(unfill_area, fill)

        filled_area.append(np.where(fill == 0))

    return filled_area


def mark_fill(image: np.ndarray, fills: list[np.ndarray]) -> np.ndarray:
    """Mark filled areas with 0.

    Args:
        image: np.ndarray
            an image.
        fills: list[np.ndarray]
            an array of fills' points.

    """
    result = image.copy()

    for fill in fills:
        result[fill] = 0

    return result


def build_fill_map(image: np.ndarray, fills: np.ndarray) -> np.ndarray:
    """Make an image(array) with each pixel(element) marked with fills' id. id of line is 0.

    Args:
        image: np.ndarray
            an image.
        fills: np.ndarray
            an array of fills' points.

    """
    result = np.zeros(image.shape[:2], np.int64)

    for index, fill in enumerate(fills):
        result[fill] = index + 1

    return result


def show_fill_map(fillmap: np.ndarray) -> np.ndarray:
    """Mark filled areas with colors. It is useful for visualization.

    Args:
        fillmap: np.ndarray

    """
    # Generate color for each fill randomly.
    colors = np.random.randint(0, 255, (np.max(fillmap) + 1, 3))
    # Id of line is 0, and its color is black.
    colors[0] = [0, 0, 0]

    return colors[fillmap]


def get_bounding_rect(points: np.ndarray) -> tuple[int, int, int, int]:
    """Get a bounding rect of points.

    Args:
        points: np.ndarray
            array of points.

    """
    x1, y1, x2, y2 = np.min(points[1]), np.min(points[0]), np.max(points[1]), np.max(points[0])
    return x1, y1, x2, y2


def get_border_bounding_rect(
    h: int, w: int, p1: Iterable[int, int], p2: Iterable[int, int], r: int
) -> tuple[int, int, int, int]:
    """Get a valid bounding rect in the image with border of specific size.

    Args:
        h: int
            image max height.
        w: int
            image max width.
        p1: Iterable[int, int]
            start point of rect.
        p2:  Iterable[int, int]
            end point of rect.
        r: int
            border radius.

    """
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]

    x1 = x1 - r if x1 - r > 0 else 0
    y1 = y1 - r if y1 - r > 0 else 0
    x2 = x2 + r + 1 if x2 + r + 1 < w else w
    y2 = y2 + r + 1 if y2 + r + 1 < h else h

    return x1, y1, x2, y2


def get_border_point(
    points: tuple[np.ndarray, np.ndarray], rect: tuple[int, int, int, int], max_height: int, max_width: int
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Get border points of a fill area.

    Args:
        points: tuple[np.ndarray, np.ndarray]
            points of fill .
        rect: tuple[int, int, int, int]
            bounding rect of fill.
        max_height: int
            image max height.
        max_width: int
            image max width.

    """
    # Get a local bounding rect.
    border_rect = get_border_bounding_rect(max_height, max_width, rect[:2], rect[2:], 2)

    # Get fill in rect.
    fill = np.zeros((border_rect[3] - border_rect[1], border_rect[2] - border_rect[0]), np.uint8)
    # Move points to the rect.
    fill[(points[0] - border_rect[1], points[1] - border_rect[0])] = 255

    # Get shape.
    contours, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_shape = cv2.approxPolyDP(contours[0], 0.02 * cv2.arcLength(contours[0], True), True)

    # Get border pixel.
    # Structuring element in cross shape is used instead of box to get 4-connected border.
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    border_pixel_mask = cv2.morphologyEx(fill, cv2.MORPH_DILATE, cross, anchor=(-1, -1), iterations=1) - fill

    border_pixel_points = np.where(border_pixel_mask == 255)  # noqa: PLR2004

    # Transform points back to fillmap.
    border_pixel_points = (border_pixel_points[0] + border_rect[1], border_pixel_points[1] + border_rect[0])

    return border_pixel_points, approx_shape


def merge_fill(fillmap: np.ndarray, max_iter: int = 10) -> np.ndarray:
    """Merge fill areas.

    Args:
        fillmap: np.ndarray
            an image.
        max_iter: int (default: 10)
            max iteration number.

    """
    max_height, max_width = fillmap.shape[:2]
    result = fillmap.copy()

    for _ in range(max_iter):
        result[np.where(fillmap == 0)] = 0

        fill_id = np.unique(result.flatten())
        fills = []

        for j in fill_id:
            point = np.where(result == j)

            fills.append({'id': j, 'point': point, 'area': len(point[0]), 'rect': get_bounding_rect(point)})

        for f in fills:
            # ignore lines
            if f['id'] == 0:
                continue

            border_points, approx_shape = get_border_point(f['point'], f['rect'], max_height, max_width)
            border_pixels = result[border_points]

            pixel_ids, counts = np.unique(border_pixels, return_counts=True)

            ids = pixel_ids[np.nonzero(pixel_ids)]
            new_id = f['id']
            if len(ids) == 0:
                # points with lines around color change to line color
                # regions surrounded by line remain the same
                if f['area'] < 5:  # noqa: PLR2004
                    new_id = 0
            else:
                # "region id may be set to region with largest contact
                new_id = ids[0]

            # a point
            if len(approx_shape) == 1 or f['area'] == 1:
                result[f['point']] = new_id

            if len(approx_shape) in [2, 3, 4, 5] and f['area'] < 500:  # noqa: PLR2004
                result[f['point']] = new_id

            if f['area'] < 250 and len(ids) == 1:  # noqa: PLR2004
                result[f['point']] = new_id

            if f['area'] < 50:  # noqa: PLR2004
                result[f['point']] = new_id

        if len(fill_id) == len(np.unique(result.flatten())):
            break

    return result


def thinning(fillmap: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Fill area of line with surrounding fill color.

    Args:
        fillmap: np.ndarray
            an image.
        max_iter: int (default: 10)
            max iteration number.

    """
    line_id = 0
    h, w = fillmap.shape[:2]
    result = fillmap.copy()

    for _ in range(max_iter):
        # Get points of line. if there is not point, stop.
        line_points = np.where(result == line_id)
        if not len(line_points[0]) > 0:
            break

        # Get points between lines and fills.
        line_mask = np.full((h, w), 255, np.uint8)
        line_mask[line_points] = 0
        line_border_mask = (
            cv2.morphologyEx(
                line_mask,
                cv2.MORPH_DILATE,
                cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                anchor=(-1, -1),
                iterations=1,
            )
            - line_mask
        )
        line_border_points = np.where(line_border_mask == 255)  # noqa: PLR2004

        result_tmp = result.copy()
        # Iterate over points, fill each point with nearest fill's id.
        for i, _ in enumerate(line_border_points[0]):
            x, y = line_border_points[1][i], line_border_points[0][i]

            if x - 1 > 0 and result[y][x - 1] != line_id:
                result_tmp[y][x] = result[y][x - 1]
                continue

            if x - 1 > 0 and y - 1 > 0 and result[y - 1][x - 1] != line_id:
                result_tmp[y][x] = result[y - 1][x - 1]
                continue

            if y - 1 > 0 and result[y - 1][x] != line_id:
                result_tmp[y][x] = result[y - 1][x]
                continue

            if y - 1 > 0 and x + 1 < w and result[y - 1][x + 1] != line_id:
                result_tmp[y][x] = result[y - 1][x + 1]
                continue

            if x + 1 < w and result[y][x + 1] != line_id:
                result_tmp[y][x] = result[y][x + 1]
                continue

            if x + 1 < w and y + 1 < h and result[y + 1][x + 1] != line_id:
                result_tmp[y][x] = result[y + 1][x + 1]
                continue

            if y + 1 < h and result[y + 1][x] != line_id:
                result_tmp[y][x] = result[y + 1][x]
                continue

            if y + 1 < h and x - 1 > 0 and result[y + 1][x - 1] != line_id:
                result_tmp[y][x] = result[y + 1][x - 1]
                continue

        result = result_tmp.copy()

    return result


def trappedball_segmentation(
    line_image: np.ndarray,
    color_image: np.ndarray = None,
    max_radius: int = 3,
    method: str = 'max',
    threshold: int = 220,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Trapped-ball segmentation.

    Args:
        line_image (np.ndarray): Image of line drawings.
        color_image (np.ndarray, optional): Color image used to colorize the segments.
            If given, returns the colorized segment image. Default: None.
        max_radius (int, optional): Maximum radius size. Default: 3.
        method (str, optional): method for filtering the fills. Default: 'max'.
        threshold (int, optional): threshold used to binarize line_image. Default: 220.

    Returns:
        (tuple[np.ndarray, np.ndarray]):
            Segment map, filled image.

    """
    _, result = cv2.threshold(line_image, threshold, 255, cv2.THRESH_BINARY)

    radius_list = list(range(1, max_radius + 1))
    fills = []
    for radius in reversed(radius_list):
        if radius != radius_list[-1]:
            method = None
        fills += trapped_ball_fill_multi(result, radius, method)
        result = mark_fill(result, fills)

    fills += flood_fill_multi(result)
    fillmap = build_fill_map(result, fills)
    fillmap = merge_fill(fillmap)
    fillmap = thinning(fillmap)

    if color_image is not None:
        image = label2rgb(fillmap, color_image, bg_label=0, kind='avg')
        return fillmap, image

    return fillmap
