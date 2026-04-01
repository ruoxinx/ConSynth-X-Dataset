"""
Dust particle/haze generation using physics-based atmospheric scattering model.

Adapts the fog generation model (Koschmieder's law) for dust conditions:
- Warm brown/tan atmospheric light (instead of gray/white fog)
- Ground-level concentration weighting (dust settles near ground)
- Visible dust particle overlay
- Lower visibility ranges typical of construction dust (50-500m)
"""

import cv2
import random
import numpy as np
from PIL import Image
from noise import pnoise3


def perlin_noise(w, h, depth):
    """Generate Perlin noise texture for non-uniform dust distribution."""
    p1 = Image.new('L', (w, h))
    p2 = Image.new('L', (w, h))
    p3 = Image.new('L', (w, h))

    scale = 1 / 130.0
    for y in range(h):
        for x in range(w):
            v = pnoise3(x * scale, y * scale, depth[y, x] * scale,
                        octaves=1, persistence=0.5, lacunarity=2.0)
            color = int((v + 1) * 128.0)
            p1.putpixel((x, y), color)

    scale = 1 / 60.0
    for y in range(h):
        for x in range(w):
            v = pnoise3(x * scale, y * scale, depth[y, x] * scale,
                        octaves=1, persistence=0.5, lacunarity=2.0)
            color = int((v + 0.5) * 128)
            p2.putpixel((x, y), color)

    scale = 1 / 10.0
    for y in range(h):
        for x in range(w):
            v = pnoise3(x * scale, y * scale, depth[y, x] * scale,
                        octaves=1, persistence=0.5, lacunarity=2.0)
            color = int((v + 1.2) * 128)
            p3.putpixel((x, y), color)

    perlin = (np.array(p1) + np.array(p2) / 2 + np.array(p3) / 4) / 3
    return perlin


def generate_dust_color():
    """
    Generate a random dust atmospheric light color (warm brown/tan/sandy).
    Returns (R, G, B) tuple in 0-255 range.
    """
    r = random.randint(180, 220)
    g = random.randint(140, 180)
    b = random.randint(80, 130)
    return np.array([[[r, g, b]]], dtype=np.float64)


def generate_dust(image, depth, visibility=None, dust_color=None, dust_height=15.0):
    """
    Apply dust effect using atmospheric scattering model adapted for dust.

    Key differences from fog:
    - Atmospheric light is warm brown/tan (not gray)
    - Dust concentrates near ground level via elevation weighting
    - Higher extinction coefficient (lower visibility)

    Args:
        image: numpy array (h, w, c) RGB image
        depth: numpy array (h, w) depth map
        visibility: dust visibility in meters (lower = denser dust). Default: auto from depth.
        dust_color: (R, G, B) tuple or None for random warm color
        dust_height: height in meters above which dust thins out (default 15m for construction)

    Returns:
        Dust-affected image as uint8 numpy array
    """
    height, width = depth.shape
    perlin = perlin_noise(width, height, depth)

    depth_max = depth.max()

    if visibility:
        dust_visibility = visibility
    else:
        # Construction dust visibility auto-range based on scene depth
        dust_visibility = float(np.random.randint(
            max(100, int(depth_max * 0.5)),
            max(200, int(depth_max * 1.2))
        ))
        dust_visibility = np.clip(dust_visibility, 100, 800)

    VERTICAL_FOV = 60  # degrees
    CAMERA_ALTITUDE = 1.8  # meters
    # Dust molecular scattering is weaker than fog (larger particles, less Rayleigh)
    VISIBILITY_RANGE_MOLECULE = 80  # m (vs 12 for fog — dust particles scatter differently)
    VISIBILITY_RANGE_AEROSOL = dust_visibility  # m
    ECM_ = 3.912 / VISIBILITY_RANGE_MOLECULE
    ECA_ = 3.912 / VISIBILITY_RANGE_AEROSOL

    # Dust-specific: moderate height, transitions gradually
    FT = dust_height  # Dust top (m)
    HT = dust_height * 0.4  # Haze transition

    angle = np.repeat(
        -1 * np.linspace(-0.5 * VERTICAL_FOV, 0.5 * VERTICAL_FOV, height).reshape(-1, 1),
        axis=1, repeats=width
    )
    distance = depth / np.cos(np.radians(angle))
    elevation = CAMERA_ALTITUDE + distance * np.sin(np.radians(angle))

    distance_through_fog = np.zeros_like(distance)
    distance_through_haze = np.zeros_like(distance)
    distance_through_haze_free = np.zeros_like(distance)

    # Elevation-dependent concentration: dust is denser near ground
    ECA = ECA_
    c = 1 - elevation / (FT + 0.00001)
    c[c < 0] = 0
    # Gradual falloff — dust thins out smoothly with elevation
    c = c ** 1.2
    ECM = (ECM_ * c + (1 - c) * ECA_) * (perlin / 255)

    idx1 = np.logical_and(FT > elevation, elevation > HT)
    idx2 = elevation <= HT
    idx3 = elevation >= FT

    distance_through_haze[idx2] = distance[idx2]
    distance_through_fog[idx1] = (
        (elevation[idx1] - HT)
        * distance[idx1]
        / (elevation[idx1] - CAMERA_ALTITUDE)
    )
    distance_through_haze[idx1] = distance[idx1] - distance_through_fog[idx1]
    distance_through_haze[idx3] = (
        (HT - CAMERA_ALTITUDE)
        * distance[idx3]
        / (elevation[idx3] - CAMERA_ALTITUDE)
    )
    distance_through_fog[idx3] = (
        (FT - HT)
        * distance[idx3]
        / (elevation[idx3] - CAMERA_ALTITUDE)
    )
    distance_through_haze_free[idx3] = (
        distance[idx3] - distance_through_haze[idx3] - distance_through_fog[idx3]
    )

    attenuation = np.exp(-ECA * distance_through_haze - ECM * distance_through_fog)

    I_ex = image * attenuation[:, :, None]
    O_p = 1 - attenuation

    # Dust atmospheric light: warm brown/tan color
    if dust_color is None:
        I_al = generate_dust_color()
    else:
        I_al = np.array([[[dust_color[0], dust_color[1], dust_color[2]]]], dtype=np.float64)

    I = I_ex + O_p[:, :, None] * I_al
    return np.clip(I, 0, 255).astype(np.uint8)


def generate_dust_particles(h, w, num_particles=None, wind_angle=0):
    """
    Generate visible dust particle overlay layer.

    Dust particles are larger than fog/rain droplets and appear as
    semi-transparent specks, especially visible in the lower portion of the image.

    Args:
        h: image height
        w: image width
        num_particles: number of particles (auto-scaled if None)
        wind_angle: wind direction in degrees for motion blur

    Returns:
        Dust particle layer as uint8 numpy array (h, w)
    """
    if num_particles is None:
        num_particles = random.randint(int(h * w * 0.0001), int(h * w * 0.0005))

    layer = np.zeros((h, w), dtype=np.float64)

    for _ in range(num_particles):
        # Particles concentrate in lower 70% of image (ground level)
        x = random.randint(0, w - 1)
        y_bias = random.random() ** 0.5  # Bias toward bottom
        y = int(y_bias * h * 0.95) + int(h * 0.05)
        y = min(y, h - 1)

        # Small dust specks: 1-2 pixels (not snow-like blobs)
        radius = random.choices([1, 1, 1, 2], weights=[5, 3, 2, 1])[0]
        opacity = random.uniform(0.1, 0.4)

        # Subtle warm dust intensity
        intensity = random.randint(100, 170)

        cv2.circle(layer, (x, y), radius, intensity * opacity, -1)

    # Gentle Gaussian blur to blend particles naturally
    layer = cv2.GaussianBlur(layer, (3, 3), 0.5)

    # Apply motion blur if wind
    if wind_angle != 0:
        blur_size = random.choice([3, 5, 7])
        k = np.zeros((blur_size, blur_size), dtype=np.float32)
        k[(blur_size - 1) // 2, :] = np.ones(blur_size, dtype=np.float32)
        k = cv2.warpAffine(
            k, cv2.getRotationMatrix2D(
                (blur_size / 2 - 0.5, blur_size / 2 - 0.5), wind_angle, 1.0
            ), (blur_size, blur_size)
        )
        k = k * (1.0 / np.sum(k))
        layer = cv2.filter2D(layer, -1, k)

    return np.clip(layer, 0, 255).astype(np.uint8)


def dustAttenuation(img, depth, visibility=None, dust_color=None, dust_height=15.0):
    """Convenience wrapper matching fogAttenuation interface."""
    return generate_dust(
        img.copy(), depth.copy(),
        visibility=visibility,
        dust_color=dust_color,
        dust_height=dust_height
    )
