#!/usr/bin/env python
"""
Dust Effect Generator - Physics-based dust augmentation for construction site images.

Follows the same pattern as SnowEffectGenerator/RainEffectGenerator:
1. Estimate illumination
2. Apply atmospheric scattering with dust-specific parameters
3. Overlay visible dust particles
"""

import os
import random
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from skimage import color
from tqdm.auto import tqdm

from dust_gen import dustAttenuation, generate_dust_particles


def reduce_lightHSV(rgb, sat_red=0.5, val_red=0.5):
    hsv = color.rgb2hsv(rgb / 255)
    hsv[..., 1] *= sat_red
    hsv[..., 2] *= val_red
    return (color.hsv2rgb(hsv) * 255).astype(np.uint8)


def scale_depth(im, nR, nC):
    nR0 = len(im)
    nC0 = len(im[0])
    return np.asarray([[im[int(nR0 * r / nR)][int(nC0 * c / nC)]
                        for c in range(nC)] for r in range(nR)])


def screen_blend(image, layer):
    result = 255.0 * (1 - (1 - image / 255.0) * (1 - layer[:, :, None] / 255.0))
    return result.astype(np.uint8)


def alpha_blend(img, layer, alpha):
    if layer.ndim == 3:
        import cv2
        layer = cv2.cvtColor(layer.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    assert alpha.ndim == 2
    assert layer.ndim == 2
    blended = img * (1 - alpha[:, :, None]) + layer[:, :, None] * alpha[:, :, None]
    return blended


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear_path", type=str, required=True, help="path to the file or the folder")
    parser.add_argument("--depth_path", type=str, required=True, help="path to the file or the folder")
    parser.add_argument("--save_folder", type=str, default="./generated/", help="path to the folder")
    parser.add_argument("--txt_file", default=None, help="path to text file with image names")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


class DustEffectGenerator:
    """
    Physics-based dust effect generator for construction site images.

    Key differences from fog/rain/snow:
    - Warm brown/tan atmospheric light color
    - Dust concentrates near ground (elevation-weighted)
    - Lower visibility (50-500m for construction dust)
    - Visible dust particle overlay
    """

    def __init__(self):
        # Darkness reduction per illumination level (less than snow/rain - dust doesn't darken as much)
        self._illumination2darkness = {0: 1.0, 1: 0.95, 2: 0.9, 3: 0.85}

        # Visibility range in meters (much lower than fog/rain)
        self._weather2visibility = (100, 500)

        # Dust color ranges per illumination level: (R_range, G_range, B_range)
        # Warmer/brighter dust for brighter scenes
        self._illumination2dustcolor = {
            0: {'r': (160, 180), 'g': (120, 140), 'b': (70, 90)},    # dark scene: muted dust
            1: {'r': (180, 200), 'g': (140, 160), 'b': (80, 110)},   # dim
            2: {'r': (190, 210), 'g': (150, 175), 'b': (90, 120)},   # medium
            3: {'r': (200, 220), 'g': (160, 185), 'b': (100, 135)},  # bright: sandy dust
        }

        # Dust height: how high dust cloud extends (meters)
        self._dust_height_range = (8, 20)

        # Particle overlay intensity (subtle specks, not snow-like)
        self._particle_density_ratio = (0.0001, 0.0004)  # ratio of h*w

    def getIlluminationMapCheat(self, img):
        T = color.rgb2gray(img)
        return T

    def _get_dust_color(self, illumination):
        """Get random dust color based on scene illumination."""
        cfg = self._illumination2dustcolor[illumination]
        r = random.randint(cfg['r'][0], cfg['r'][1])
        g = random.randint(cfg['g'][0], cfg['g'][1])
        b = random.randint(cfg['b'][0], cfg['b'][1])
        return (r, g, b)

    def genDustParticleLayer(self, h=720, w=1280):
        """Generate dust particle overlay layer."""
        num_particles = random.randint(
            int(h * w * self._particle_density_ratio[0]),
            int(h * w * self._particle_density_ratio[1])
        )
        wind_angle = random.choice([-1, 1]) * random.randint(0, 30)
        return generate_dust_particles(h, w, num_particles=num_particles, wind_angle=wind_angle)

    def genEffect(self, img_path, depth_path):
        """
        Generate dust effect on a single image.

        Args:
            img_path: path to input image
            depth_path: path to depth map (.npy)

        Returns:
            Dust-augmented image as uint8 numpy array
        """
        I = np.array(Image.open(img_path))
        D = np.load(depth_path)

        hI, wI, _ = I.shape
        hD, wD = D.shape

        if hI != hD or wI != wD:
            D = scale_depth(D, hI, wI)

        # Estimate scene illumination
        T = self.getIlluminationMapCheat(I)
        illumination_array = np.histogram(T, bins=4, range=(0, 1))[0] / T.size
        illumination = illumination_array.argmax()

        # Get dust parameters
        visibility = random.randint(self._weather2visibility[0], self._weather2visibility[1])
        dust_color = self._get_dust_color(illumination)
        dust_height = random.uniform(self._dust_height_range[0], self._dust_height_range[1])

        # Reduce light slightly (dust dims the scene less than rain/snow)
        if illumination > 0:
            I_dark = reduce_lightHSV(
                I,
                sat_red=self._illumination2darkness[illumination],
                val_red=self._illumination2darkness[illumination]
            )
        else:
            I_dark = I
            visibility = min(visibility, int(D.max() * 0.6)) if D.max() < 1000 else 300

        # Apply atmospheric dust scattering
        I_dust = dustAttenuation(I_dark, D, visibility=visibility,
                                 dust_color=dust_color, dust_height=dust_height)

        # Overlay visible dust particles
        dust_layer = self.genDustParticleLayer(h=hI, w=wI)
        I_final = screen_blend(I_dust, dust_layer)

        return I_final.astype(np.uint8)


def main():
    args = parse_arguments()
    dustgen = DustEffectGenerator()

    clearP = Path(args.clear_path)
    depthP = Path(args.depth_path)

    if clearP.is_file() and (depthP.is_file() and depthP.suffix == ".npy"):
        dusty = dustgen.genEffect(clearP, depthP)
        if args.show:
            Image.fromarray(dusty).show()

    if clearP.is_dir() and depthP.is_dir():
        if args.txt_file:
            with open(args.txt_file, 'r') as f:
                files = f.read().split('\n')
            image_files = [clearP / f for f in files if f.strip()]
        else:
            image_files = sorted(Path(clearP).glob("*"))

        depth_files = [Path(depthP) / ("-".join(imgf.name.split('-')[:2]) + ".npy") for imgf in image_files]

        valid_files = [idx for idx, f in enumerate(depth_files) if f.exists()]
        image_files = [image_files[idx] for idx in valid_files]
        depth_files = [depth_files[idx] for idx in valid_files]

        save_folder = Path(args.save_folder)
        if not save_folder.exists():
            os.makedirs(str(save_folder))

        for imgp, depthp in tqdm(zip(image_files, depth_files), total=len(image_files)):
            dusty = dustgen.genEffect(imgp, depthp)
            Image.fromarray(dusty).save(save_folder / (imgp.stem + "-dsyn.jpg"))


if __name__ == '__main__':
    main()
