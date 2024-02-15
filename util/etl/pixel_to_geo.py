import argparse
import os
import pandas as pd
from pathlib import Path
import rasterio
import numpy as np
from rasterio import CRS
from rasterio.control import GroundControlPoint
from rasterio.transform import Affine, from_gcps

from typing import Tuple


def load_raw_geo_tiff(filepath: Path) -> np.ndarray:
    with rasterio.open(filepath) as fh:
        image = fh.read()
    if image is None:
        msg = f'Unknown issue caused "{filepath}" to fail while loading'
        raise Exception(msg)

    return image


def save_geo_tiff(filename: str, image: np.ndarray, crs: CRS, transform: Affine):
    image = np.array(image[...], ndmin=3)
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        compress="lzw",
        height=image.shape[1],
        width=image.shape[2],
        count=image.shape[0],
        dtype=image.dtype,
        crs=crs,
        transform=transform,
    ) as fh:
        fh.write(image)


def build_geo_ref_csv_result(map_name: str, gcp_csv_path: str) -> Tuple[CRS, Affine]:
    # Default to WGS84
    crs = CRS.from_epsg(4326)
    # Possibly switch to NAD83? epsg num : 4269
    df = pd.read_csv(gcp_csv_path)
    map_gcps = df[df["raster_id"] == map_name]
    rasterio_gcps = []
    for _, row in map_gcps.iterrows():
        rasterio_gcps.append(
            GroundControlPoint(row["row"], row["col"], row["NAD83_x"], row["NAD83_y"])
        )

    transform = from_gcps(rasterio_gcps)
    return crs, transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gcp_file", type=str, required=True)
    p = parser.parse_args()

    os.makedirs(p.output_dir, exist_ok=True)

    for root, _, files in os.walk(p.image_dir):
        for file in files:
            print(f"processing {file}")
            output_file_path = os.path.join(p.output_dir, file)
            file_path = os.path.join(root, file)
            map_id = os.path.splitext(file)[0]
            crs, transform = build_geo_ref_csv_result(map_id, p.gcp_file)

            # load the image
            im = load_raw_geo_tiff(file_path)

            # save with the new CRS
            save_geo_tiff(output_file_path, im, crs, transform)


if __name__ == "__main__":
    main()
