#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk
from radiomics import featureextractor, imageoperations


def validate_case(
    sample: str,
    image_path: Path,
    mask_path: Path,
    params_path: Path,
    label: int,
) -> None:
    extractor = featureextractor.RadiomicsFeatureExtractor(str(params_path))
    settings = dict(extractor.settings)
    settings["label"] = label

    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask does not exist: {mask_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"Params file does not exist: {params_path}")

    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))

    try:
        imageoperations.checkMask(image, mask, **settings)
    except ValueError as exc:
        raise ValueError(f"PyRadiomics mask validation failed for {sample}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    image_path = Path(args.image)
    mask_path = Path(args.mask)
    params_path = Path(args.params)

    validate_case(args.sample, image_path, mask_path, params_path, args.label)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{args.sample}\tvalidated\n")


if __name__ == "__main__":
    main()
