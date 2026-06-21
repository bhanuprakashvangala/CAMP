#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from radiomics import featureextractor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--params", required=True)
    parser.add_argument("--feature-class", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    extractor = featureextractor.RadiomicsFeatureExtractor(args.params)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName(args.feature_class)

    features = extractor.execute(args.image, args.mask, label=args.label)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "feature_class", "feature", "value"])
        for feature, value in features.items():
            if str(feature).startswith("diagnostics_"):
                continue
            writer.writerow([args.sample, args.feature_class, feature, value])


if __name__ == "__main__":
    main()
