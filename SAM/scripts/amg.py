# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import shutil
import os
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore
import numpy as np

from SAM.segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import json
import os
from typing import Any, Dict, List

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--sam_pic",
    type=str,
    required=True,
    help="Path to either a single sam_pic image or folder of images.",
)

parser.add_argument(
    "--sam_mask",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    default='vit_h',
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default='../model/sam_vit_h_4b8939.pth',
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor."

)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)
"""
mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)
"""


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:

    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"
    metadata = [header]

    zero_folder = os.path.join(path, '0')
    os.makedirs(zero_folder, exist_ok=True)

    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"

        file_path = os.path.join(path, filename)
        cv2.imwrite(file_path, mask * 255)

        height, width = mask.shape
        center_x, center_y = width // 2, height // 2
        left_bound, right_bound = center_x - 400, center_x + 400
        top_bound, bottom_bound = center_y - 400, center_y + 400

        top_region = mask[:top_bound, :]
        bottom_region = mask[bottom_bound:, :] if height > bottom_bound else np.zeros((0, width), dtype=np.uint8)
        left_region = mask[:, :left_bound]
        right_region = mask[:, right_bound:] if width > right_bound else np.zeros((height, 0), dtype=np.uint8)

        white_pixels = (np.count_nonzero(top_region) + np.count_nonzero(bottom_region) +
                        np.count_nonzero(left_region) + np.count_nonzero(right_region))

        if white_pixels > 10000:
            destination_file_path = os.path.join(zero_folder, "0.png")
            shutil.move(file_path, destination_file_path)
            continue

        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)

    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return

def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def sam(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.95,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=300,
        box_nms_thresh=0.2,
        crop_nms_thresh=0.1
    )

    if not os.path.isdir(args.sam_pic):
        targets = [args.sam_pic]
    else:
        targets = [
            f for f in os.listdir(args.sam_pic) if not os.path.isdir(os.path.join(args.sam_pic, f))
        ]
        targets = [os.path.join(args.sam_pic, f) for f in targets]

    os.makedirs(args.sam_mask, exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = generator.generate(image)

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.sam_mask, base)
        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=True)
            write_masks_to_folder(masks, save_base)
        else:
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    sam(args)
