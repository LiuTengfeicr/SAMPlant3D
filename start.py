import argparse
from main import start

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script parameters")

    """SAMPoint Script parameters"""
    parser.add_argument('--file_path', type=str, default='data', help='input file')
    parser.add_argument('--output_pointcloud', type=str,
                        default="output/pointcloud",
                        help='output file')
    parser.add_argument('--output_measure', type=str, default='output/measure',
                        help='output measure directory')
    parser.add_argument('--input_pic', type=str, default='output/sampic', help='SAM picture')
    parser.add_argument('--output_mask', type=str, default='output/sammask', help='SAM mask')
    parser.add_argument('--alpha_v', type=int, default=0.16, help='alpha v')
    parser.add_argument('--alpha_s', type=int, default=0.04, help='alpha s')
    parser.add_argument('--method_noise', type=str, default='median', help='median/gaussian/bilateral')
    parser.add_argument('--method_Volume', type=str, default='Alpha', help='Alpha/Normalized')
    parser.add_argument('--eps', type=int, default=0.05, help='eps')
    parser.add_argument('--min_points', type=int, default=20, help='min_points in plant')
    parser.add_argument('--Normalized_Volume', type=str, default=False, help='Normalized Volume')
    parser.add_argument('--measure', type=str, default=False, help='Whether measurements are required')

    """SAM Script parameters"""
    parser.add_argument("--checkpoint", type=str, default='SAM/model/sam_vit_h_4b8939.pth',
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--model-type", type=str, default='vit_h',
                        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
    parser.add_argument("--convert-to-rle", action="store_true", help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. ""Requires pycocotools."))
    amg_settings = parser.add_argument_group("AMG Settings")
    amg_settings.add_argument("--points-per-side", type=int, default=None,
                              help="Generate masks by sampling a grid over the image with this many points to a side.")
    amg_settings.add_argument("--points-per-batch", type=int, default=None,
                              help="How many input points to process simultaneously in one batch.")
    amg_settings.add_argument("--pred-iou-thresh", type=float, default=None,
                              help="Exclude masks with a predicted score from the model that is "
                                   "lower than this threshold.")
    amg_settings.add_argument("--stability-score-thresh", type=float, default=None,
                              help="Exclude masks with a stability score lower than this threshold.")
    amg_settings.add_argument("--stability-score-offset", type=float, default=None,
                              help="Larger values perturb the mask more when measuring stability score.")
    amg_settings.add_argument("--box-nms-thresh", type=float, default=None,
                              help="The overlap threshold for excluding a duplicate mask.")
    amg_settings.add_argument("--crop-n-layers", type=int, default=None, help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."))
    amg_settings.add_argument("--crop-nms-thresh", type=float, default=None,
                              help="The overlap threshold for excluding duplicate masks across different crops.")
    amg_settings.add_argument("--crop-overlap-ratio", type=int, default=None,
                              help="Larger numbers mean image crops will overlap more.", )
    amg_settings.add_argument("--crop-n-points-downscale-factor", type=int, default=None,
                              help="The number of points-per-side in each layer of crop is reduced by this factor.")
    amg_settings.add_argument("--min-mask-region-area", type=int, default=None, help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."))

    args = parser.parse_args()
    start(args)
