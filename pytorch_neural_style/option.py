import argparse


def parse_args():
    """
    Args parser
    :return:
    """
    parser = argparse.ArgumentParser(description='Neural Style')
    parser.add_argument('--content',
                            help='path to content image')
    parser.add_argument('--style',
                            help='path to style image')
    parser.add_argument('--output', default="./output",
                        help='path to output folder')
    parser.add_argument('--size', type=int,
                            help='image size (as width)')
    parser.add_argument('--steps', default=1000, type=int,
                        help='number of steps')
    parser.add_argument('--save_every', default=50, type=int,
                        help='save output file every n steps')
    parser.add_argument('--style_weight', default=1000000, type=int,
                        help='style weight')
    parser.add_argument('--content_weight', default=1, type=int,
                        help='content weight')
    args = parser.parse_args()
    return args