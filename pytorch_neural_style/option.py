import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='Neural Style')
    parser.add_argument('--content',
                            help='path to content image')
    parser.add_argument('--style',
                            help='path to style image')
    parser.add_argument('--output', default="./output",
                        help='path to output folder')
    parser.add_argument('--size', type=int,
                            help='image size')
    parser.add_argument('--num_steps', default=1000, type=int,
                        help='number of steps')
    args = parser.parse_args()
    return args