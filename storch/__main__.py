import argparse

import storch
from storch._funtext import HEADER, get_detailed_header

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-V', action='store_true', help='What the version?')
    parser.add_argument('--verbose', '-v', action='store_true', help='What more detailed info?')
    args = parser.parse_args()

    if args.version:
        print('storch version:', storch.__version__)
    elif args.verbose:
        print(get_detailed_header())
    else:
        print(HEADER)
