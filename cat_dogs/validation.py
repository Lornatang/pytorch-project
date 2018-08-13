"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: validation.py
# time: 2018/8/13 10:12
# license: MIT
"""

import argparse

from . import train

parser = argparse.ArgumentParser("""Val complete!""")
parser.add_argument('--img', '-i', type=str,
                    help="""Input your img.""")
args = parser.parse_args()

if __name__ == '__main__':

    train.val(args.img)
