

import argparse

from bilm.training import dump_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--outfile', help='Output hdf5 file with weights')

    args = parser.parse_args()
    dump_weights(args.save_dir, args.outfile)

