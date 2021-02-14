import argparse
import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron.data import indexed_dataset
from megatron.tokenizer import build_tokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument('--dataset-impl', type=str, default='mmap',
                        choices=['lazy', 'cached', 'mmap'])

    parser.add_argument('--tokenizer-type', type=str, required=True,
                        choices=['BertWordPieceLowerCase', 'BertWordPieceCase',
                                 'GPT2BPETokenizer'])
    parser.add_argument('--vocab-file', type=str, default=None,
                        help='Path to the vocab file')

    args = parser.parse_args()

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


def main():
    """
    Not completed yet
    """

    args = get_args()

    tokenizer = build_tokenizer(args)

    output_bin_file = "{}_text_document.bin".format(args.output)
    output_idx_file = "{}_text_document.idx".format(args.output)

    builder = indexed_dataset.make_builder(out_file=output_bin_file,
        impl=args.dataset_impl,
        vocab_size=tokenizer.vocab_size)

    filelist = [f for f in glob.iglob(f"{args.input}/**/*.bin", recursive=True) if os.path.isfile(f)]

    for input_file in filelist:
        builder.merge_file_(os.path.splitext(input_file)[0])
        # builder.end_document()

    builder.finalize(output_idx_file)


if __name__ == "__main__":
    main()
