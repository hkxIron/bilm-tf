
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, max_char_count_in_token=50)

    # define the options
    batch_size = 128  # batch size for each GPU
    n_gpus = 3

    # number of tokens in training data (this for 1B Word Benchmark)
    #n_train_tokens = 768648884
    n_train_tokens = 50 # TODO:暂时用测试集中的小数据

    options = {
     'bidirectional': True,

     'char_cnn': {
      'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [
           [1, 32], # kernel_size=1的有32个
           [2, 32], # kernel_size=2的有32个
           [3, 64],
           #[4, 128],
           #[5, 256],
           #[6, 512],
           #[7, 1024]
      ],
      'max_characters_per_token': 50,
      'n_characters': 261, # 256 + 5(mask,unk,sos,eos,padding)
      'n_highway': 2
     },
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      #'dim': 4096,
      'dim': 30, # TODO
      'n_layers': 2,
      'proj_clip': 3,
      #'projection_dim': 512,
      'projection_dim': 10, # TODO
      'use_skip_connections': True
     },
    
     'all_clip_norm_val': 10.0,
     'n_epochs': 10,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
      #'n_negative_samples_batch': 8192,
     'n_negative_samples_batch': 20, # TODO
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='../data/checkpoint/', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', default='../tests/fixtures/train/vocab.txt', help='Vocabulary file')
    parser.add_argument('--train_prefix', default='../tests/fixtures/train/data.txt', help='Prefix for train files')

    args = parser.parse_args()
    main(args)

