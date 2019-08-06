#coding=utf-8

import os
import sys
import argparse
import configparser

dir_path = os.path.dirname('__file__')
sys.path.append(dir_path)

config = configparser.ConfigParser()
config.read(os.path.join(dir_path, 'config.ini'))

parser = argparse.ArgumentParser()

## Common
parser.add_argument('--mode', type=str, default=config.get('common','mode'), help='Control mode.')
parser.add_argument('--restore', type=int, default=config.get('common','restore'), help='If restore from current model.')
parser.add_argument('--model_path', type=str, default=config.get('common','model_path'), help='model directory to save && restore')
parser.add_argument('--build_path', type=str, default=config.get('common','build_path'), help='output dir for processed data')
parser.add_argument('--data_path', type=str, default=config.get('common','data_path'), help='corpus data dir')
parser.add_argument('--log_dir', type=str, default=config.get('common','log_dir'), help='tensorboard log dir')

## Data
parser.add_argument('--max_sent_len', type=int, default=config.get('common','max_sent_len'), help='Max sentence length to input.')
parser.add_argument('--num_labels', type=int, default=config.get('common','num_labels'), help='size of labels to classify.')

## Model
parser.add_argument('--vocab_file', type=str, default=config.get('common','vocab_file'), help='vocab file to build dataset')
parser.add_argument('--vocab_size', type=int, default=config.get('common','vocab_size'), help='vocab size of vocab file')
parser.add_argument('--embedding_dim', type=int, default=config.get('common','embedding_dim'), help='vocab size of vocab file')
parser.add_argument('--hidden_dim', type=int, default=config.get('common','hidden_dim'), help='hidden_dim of the network')
parser.add_argument('--embedding_file', type=str, default=config.get('common','embedding_file'), help='embedding bin file')
parser.add_argument('--epoch', type=int, default=config.get('common','epoch'), help='epoch to train')
parser.add_argument('--batch_size', type=int, default=config.get('common','batch_size'), help='batch_size to train')
parser.add_argument('--keep_prob', type=float, default=config.get('common','keep_prob'), help='keep prob of dropout')
parser.add_argument('--lr', type=float, default=config.get('common','lr'), help='start learning rate')
parser.add_argument('--gradient_clip_num', type=int, default=config.get('common','gradient_clip_num'), help='gradient_clip_num')
parser.add_argument('--warmup_steps', type=int, default=config.get('common','warmup_steps'), help='steps to warmup lr')
parser.add_argument('--num_filters', type=int, default=config.get('common','num_filters'), help='number of filter in cov layer')
parser.add_argument('--save_period', type=int, default=config.get('common','save_period'), help='epoch period to save model')

args = parser.parse_args()
