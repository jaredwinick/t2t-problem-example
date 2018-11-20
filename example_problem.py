from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import registry

import tensorflow as tf

import numpy as np
import collections
import os
import urllib.request
import gzip

def _build_vocab(filename, vocab_path):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    #words = words[:vocab_size]
    with open(vocab_path, "w") as f:
        f.write("\n".join(words))

    return text_encoder.TokenTextEncoder(vocab_path)

def _read_words(filename):
  """Reads tokens from a sequencee file. Returns list of tokens"""
  file = gzip.open(filename, mode='r')
  contents = file.read().decode('utf-8')
  return contents.replace("\n", " ").replace("|", " ").split()
  #with gzip.open(filename, mode="r") as f:
  #  return f.read().replace("\n", " ").replace("|", " ").split()

def _download(tmp_dir):
    path, _ = urllib.request.urlretrieve(SEQUENCE_FILE_URL, os.path.join(tmp_dir, SEQUENCE_FILE_NAME))
    return path
    
SEQUENCE_FILE_URL = 'https://storage.googleapis.com/js-code/sequences_15_week_eos.txt.gz'
SEQUENCE_FILE_NAME = "sequences_15_week_eos.txt.gz"

@registry.register_problem
class Tracks(text_problems.Text2TextProblem):

    @property
    def approx_vocab_size(self):
        return 32000

    @property
    def is_generate_per_split(self):
        return False

    @property
    def vocab_filename(self):
        return "vocab.tracks"

    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN
    
    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):

        sequence_file_path = _download(tmp_dir)
        print("Downloaded to: " + sequence_file_path)
        
        file = gzip.open(sequence_file_path, 'r')
        lines = file.read().decode('utf-8').split('\n')
        print(len(lines))
        
        vocab_path = os.path.join(data_dir, self.vocab_filename)
        _build_vocab(sequence_file_path, vocab_path)

        def _generate_samples():
            for line in lines:
                fields = line.split("|")
                if (len(fields) == 2):
                    yield {
                        "inputs": fields[0],
                        "targets": fields[1]
                    }
            
        return _generate_samples()

        
