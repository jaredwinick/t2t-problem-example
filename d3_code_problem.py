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

def _download(tmp_dir):
    path, _ = urllib.request.urlretrieve(CODE_FILE_URL, os.path.join(tmp_dir, CODE_FILE_NAME))
    return path
    
CODE_FILE_URL = 'https://storage.googleapis.com/js-code/d3SourceFiles.txt.gz'
CODE_FILE_NAME = "d3SourceFiles.txt.gz"

@registry.register_problem
class D3Code(text_problems.Text2TextProblem):

    @property
    def is_generate_per_split(self):
        return False

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER
    
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
        del data_dir
        del tmp_dir
        del dataset_split

        code_file_path = _download(tmp_dir)
        print("Downloaded to: " + code_file_path)
        
        file = gzip.open(sequence_file_path, 'r')
        lines = file.read().decode('utf-8').split('\n')
        print(len(lines))
        
        ex_count = 0
        prev_line = None

        for line in lines:
            if prev_line and line:
                yield {
                    "inputs": prev_line,
                    "targets": line,
                }
                ex_count += 1
            prev_line = line