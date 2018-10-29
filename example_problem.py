from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import registry

import numpy as np

@registry.register_problem
class Parabola(text_problems.Text2TextProblem):

    @property
    def approx_vocab_size(self):
        return 1000

    @property
    def is_generate_per_split(self):
        return False
    
    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def parabola_function(self, x): 
        y = .05 * ((x - 100)*(x - 100))
        return y
    
    def format(self, x, y):
        return f'A{int(x)}{int(y)}'

    def generate_samples(self, data_dir, tmp_dir, dataset_split):

        for i in range(1000):
            x_start = np.random.randint(200)
            inputs = []
            targets = []
            for x in range(x_start, x_start + 10):
                y = self.parabola_function(x)
                inputs.append(self.format(x, y))
            for x in range(x_start + 10, x_start + 20):
                y = self.parabola_function(x)
                targets.append(self.format(x, y))
            print(inputs)
            print(targets)
            yield {
                "inputs": ' '.join(inputs),
                "targets": ' '.join(targets)
            }
