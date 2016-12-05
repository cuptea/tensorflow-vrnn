'''
Variational RNN utils script using TensorFlow
Model introduced in https://arxiv.org/abs/1506.02216

Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A. C., & Bengio, Y. (2015).
A recurrent latent variable model for sequential data.
In Advances in neural information processing systems (pp. 2980-2988).

Code original author : phreeza (taken from https://github.com/phreeza/tensorflow-vrnn)

Author : Anirudh Vemula
Date : December 5th, 2016
'''

import numpy as np


class DataLoader:

    def __init__(self, args):
        '''
        Initialization function
        Params:
        args : Input arguments
        '''
        # Store arguments
        self.args = args

    def next_batch(self):
        '''
        Function to get the next batch of data
        '''
        # Generate a matrix of random numbers according to the standard normal
        # distribution of shape batch_size x 1 x n_input
        t0 = np.random.randn(self.args.batch_size, 1, (2 * self.args.chunk_samples))
        # Generate a tensor of random numbers according to the standard normal
        # distribution of shape batch_size x seq_length x n_input
        mixed_noise = np.random.randn(self.args.batch_size, self.args.seq_length, (2 * self.args.chunk_samples)) * 0.1

        # input data
        x = np.sin(2 * np.pi * (np.arange(self.args.seq_length)[np.newaxis, :, np.newaxis] / 10. + t0)) + np.random.randn(self.args.batch_size, self.args.seq_length, (2 * self.args.chunk_samples))*0.1 + mixed_noise*0.1

        # target data
        y = np.sin(2 * np.pi * (np.arange(1, self.args.seq_length+1)[np.newaxis, :, np.newaxis] / 10. + t0)) + np.random.randn(self.args.batch_size, self.args.seq_length, (2 * self.args.chunk_samples)) * 0.1 + mixed_noise * 0.1

        y[:, :, self.args.chunk_samples:] = 0.
        x[:, :, self.args.chunk_samples:] = 0.

        return x, y
