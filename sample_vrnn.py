'''
Variational RNN sample script using TensorFlow
Model introduced in https://arxiv.org/abs/1506.02216

Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A. C., & Bengio, Y. (2015).
A recurrent latent variable model for sequential data.
In Advances in neural information processing systems (pp. 2980-2988).

Code original author : phreeza (taken from https://github.com/phreeza/tensorflow-vrnn)

Author : Anirudh Vemula
Date : December 5th, 2016
'''

import tensorflow as tf

import os
import cPickle
from model_vrnn import VRNN
from utils_vrnn import DataLoader
import numpy as np


def main():
    '''
    Main function
    '''
    # Laod the saved arguments
    with open(os.path.join('save-vrnn', 'config.pkl')) as f:
        saved_args = cPickle.load(f)

    # Initialize the model with the saved arguments in inference mode
    model = VRNN(saved_args, True)
    # Initialize the TensorFlow session
    sess = tf.InteractiveSession()
    # Initialize the saver
    saver = tf.train.Saver(tf.all_variables())

    # Get model checkpoint
    ckpt = tf.train.get_checkpoint_state('save-vrnn')
    print "loading model: ", ckpt.model_checkpoint_path

    # Restore the model from the saved file
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Sample the model
    sample_data, mus, sigmas = model.sample(sess, saved_args)


if __name__ == '__main__':
    main()
