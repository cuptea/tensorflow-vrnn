'''
Variational RNN train script using TensorFlow
Model introduced in https://arxiv.org/abs/1506.02216

Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A. C., & Bengio, Y. (2015).
A recurrent latent variable model for sequential data.
In Advances in neural information processing systems (pp. 2980-2988).

Code original author : phreeza (taken from https://github.com/phreeza/tensorflow-vrnn)

Author : Anirudh Vemula
Date : December 4th, 2016
'''

import numpy as np
import tensorflow as tf

import argparse
import glob
import time
from datetime import datetime
import os
import cPickle
import pdb

from model_vrnn import VRNN
from utils_vrnn import DataLoader
from matplotlib import pyplot as plt

'''
TODOS:
    - parameters for depth and width of hidden layers
    - implement predict and sample functions
    - separate binary and gaussian variables
    - clean up nomenclature to remove MDCT references
'''


def main():
    '''
    Main function
    '''
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=3,
                        help='size of RNN hidden state')
    parser.add_argument('--latent_size', type=int, default=3,
                        help='size of latent space')
    parser.add_argument('--batch_size', type=int, default=3000,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.,
                        help='decay of learning rate')
    parser.add_argument('--chunk_samples', type=int, default=1,
                        help='number of samples per mdct chunk')
    args = parser.parse_args()

    # Call the train function
    train(args)


def train(args):
    '''
    The train function
    Params:
    args : Input arguments
    '''
    # Initialize the model
    model = VRNN(args)

    # Initialize the data loader object
    dataloader = DataLoader(args)

    pdb.set_trace()

    # Directory to save the trained model
    dirname = 'save-vrnn'
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Dump the input arguments into the file
    with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)

    # get checkpoint
    ckpt = tf.train.get_checkpoint_state(dirname)

    # Initialize TensorFlow session
    with tf.Session() as sess:
        # Initialize summary writer
        summary_writer = tf.train.SummaryWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
        check = tf.add_check_numerics_ops()
        # Write all summaries
        merged = tf.merge_all_summaries()
        # Initialize all variables in the graph
        tf.initialize_all_variables().run()
        # Initialize a saver for all variables
        saver = tf.train.Saver(tf.all_variables())

        # Load already saved model
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Loaded model"
        # Initialize timer
        start = time.time()

        # For each epoch
        for e in xrange(args.num_epochs):
            # Set the learning rate for this epoch
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            # state = model.initial_state_c, model.initial_state_h

            # For each minibatch
            for b in xrange(100):
                # Get the input and target data of the next minibatch
                x, y = dataloader.next_batch()
                # Create the feed dict
                feed = {model.input_data: x, model.target_data: y}
                # Run the session and get loss
                train_loss, _, cr, summary, sigma, mu, input, target = sess.run(
                    [model.cost, model.train_op, check, merged, model.sigma, model.mu, model.flat_input, model.target],
                    feed)
                # Write summary
                summary_writer.add_summary(summary, e * 100 + b)

                # Save model
                if (e * 100 + b) % args.save_every == 0 and ((e * 100 + b) > 0):
                    checkpoint_path = os.path.join(dirname, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * 100 + b)
                    print "model saved to {}".format(checkpoint_path)

                # End timer
                end = time.time()
                # Print info
                print "{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.1f}, std = {:.3f}" \
                    .format(e * 100 + b,
                            args.num_epochs * 100,
                            e, args.chunk_samples * train_loss, end - start, sigma.mean(axis=0).mean(axis=0))
                start = time.time()

if __name__ == '__main__':
    main()
