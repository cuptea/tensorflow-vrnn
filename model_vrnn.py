'''
Variational RNN model using TensorFlow
Model introduced in https://arxiv.org/abs/1506.02216

Chung, J., Kastner, K., Dinh, L., Goel, K., Courville, A. C., & Bengio, Y. (2015).
A recurrent latent variable model for sequential data.
In Advances in neural information processing systems (pp. 2980-2988).

Code original author : phreeza (taken from https://github.com/phreeza/tensorflow-vrnn)

Author : Anirudh Vemula
Date : December 4th, 2016
'''

import tensorflow as tf
import numpy as np


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    '''
    Function that defines a linear NN layer that just performs W*x + b
    Params:
    input_ : The input data
    output_size : Dimensions of the output
    scope (optional) : Variable scope
    stddev (optional) : Standard deviation of the normal distribution for weight initialization
    bias_start (optional) : Constant to initialize the bias vector with
    with_w (optional) : Return weights with the output or not
    '''
    # Get shape of the input
    # [num_inputs, input_dimension]
    shape = input_.get_shape().as_list()

    # Define variable scope
    with tf.variable_scope(scope or "Linear"):
        # Initialize W
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        # Initialize b
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        # Return weights with output?
        if with_w:
            return tf.add(tf.matmul(input_, matrix), bias), matrix, bias
        else:
            return tf.add(tf.matmul(input_, matrix), bias)


class VartiationalRNNCell(tf.nn.rnn_cell.RNNCell):
    '''
    Class to define a Variational RNN cell
    Inherits TensorFlow RNNCell
    '''

    def __init__(self, x_dim, h_dim, z_dim=100):
        '''
        Initialization function
        Params:
        x_dim : Dimensions of the input data
        h_dim : Dimensions of the hidden state
        z_dim : Dimensions of the latent variable
        '''
        # Store dimensions of data
        self.n_h = h_dim
        self.n_x = x_dim
        self.n_z = z_dim

        # Store dimensions of extracted features from x and z
        self.n_x_1 = x_dim
        self.n_z_1 = z_dim

        # Dimensions of the encoder, decoder and prior networks
        self.n_enc_hidden = z_dim
        self.n_dec_hidden = x_dim
        self.n_prior_hidden = z_dim

        # LSTM cell with the hidden state dimension, as given
        self.lstm = tf.nn.rnn_cell.LSTMCell(self.n_h, state_is_tuple=True)

    @property
    def state_size(self):
        '''
        Returns the hidden state dimensions of the VRNNCell
        Since state_is_tuple is True, we have a tuple
        '''
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        '''
        Returns the output dimensions of the VRNNCell
        '''
        return self.n_h

    def __call__(self, x, state, scope=None):
        '''
        A single step of the VRNN Cell
        Params:
        x : input data
        state : Current hidden state of the VRNNCell
        scope (optional) : Variable scope
        '''
        # Define variable scope
        with tf.variable_scope(scope or type(self).__name__):
            # Get the hidden and cell state from the input state
            h, c = state

            # Prior variable scope
            with tf.variable_scope("Prior"):
                # A ReLU nonlinear layer on top of the hidden state h
                # resulting in an output of the same dimension as z
                with tf.variable_scope("hidden"):
                    prior_hidden = tf.nn.relu(linear(h, self.n_prior_hidden))

                # A linear layer on top of the previous layer output to get mu
                with tf.variable_scope("mu"):
                    prior_mu = linear(prior_hidden, self.n_z)

                # A softplus nonlinear layer on top of prior_hidden to get sigma
                with tf.variable_scope("sigma"):
                    prior_sigma = tf.nn.softplus(linear(prior_hidden, self.n_z))

            # A ReLU nonlinear layer on top of input data x to extract relevant features
            # of the same dimension as x
            with tf.variable_scope("phi_x"):
                x_1 = tf.nn.relu(linear(x, self.n_x_1))

            # Encoder
            with tf.variable_scope("Encoder"):
                # A ReLU nonlinear layer on top of concatentation of input data features and hidden state
                # resulting in an output of the same dimensions as z
                with tf.variable_scope("hidden"):
                    enc_hidden = tf.nn.relu(linear(tf.concat(1, (x_1, h)), self.n_enc_hidden))

                # A linear layer on top of the previous layer to get mu
                with tf.variable_scope("mu"):
                    enc_mu = linear(enc_hidden, self.n_z)

                # A softplus nonlinear layer on top of enc_hidden to get sigma
                with tf.variable_scope("sigma"):
                    enc_sigma = tf.nn.softplus(linear(enc_hidden, self.n_z))

            # Sample the auxiliary variable epsilon from a standard normal distribution
            # epsilon is of shape (num_inputs, dimension_of_z)
            eps = tf.random_normal((x.get_shape().as_list()[0], self.n_z), 0.0, 1.0, dtype=tf.float32)

            # Reparameterization trick. Get value of z
            # z = mu + sigma*epsilon
            z = tf.add(enc_mu, tf.mul(enc_sigma, eps))

            # A ReLU nonlinear layer on top of latent variable z to extract relevant features
            # of the same dimension as z
            with tf.variable_scope("phi_z"):
                z_1 = tf.nn.relu(linear(z, self.n_z_1))

            # Decoder
            with tf.variable_scope("Decoder"):
                # A ReLU nonlinear layer on top of concatenation of latent variable features and hidden state
                # resulting in an output of the same dimensions as x
                with tf.variable_scope("hidden"):
                    dec_hidden = tf.nn.relu(linear(tf.concat(1, (z_1, h)), self.n_dec_hidden))

                # A linear layer on top of the previous layer to get mu
                with tf.variable_scope("mu"):
                    dec_mu = linear(dec_hidden, self.n_x)

                # A softplus nonlinear layer on top of dec_hidden to get sigma
                with tf.variable_scope("sigma"):
                    dec_sigma = tf.nn.softplus(linear(dec_hidden, self.n_x))

                # A sigmoid nonlinear layer on top of dec_hidden to get rho (correlation?)
                # NOTE not proposed in paper, but makes sense
                with tf.variable_scope("rho"):
                    dec_rho = tf.nn.sigmoid(linear(dec_hidden, self.n_x))

            # Do one step of LSTM with input as concatenation of the input data features and latent variable features
            output, state2 = self.lstm(tf.concat(1, (x_1, z_1)), state)

        # Return all learnt parameters and the LSTM final state
        return (enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, prior_mu, prior_sigma), state2


class VRNN():
    '''
    Class for the VRNN network
    '''
    def __init__(self, args, sample=False):
        '''
        Initialization function
        Params:
        args : Arguments for the model
        sample : Training/inference mode
        '''
        # Helper functions

        def tf_normal(y, mu, s, rho):
            '''
            Computes the log likelihood w.r.t Gaussian distribution
            Params:
            y : input data
            mu : mean
            s : sigma
            rho : correlation
            formula: L(mu,sigma;y) 
            			= -(n/2)* ln(2*pi) - (n/2)*ln(sigma^2) - (1/ (2*sigma^2))* reduce_sum((y-mu)^2)
               			= -(1/2)* reduce_sum( ln(2*pi*sigma^2) +. (y-mu)^2 / sigma^2 ) # "+." means element-wise add

            '''
            # Define variable scope
            with tf.variable_scope('normal'):
                # Compute sigma squared (or variance)
                ss = tf.maximum(1e-10, tf.square(s))
                # Compute (y - mu)
                norm = tf.sub(y[:, :args.chunk_samples], mu)
                # Compute (y-mu)^2/sigma^2
                z = tf.div(tf.square(norm), ss)
                # Compute denominator log(2*pi*sigma^2)
                denom_log = tf.log(2*np.pi*ss, name='denom_log')
                # Compute (1/2)* reduce_sum( ln(2*pi*sigma^2) +. (y-mu)^2 / sigma^2 ) # "+." means element-wise add
                # Please note that the return is -L(mu,sigma;y)
                result = tf.reduce_sum(z+denom_log, 1)/2
                # -
                # (tf.log(tf.maximum(1e-20,rho),name='log_rho')*(1+y[:,args.chunk_samples:])
                # +tf.log(tf.maximum(1e-20,1-rho),name='log_rho_inv')*(1-y[:,args.chunk_samples:]))/2, 1)

            return result

        def tf_kl_gaussgauss(mu_1, sigma_1, mu_2, sigma_2):
            '''
            Function to compute the KL-divergence between two Gaussian
            distributions
            Params:
            mu_1, sigma_1 : mean and std dev of first distribution
            mu_2, sigma_2 : mean and std dev of second distribution
            '''
            # Define variable scope
            with tf.variable_scope("kl_gaussgauss"):
                # Compute the KL-divergence term given in
                # Auto-encoding VB paper eqn. 10
                # KL(mu_1,sigma_1,mu_2,sigma_2) = log(sigma_2/sigma_1) + (sigma_1^2 + (mu_1 - mu_2)^2) / ( 2 * sigma_2^2 ) - 0.5
                # the derivation can be found here: http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
                return tf.reduce_sum(0.5 * (
                    2 * tf.log(tf.maximum(1e-9, sigma_2), name='log_sigma_2')
                    - 2 * tf.log(tf.maximum(1e-9, sigma_1), name='log_sigma_1')
                    + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9, (tf.square(sigma_2))) - 1
                ), 1)

        def get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, prior_mu, prior_sigma, y):
            '''
            Function to compute loss given the predicted parameters and the true data
            Params:
            enc_mu, enc_sigma : mean and stddev for encoder
            dec_mu, dec_sigma : mean and stddev for decoder
            dec_rho : correlation for decoder
            prior_mu, prior_sigma : mean and stddev for prior distribution
            y : true data
            '''
            # Compute KL divergence between encoder distribution and the prior distribution
            kl_loss = tf_kl_gaussgauss(enc_mu, enc_sigma, prior_mu, prior_sigma)
            # Compute the log likelihood of the true data w.r.t the decoder distribution (reconstruction error)
            likelihood_loss = tf_normal(y, dec_mu, dec_sigma, dec_rho)

            # Add both the losses to get the final loss
            return tf.reduce_mean(kl_loss + likelihood_loss)
            # return tf.reduce_mean(likelihood_loss)

        # Store input arguments
        self.args = args

        # If in inference mode, then batch size and sequence length is 1
        if sample:
            args.batch_size = 1
            args.seq_length = 1

        # Define the VRNNCell
        # TODO what is args.chunk_samples?
        cell = VartiationalRNNCell(args.chunk_samples, args.rnn_size, args.latent_size)

        # Store the cell
        self.cell = cell

        # Define placeholders for the input, target data
        # input_data would be of size batch_size x seq_length x (2*chunk_samples)
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 2*args.chunk_samples], name='input_data')

        # target_data would be of size batch_size x seq_length x (2*chunk_samples)
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 2*args.chunk_samples], name='target_data')

        # Initialize the cell state and the hidden state of the VRNNCell (remember state_is_tuple is true)
        self.initial_state_c, self.initial_state_h = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)

        # input shape: (batch_size, n_steps, n_input)
        with tf.variable_scope("inputs"):
            # Permute the input data to be of shape (seq_length x batch_size x n_input)
            inputs = tf.transpose(self.input_data, [1, 0, 2])
            # Reshape so that the shape is ((seq_length*batch_size) x n_input)
            inputs = tf.reshape(inputs, [-1, 2*args.chunk_samples])

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            inputs = tf.split(0, args.seq_length, inputs)  # n_steps * (batch_size, n_input)

        # Flatten the target data to be of shape ((batch_size*seq_length) x n_input)
        flat_target_data = tf.reshape(self.target_data, [-1, 2*args.chunk_samples])

        # Store flattened target data
        self.target = flat_target_data

        # Flatten input data in the same way as target_data was flattened
        self.flat_input = tf.reshape(tf.transpose(tf.pack(inputs), [1, 0, 2]), [args.batch_size*args.seq_length, -1])

        # Store the list of inputs
        self.input = tf.pack(inputs)

        # Get vrnn cell output
        outputs, last_state = tf.nn.rnn(cell, inputs, initial_state=(self.initial_state_c, self.initial_state_h))

        # print outputs
        # outputs = map(tf.pack,zip(*outputs))

        outputs_reshape = []
        names = ["enc_mu", "enc_sigma", "dec_mu", "dec_sigma", "dec_rho", "prior_mu", "prior_sigma"]
        for n, name in enumerate(names):
            with tf.variable_scope(name):
                # Pack the list of values into a tensor
                x = tf.pack([o[n] for o in outputs])
                # Permute so that it is of shape batch_size x seq_length x n_input
                x = tf.transpose(x, [1, 0, 2])
                # Convert it into the shape of flattened data
                x = tf.reshape(x, [args.batch_size*args.seq_length, -1])
                # Append to list of outputs
                outputs_reshape.append(x)

        # Extract tensors of predicted parameters
        enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, prior_mu, prior_sigma = outputs_reshape

        # Store the final cell and hidden state of the VRNNCell
        self.final_state_c, self.final_state_h = last_state

        # Store the decoder parameters
        self.mu = dec_mu
        self.sigma = dec_sigma
        self.rho = dec_rho

        # Compute the loss
        lossfunc = get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_sigma, prior_mu, prior_sigma, flat_target_data)

        # Store the loss as cost
        with tf.variable_scope('cost'):
            self.cost = lossfunc

        tf.scalar_summary('cost', self.cost)
        tf.scalar_summary('mu', tf.reduce_mean(self.mu))
        tf.scalar_summary('sigma', tf.reduce_mean(self.sigma))

        # Learning rate
        self.lr = tf.Variable(0.0, trainable=False)
        # Get all trainable variables
        tvars = tf.trainable_variables()

        # Print the names of all trainable variables
        for t in tvars:
            print t.name

        # Compute gradients for all trainable variables
        grads = tf.gradients(self.cost, tvars)
        # grads = tf.cond(
        #    tf.global_norm(grads) > 1e-20,
        #    lambda: tf.clip_by_global_norm(grads, args.grad_clip)[0],
        #    lambda: grads)

        # Define the optimizer
        optimizer = tf.train.AdamOptimizer(self.lr)

        # Define the train operator and apply gradients
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        # self.saver = tf.train.Saver(tf.all_variables())

    def sample(self, sess, args, num=4410, start=None):
        '''
        Function to do inference in the model
        Params:
        sess: TensorFlow session
        args: input arguments
        num: Number of steps to be predicted
        start (optional): Data to start with
        '''
        # a helper function
        def sample_gaussian(mu, sigma):
            '''
            Function to get a sample from a Gaussian distribution
            Params:
            mu : mean
            sigma : stddev
            '''
            return mu + (sigma*np.random.randn(*sigma.shape))

        # Initialize the state of the VRNNCell
        prev_state = sess.run(self.cell.zero_state(1, tf.float32))
        # If no start data
        if start is None:
            # Initialize with a random input of seq_length 1 and dimensions n_input
            prev_x = np.random.randn(1, 1, 2*args.chunk_samples)
        # If start data is a vector
        elif len(start.shape) == 1:
            # Reshape it to size 1 x 1 x n_input
            prev_x = start[np.newaxis, np.newaxis, :]
        # If start data is a matrix
        elif len(start.shape) == 2:
            # For all time-steps until the last
            for i in range(start.shape[0]-1):
                # Get the input data
                prev_x = start[i, :]
                # Reshape it to 1 x 1 x n_input
                prev_x = prev_x[np.newaxis, np.newaxis, :]

                # Construct the feed dict
                feed = {self.input_data: prev_x,
                        self.initial_state_c: prev_state[0],
                        self.initial_state_h: prev_state[1]}

                # Run session and get the predicted parameters and final state
                [o_mu, o_sigma, o_rho, prev_state_c, prev_state_h] = sess.run(
                        [self.mu, self.sigma, self.rho,
                         self.final_state_c, self.final_state_h], feed)

                # Update the state
                prev_state[0] = prev_state_c
                prev_state[1] = prev_state_h

            # Store the last time-step input
            prev_x = start[-1, :]
            # Reshape it to shape 1 x 1 x n_input
            prev_x = prev_x[np.newaxis, np.newaxis, :]

        # Matrices to store predicted parameters and data
        chunks = np.zeros((num, 2*args.chunk_samples), dtype=np.float32)
        mus = np.zeros((num, args.chunk_samples), dtype=np.float32)
        sigmas = np.zeros((num, args.chunk_samples), dtype=np.float32)

        # For each time-step at prediction time
        for i in xrange(num):
            # Construct the feed dict
            feed = {self.input_data: prev_x,
                    self.initial_state_c: prev_state[0],
                    self.initial_state_h: prev_state[1]}
            # Run session and get the predicted parameters and final state
            [o_mu, o_sigma, o_rho, next_state_c, next_state_h] = sess.run([self.mu, self.sigma,
                                                                           self.rho, self.final_state_c, self.final_state_h], feed)

            # Sample from the predicted gaussian to get the next input
            # TODO What is this? What is the second half of each row?
            next_x = np.hstack((sample_gaussian(o_mu, o_sigma),
                                2.*(o_rho > np.random.random(o_rho.shape[:2]))-1.))

            # Store the predicted data and parameters
            chunks[i] = next_x
            mus[i] = o_mu
            sigmas[i] = o_sigma

            # Construct the data for next time step
            prev_x = np.zeros((1, 1, 2*args.chunk_samples), dtype=np.float32)
            prev_x[0][0] = next_x
            # Update state
            prev_state = next_state_c, next_state_h

        # return predicted data and parameters
        return chunks, mus, sigmas
