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
    def __init__(self, args, sample=False):

        def tf_normal(y, mu, s, rho):
            with tf.variable_scope('normal'):
                ss = tf.maximum(1e-10,tf.square(s))
                norm = tf.sub(y[:,:args.chunk_samples], mu)
                z = tf.div(tf.square(norm), ss)
                denom_log = tf.log(2*np.pi*ss, name='denom_log')
                result = tf.reduce_sum(z+denom_log, 1)/2# -
                                       #(tf.log(tf.maximum(1e-20,rho),name='log_rho')*(1+y[:,args.chunk_samples:])
                                       # +tf.log(tf.maximum(1e-20,1-rho),name='log_rho_inv')*(1-y[:,args.chunk_samples:]))/2, 1)

            return result

        def tf_kl_gaussgauss(mu_1, sigma_1, mu_2, sigma_2):
            with tf.variable_scope("kl_gaussgauss"):
                return tf.reduce_sum(0.5 * (
                    2 * tf.log(tf.maximum(1e-9,sigma_2),name='log_sigma_2') 
                  - 2 * tf.log(tf.maximum(1e-9,sigma_1),name='log_sigma_1')
                  + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9,(tf.square(sigma_2))) - 1
                ), 1)

        def get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, prior_mu, prior_sigma, y):
            kl_loss = tf_kl_gaussgauss(enc_mu, enc_sigma, prior_mu, prior_sigma)
            likelihood_loss = tf_normal(y, dec_mu, dec_sigma, dec_rho)

            return tf.reduce_mean(kl_loss + likelihood_loss)
            #return tf.reduce_mean(likelihood_loss)

        self.args = args
        if sample:
            args.batch_size = 1
            args.seq_length = 1

        cell = VartiationalRNNCell(args.chunk_samples, args.rnn_size, args.latent_size)

        self.cell = cell

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 2*args.chunk_samples], name='input_data')
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 2*args.chunk_samples],name = 'target_data')
        self.initial_state_c, self.initial_state_h = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)


        # input shape: (batch_size, n_steps, n_input)
        with tf.variable_scope("inputs"):
            inputs = tf.transpose(self.input_data, [1, 0, 2])  # permute n_steps and batch_size
            inputs = tf.reshape(inputs, [-1, 2*args.chunk_samples]) # (n_steps*batch_size, n_input)

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            inputs = tf.split(0, args.seq_length, inputs) # n_steps * (batch_size, n_hidden)
        flat_target_data = tf.reshape(self.target_data,[-1, 2*args.chunk_samples])

        self.target = flat_target_data
        self.flat_input = tf.reshape(tf.transpose(tf.pack(inputs),[1,0,2]),[args.batch_size*args.seq_length, -1])
        self.input = tf.pack(inputs)
        # Get vrnn cell output
        outputs, last_state = tf.nn.rnn(cell, inputs, initial_state=(self.initial_state_c,self.initial_state_h))
        #print outputs
        #outputs = map(tf.pack,zip(*outputs))
        outputs_reshape = []
        names = ["enc_mu", "enc_sigma", "dec_mu", "dec_sigma", "dec_rho", "prior_mu", "prior_sigma"]
        for n,name in enumerate(names):
            with tf.variable_scope(name):
                x = tf.pack([o[n] for o in outputs])
                x = tf.transpose(x,[1,0,2])
                x = tf.reshape(x,[args.batch_size*args.seq_length, -1])
                outputs_reshape.append(x)

        enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, prior_mu, prior_sigma = outputs_reshape
        self.final_state_c,self.final_state_h = last_state
        self.mu = dec_mu
        self.sigma = dec_sigma
        self.rho = dec_rho

        lossfunc = get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_sigma, prior_mu, prior_sigma, flat_target_data)
        self.sigma = dec_sigma
        self.mu = dec_mu
        with tf.variable_scope('cost'):
            self.cost = lossfunc 
        tf.scalar_summary('cost', self.cost)
        tf.scalar_summary('mu', tf.reduce_mean(self.mu))
        tf.scalar_summary('sigma', tf.reduce_mean(self.sigma))


        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        for t in tvars:
            print t.name
        grads = tf.gradients(self.cost, tvars)
        #grads = tf.cond(
        #    tf.global_norm(grads) > 1e-20,
        #    lambda: tf.clip_by_global_norm(grads, args.grad_clip)[0],
        #    lambda: grads)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        #self.saver = tf.train.Saver(tf.all_variables())

    def sample(self, sess, args, num=4410, start=None):

        def sample_gaussian(mu, sigma):
            return mu + (sigma*np.random.randn(*sigma.shape))

        if start is None:
            prev_x = np.random.randn(1, 1, 2*args.chunk_samples)
        elif len(start.shape) == 1:
            prev_x = start[np.newaxis,np.newaxis,:]
        elif len(start.shape) == 2:
            for i in range(start.shape[0]-1):
                prev_x = start[i,:]
                prev_x = prev_x[np.newaxis,np.newaxis,:]
                feed = {self.input_data: prev_x,
                        self.initial_state_c:prev_state[0],
                        self.initial_state_h:prev_state[1]}
                
                [o_mu, o_sigma, o_rho, prev_state_c, prev_state_h] = sess.run(
                        [self.mu, self.sigma, self.rho,
                         self.final_state_c,self.final_state_h],feed)

            prev_x = start[-1,:]
            prev_x = prev_x[np.newaxis,np.newaxis,:]

        prev_state = sess.run(self.cell.zero_state(1, tf.float32))
        chunks = np.zeros((num, 2*args.chunk_samples), dtype=np.float32)
        mus = np.zeros((num, args.chunk_samples), dtype=np.float32)
        sigmas = np.zeros((num, args.chunk_samples), dtype=np.float32)

        for i in xrange(num):
            feed = {self.input_data: prev_x,
                    self.initial_state_c:prev_state[0],
                    self.initial_state_h:prev_state[1]}
            [o_mu, o_sigma, o_rho, next_state_c, next_state_h] = sess.run([self.mu, self.sigma,
                self.rho, self.final_state_c, self.final_state_h],feed)

            next_x = np.hstack((sample_gaussian(o_mu, o_sigma),
                                2.*(o_rho > np.random.random(o_rho.shape[:2]))-1.))
            chunks[i] = next_x
            mus[i] = o_mu
            sigmas[i] = o_sigma

            prev_x = np.zeros((1, 1, 2*args.chunk_samples), dtype=np.float32)
            prev_x[0][0] = next_x
            prev_state = next_state_c, next_state_h

        return chunks, mus, sigmas
