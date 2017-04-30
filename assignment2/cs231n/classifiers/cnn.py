from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        self.conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        self.use_batchnorm = use_batchnorm

        C, H, W = input_dim
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        if self.use_batchnorm:
            self.params['gamma1'] = np.ones(num_filters)
            self.params['beta1'] = np.zeros(num_filters)

        H_prime = 1 + (H + 2*self.conv_param['pad'] - filter_size) // self.conv_param['stride']
        W_prime = 1 + (W + 2*self.conv_param['pad'] - filter_size) // self.conv_param['stride']
        H_pool_prime = (H_prime - self.pool_param['pool_height']) // self.pool_param['stride'] + 1
        W_pool_prime = (W_prime - self.pool_param['pool_width']) // self.pool_param['stride'] + 1
        conv_out_dim = H_pool_prime * W_pool_prime * num_filters

        self.params['W2'] = np.random.normal(scale=weight_scale, size=(conv_out_dim, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        if self.use_batchnorm:
            self.params['gamma2'] = np.ones(hidden_dim)
            self.params['beta2'] = np.zeros(hidden_dim)

        self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        bn_param = [{'mode': 'test' if y is None else 'train'} for i in range(2)]
        cache = {}

        out_conv, cache['conv'] = conv_forward_fast(X, W1, b1, self.conv_param)
        out_conv, cache['conv_relu'] = relu_forward(out_conv)
        if self.use_batchnorm:
            out_conv, cache['bn_conv'] = spatial_batchnorm_forward(
                    out_conv, self.params['gamma1'], self.params['beta1'], bn_param[0])
        out_conv, cache['conv_pool'] = max_pool_forward_fast(out_conv, self.pool_param)

        out_fc, cache['fc'] = affine_forward(out_conv, W2, b2)
        if self.use_batchnorm:
            out_fc, cache['bn_fc'] = batchnorm_forward(
                    out_fc, self.params['gamma2'], self.params['beta2'], bn_param[1])
        out_fc, cache['fc_relu'] = relu_forward(out_fc)

        scores, cache['scores'] = affine_forward(out_fc, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)

        dfc, grads['W3'], grads['b3'] = affine_backward(dscores, cache['scores'])

        dconv = relu_backward(dfc, cache['fc_relu'])
        if self.use_batchnorm:
            dconv, grads['gamma2'], grads['beta2'] = batchnorm_backward(
                    dconv, cache['bn_fc'])
        dconv, grads['W2'], grads['b2'] = affine_backward(dconv, cache['fc'])

        dX = max_pool_backward_fast(dconv, cache['conv_pool'])
        if self.use_batchnorm:
            dX, grads['gamma1'], grads['beta1'] = spatial_batchnorm_backward(
                    dX, cache['bn_conv'])
        dX = relu_backward(dX, cache['conv_relu'])
        dX, grads['W1'], grads['b1'] = conv_backward_fast(dX, cache['conv'])

        loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
        for i in (1, 2, 3):
            grads['W' + str(i)] += self.reg * self.params['W' + str(i)]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
