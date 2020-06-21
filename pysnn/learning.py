"""
Approach correlation based learning, gradient-based learning uses PyTorch's learning rules.
    1. Base LR loops over all layers and provides access to presyn neuron, postsyn neuron, and connection objects.
    2. Applies consistent update function to each layer. This method is variable and can use any of the previously metnioned objects.
    3. Implement a hook mechanism that can modulate the update, i.e. for RL based learning rules.
"""

from collections import OrderedDict
import numpy as np
import torch
from .connection import BaseConnection, _Linear, _ConvNd


#########################################################
# Learning rule base class
#########################################################
class LearningRule:
    r"""Base class for correlation based learning rules in spiking neural networks.
    
    :param layers: An iterable or :class:`dict` of :class:`dict` 
        the latter is a dict that contains a :class:`pysnn.Connection` state dict, a pre-synaptic :class:`pysnn.Neuron` state dict, 
        and a post-synaptic :class:`pysnn.Neuron` state dict that together form a single layer. These objects their state's will be 
        used for optimizing weights.
        During initialization of a learning rule that inherits from this class it is supposed to select only the parameters it needs
        from these objects.
        The higher lever iterable or :class:`dict` contain groups that use the same parameter during training. This is analogous to
        PyTorch optimizers parameter groups.
    :param defaults: A dict containing default hyper parameters. This is a placeholder for possible changes later on, these groups would work
        exactly the same as those for PyTorch optimizers.
    """

    def __init__(self, layers, defaults):
        self.layers = layers
        self.defaults = defaults

    def update_state(self, *args, **kwargs):
        r"""Update state parameters of LearningRule based on latest network forward pass."""
        pass

    def reset_state(self, *args, **kwargs):
        r"""Reset state parameters of LearningRule."""
        pass

    def weight_update(self, layer, params, *args, **kwargs):
        raise NotImplementedError("Each learning rule needs an update function")

    def step(self, *args, **kwargs):
        r"""Performs single learning step for each layer."""
        for l in self.layers.values():
            self.weight_update(l, self.defaults, args, kwargs)

    def pre_mult_post(self, pre, post, conn):
        r"""Multiply a presynaptic term with a postsynaptic term, in the following order: pre x post.

        The outcome of this operation preserves batch size, but furthermore is directly broadcastable 
        with the weight of the connection.

        This operation differs for Linear or Convolutional connections. 

        :param pre: Presynaptic term
        :param post: Postsynaptic term
        :param conn: Connection, support Linear and Conv2d

        :return: Tensor broadcastable with the weight of the connection
        """
        # Select target datatype
        if pre.dtype == torch.bool:
            pre = pre.to(post.dtype)
        elif post.dtype == torch.bool:
            post = post.to(pre.dtype)
        elif pre.dtype != post.dtype:
            raise TypeError(
                "The pre and post synaptic terms should either be of the same datatype, or one of them has to be a Boolean."
            )

        # Perform actual multiplication
        # TODO: not sure whether this works correctly for Conv2d connections,
        # cnn_mstdpet_example.py gives an error
        if isinstance(conn, _Linear):
            pre = pre.transpose(2, 1)
        elif isinstance(conn, _ConvNd):
            pre = pre.transpose(2, 1)
            post = post.view(post.shape[0], 1, post.shape[1], -1)
        else:
            if isinstance(conn, BaseConnection):
                raise TypeError(f"Connection type {conn} is not supported.")
            else:
                raise TypeError("Provide an instance of BaseConnection.")

        output = pre * post
        return output.transpose(2, 1)

    def reduce_connections(self, tensor, conn, red_method=torch.mean):
        r"""Reduces the tensor along the dimensions that represent separate connections to an element of the weight Tensor.

        The function used for reducing has to be a callable that can be applied to single axes of a tensor.
        
        This operation differs or Linear or Convolutional connections.
        For Linear, only the batch dimension (dim 0) is reduced.
        For Conv2d, the batch (dim 0) and the number of kernel multiplications dimension (dim 3) are reduced.

        :param tensor: Tensor that will be reduced
        :param conn: Connection, support Linear and Conv2d
        :param red_method: Method used to reduce each dimension

        :return: Reduced Tensor
        """
        if isinstance(conn, _Linear):
            output = red_method(tensor, dim=0)
        elif isinstance(conn, _ConvNd):
            output = red_method(tensor, dim=(0, 3))
        else:
            if isinstance(conn, BaseConnection):
                raise TypeError(f"Connection type {conn} is not supported.")
            else:
                raise TypeError("Provide an instance of BaseConnection.")

        return output


#########################################################
# STDP
#########################################################
class OnlineSTDP(LearningRule):
    r"""Basic online STDP implementation from http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity

    :param layers: OrderedDict containing state dicts for each layer.
    :param lr: Learning rate.
    """

    def __init__(
        self, layers, lr=0.001, a_plus=1.0, a_min=1.0,
    ):
        assert (a_plus >= 0) and (
            a_min >= 0
        ), "'a_plus' and 'a_min' should both be positive."
        params = dict(lr=lr, a_plus=a_plus, a_min=a_min)
        super(OnlineSTDP, self).__init__(layers, params)

    def weight_update(self, layer, params, *args, **kwargs):
        pre_trace, post_trace = layer.presynaptic.trace, layer.postsynaptic.trace
        pre_spike, post_spike = layer.presynaptic.spikes, layer.postsynaptic.spikes
        dw = params["a_plus"] * self.pre_mult_post(
            pre_trace, post_spike, layer.connection
        )
        dw -= params["a_min"] * self.pre_mult_post(
            pre_spike, post_trace, layer.connection
        )
        layer.connection.weight += params["lr"] * self.reduce_connections(
            dw, layer.connection
        )


#########################################################
# MSTDPET
#########################################################
class MSTDPET(LearningRule):
    r"""Apply MSTDPET from (Florian 2007) to the provided connections.
    
    Uses just a single, scalar reward value.
    Update rule can be applied at any desired time step.

    :param layers: OrderedDict containing state dicts for each layer.
    :param a_pre: Scaling factor for presynaptic spikes influence on the eligibility trace.
    :param a_post: Scaling factor for postsynaptic spikes influence on the eligibility trace.
    :param lr: Learning rate.
    :param e_trace_decay: Decay factor for the eligibility trace.
    """

    def __init__(self, layers, lr=0.001, a_pre=1.0, a_post=1.0, e_trace_decay=0.99):
        assert (a_pre >= 0) and (
            a_post >= 0
        ), "'a_pre' and 'a_post' should both be positive."
        params = dict(lr=lr, a_pre=a_pre, a_post=a_post, e_trace_decay=e_trace_decay)
        super(MSTDPET, self).__init__(layers, params)

    def update_state(self, layer, params, *args, **kwargs):
        r"""Update eligibility trace based on pre and postsynaptic spiking activity.
        """
        pre_trace, post_trace = layer.presynaptic.trace, layer.postsynaptic.trace
        pre_spike, post_spike = layer.presynaptic.spikes, layer.postsynaptic.spikes

        # Update eligibility trace
        layer.connection.e_trace *= params["e_trace_decay"]
        de_trace = params["a_pre"] * self.pre_mult_post(
            pre_trace, post_spike, layer.connection
        )
        de_trace -= params["a_post"] * self.pre_mult_post(
            pre_spike, post_trace, layer.connection
        )
        # TODO: reduction here?
        layer.connection.e_trace += de_trace

    def reset_state(self):
        for layer in self.layers.values():
            layer.connection.e_trace.fill_(0)

    def weight_update(self, layer, params, reward, *args, **kwargs):
        # TODO: shape
        layer.connection.weight += (
            params["lr"]
            * reward
            * self.reduce_connections(layer.connection.e_trace, layer.connection)
        )

    def step(self, reward, *args, **kwargs):
        r"""Performs single learning step.
        
        :param reward: Scalar reward value.
        """
        for l in self.layers.values():
            self.update_state(l, self.defaults, args, kwargs)
            self.weight_update(l, self.defaults, reward, args, kwargs)


#########################################################
# Fede STDP
#########################################################
class FedeSTDP(LearningRule):
    r"""STDP version for Paredes Valles, performs mean operation over the batch dimension before weight update.

    Defined in "Unsupervised Learning of a Hierarchical Spiking Neural Network for Optical Flow Estimation: From Events to Global Motion Perception - F.P. Valles, et al."

    :param layers: OrderedDict containing state dicts for each layer.
    :param lr: Learning rate.
    :param w_init: Initialization/reference value for all weights.
    :param a: Stability parameter, a < 1.
    """

    def __init__(self, layers, lr=1e-4, w_init=0.5, a=0):
        assert (a >= 0) and (a <= 1), "For FedeSTDP 'a' should fall between 0 and 1."

        params = dict(lr=lr, w_init=w_init, a=a)
        super(FedeSTDP, self).__init__(layers, params)

    def weight_update(self, layer, params, *args, **kwargs):
        # Normalize presynaptic trace
        # TODO: shape correct?
        # TODO: why doesn't presynaptic trace work? There shouldn't be any difference with connection trace, right?
        # trace = layer.presynaptic.trace.view(-1, *layer.connection.weight.shape)
        trace = layer.connection.trace.view(-1, *layer.connection.weight.shape)
        norm_trace = trace / trace.max()

        # LTP and LTD
        dw = layer.connection.weight - params["w_init"]

        # LTP computation
        ltp_w = torch.exp(-dw)
        ltp_t = torch.exp(norm_trace) - params["a"]
        ltp = ltp_w * ltp_t

        # LTD computation
        ltd_w = -(torch.exp(dw))
        ltd_t = torch.exp(1 - norm_trace) - params["a"]
        ltd = ltd_w * ltd_t

        # Perform weight update
        # TODO: why no reduce here? Is FedeSTDP a special case in combination with Conv2d?
        # layer.connection.weight += params["lr"] * self.reduce_connections(ltp + ltd, layer.connection)
        layer.connection.weight += params["lr"] * (ltp + ltd).mean(0)
