"""
MIT licence 

Copyright 2019 XX XX

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np

import uuid
import numbers
import operator

from types import SimpleNamespace
from collections import namedtuple


# if there are multiple path to a node
# we must add the grad values 
def combine_dicts(a, b, op=operator.add):
    x = (list(a.items()) + list(b.items()) +
        [(k, op(a[k], b[k])) for k in set(b) & set(a)])
    return {x[i][0]: x[i][1] for i in range(0, len(x))}

class Node(object):
    
    NodeStore = dict()

    def _set_variables(self, name, value):
        self.value = value
        self.shape = value.shape
        self.dtype = value.dtype
        self.name = name
        self.uuid = uuid.uuid4()
        if name:
            self.grad = lambda g: {name: g}
            self._register()
        else:
            self.grad = lambda g: {}

    def _test_shape(self):
        # only 2D-Arrays are supported at moment
        assert len(self.shape) == 2


    def __init__(self, value, name=None):

        # wrap numbers in a numpy 2D-Array
        if isinstance(value, numbers.Number):
            value = np.array([[value]])
        assert isinstance(value, np.ndarray)
        if len(value.shape) == 1:
            value = value.reshape(-1, 1)

        self._set_variables(name, value)
        self._test_shape()

    def _register(self):
        Node.NodeStore[self.name] = self

    @staticmethod
    def get_param():
        param = dict()
        for n in Node.NodeStore:
            param[n] = Node.NodeStore[n].value
        return param

    @staticmethod
    def set_param(param):
        for n in Node.NodeStore:
            Node.NodeStore[n].value = param[n]
        
    def _broadcast_g_helper(o, o_):# broadcasting make things slightly more complicated
        if o_.shape[0] > o.shape[0]:
            o_ = o_.sum(axis=0).reshape(1,-1)
        if o_.shape[1] > o.shape[1]:
            o_ = o_.sum(axis=1).reshape(-1,1)
        return o_    
                  
    def _set_grad(self_, g_total_self, other, g_total_other):
        g_total_self = Node._broadcast_g_helper(self_, g_total_self)
        x = self_.grad(g_total_self)
        g_total_other = Node._broadcast_g_helper(other, g_total_other)
        x = combine_dicts(x, other.grad(g_total_other))
        return x
                
    def __add__(self, other):
        if isinstance(other, numbers.Number):
            other = Node(np.array([[other]]))
        ret = Node(self.value + other.value)
        def grad(g):
            g_total_self = g
            g_total_other = g
            x = Node._set_grad(self, g_total_self, other, g_total_other)
            return x
        ret.grad = grad
        return ret
    
    def __radd__(self, other):
        return Node(other) + self
    
    def __sub__(self, other):
        if isinstance(other, numbers.Number):
            other = Node(other)
        ret = self + (other * -1.)   
        return ret
    
    def __rsub__(self, other):
        if isinstance(other, numbers.Number):
            other = Node(other) 
            return other - self
        raise NotImplementedError()
        
    def __mul__(self, other):
        if isinstance(other, numbers.Number) or isinstance(other, np.ndarray):
            other = Node(other)           
        ret = Node(self.value * other.value) 
        def grad(g):
            g_total_self = g * other.value
            g_total_other = g * self.value
            x = Node._set_grad(self, g_total_self, other, g_total_other)
            return x
        ret.grad = grad
        return ret      
    
    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return Node(other) * self
        raise NotImplementedError()

    def concatenate(self, other, axis=0):
        assert axis in (0,1) # TODO
        ret = Node(np.concatenate((self.value, other.value), axis=axis))
        def grad(g):
            if axis == 0: 
                g_total_self = g[:self.shape[0]] 
                g_total_other = g[self.shape[0]:]
            elif axis == 1:
                g_total_self = g[:, :self.shape[1]] 
                g_total_other = g[:, self.shape[1]:]
            x = Node._set_grad(self, g_total_self, other, g_total_other)
            return x
        ret.grad = grad
        return ret
    
    # slicing
    def __getitem__(self, val):
         raise NotImplementedError()

        
    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            other = Node(np.array([[other]]))
        ret = Node(self.value / other.value)

        def grad(g):
            g_total_self = g / other.value
            g_total_other = -1. * self.value * g / (other.value**2)
            x = Node._set_grad(self, g_total_self, other, g_total_other)
            return x
        ret.grad = grad
        return ret
    
    def __rtruediv__(self, other):
        if isinstance(other, numbers.Number):
            other = Node(other)
            return other/self
        raise NotImplementedError()
        
    def __neg__(self):
        return self * -1.
    
    def dot(self, other):
        ret = Node(np.dot(self.value, other.value))    
        def grad(g):
            g_total_self = np.dot(g, other.value.T)
            g_total_other = np.dot(self.value.T, g)
            x = Node._set_grad(self, g_total_self, other, g_total_other)
            return x
        ret.grad = grad
        return ret
    
    def transpose(self):
        ret = Node(self.value.T)
        def grad(g):
            x = self.grad(g.T)
            return x
        ret.grad = grad
        return ret    

    T = transpose

    def exp(self):
        ret = Node(np.exp(self.value))
        def grad(g):
            assert self.shape == g.shape
            x = self.grad(np.exp(self.value) * g)
            return x
        ret.grad = grad
        return ret    
    
    def log(self):
        ret = Node(np.log(self.value))
        def grad(g):
            assert self.shape == g.shape
            x = self.grad(1./self.value * g)
            return x
        ret.grad = grad
        return ret     
       
    def square(self):
        ret = Node(np.square(self.value))
        def grad(g):
            assert self.shape == g.shape
            x = self.grad(2 * self.value * g)
            return x
        ret.grad = grad
        return ret       
         
    def sqrt(self):
        ret = Node(np.sqrt(self.value))
        def grad(g):
            assert self.shape == g.shape
            x = self.grad(0.5 * (1/np.sqrt(self.value)) * g)
            return x
        ret.grad = grad
        return ret     
        
    def sum(self, axis=None):
        if axis is None:
            return self._sum_all()
        assert axis in (0,1)
        return self._sum(axis)
        
    def _sum_all(self):
        ret = Node(np.sum(self.value).reshape(1,1))
        def grad(g):
            x = self.grad(np.ones_like(self.value) * g)
            return x
        ret.grad = grad
        return ret
    
    def _sum(self, axis):
        ret = self.value.sum(axis=axis)
        if axis==0: 
            ret = ret.reshape(1, -1)
        else:
            ret = ret.reshape(-1, 1)
        ret = Node(ret)
        def grad(g):
            x = self.grad(np.ones_like(self.value) * g)
            return x
        ret.grad = grad
        return ret  
    
    def relu(self):
        self.mask = self.value > 0.
        ret = Node(self.mask * self.value)
        def grad(g):
            assert self.shape == g.shape
            x = self.grad(self.mask * g)
            return x
        ret.grad = grad
        return ret    
    
    def softmax(self):
        ret = self.exp() / self.exp().sum(axis=1)
        np.testing.assert_almost_equal(ret.value.sum(axis=1), 1.)
        return ret
    
    def sigmoid(self):
        return 1./(1. + self.exp())

    def reshape(self, shape):
        ret = Node(self.value.reshape(shape))
        def grad(g):
            x = self.grad(g.reshape(self.shape))
            return x
        ret.grad = grad
        return ret

class NeuralNode(object):
     
    _sep = "_"
    #param = dict()
    #def register():
    #    NeuralNode.param = Node.get_param()

    @staticmethod
    def _get_fullname(name, suffix):
        return NeuralNode._sep.join([name, suffix])

    @staticmethod
    def _initialize_W(fan_in, fan_out):
        gain = np.sqrt(2)
        std = gain / np.sqrt(fan_in)
        bound = np.sqrt(3) * std 
        return np.random.uniform(-bound, bound, size=(fan_in, fan_out))

    @staticmethod
    def _initialize_b(fan_in, fan_out):
        bound = 1 / np.sqrt(fan_in)
        return np.random.uniform(-bound, bound, size=(1, fan_out))    

    @staticmethod
    def get_name_and_set_param(param, layer_name, ext_name):
        node_name = NeuralNode._get_fullname(layer_name, ext_name)
        node_value = param.get(node_name)
        return node_name, node_value


    def _Linear_Layer(fan_in, fan_out, name=None, param=dict()):

        assert isinstance(name, str) 
        weight_name, W_value = NeuralNode.get_name_and_set_param(param, name, "weight")
        bias_name, b_value = NeuralNode.get_name_and_set_param(param, name, "bias")
        
        assert (W_value is None and b_value is None) or (W_value is not None and b_value is not None)
        if W_value is None:
            W_value = NeuralNode._initialize_W(fan_in, fan_out)
            b_value = NeuralNode._initialize_b(fan_in, fan_out)
            param[weight_name] = W_value
            param[bias_name] = b_value
            
        W = Node(W_value, weight_name)
        b = Node(b_value, bias_name)
        nodes = dict()  # type: Dict[str, Node]
        nodes[weight_name] = W
        nodes[bias_name] = b
        return lambda X: (X.dot(W) + b), param, nodes
    
    def _ReLu_Layer(fan_in, fan_out, name=None, param=dict()):
        ll, param, nodes = NeuralNode._Linear_Layer(fan_in, fan_out, name, param)
        f = lambda X: ll(X).relu()
        return f, param, nodes

# short namespace name        
nn = NeuralNode 


class Model(object):  

        # nodes are neural nodes
        _NNode = namedtuple("NNode", ['layer_type', 'param', 'nodes'])  # type: Type[NNode]
        
        def __init__(self):
            self.neural_nodes = dict()
            
        # params are not structured in nodes
        def get_param(self):
            param = dict()
            for node_name, node_value in self.neural_nodes.items():
                param = {**param, **node_value.param} # merge dicts
            return param
        
        def set_param(self, param):
            for node_name, node_value in self.neural_nodes.items():
                for param_name in node_value.param:
                    node_value.param[param_name] = param[param_name]
            
        def get_grad(self, x, y):
            loss_ = self.loss(x, y)
            g = np.ones_like(loss_.value)
            return loss_.grad(g), loss_
        
        
        def _set_layer(self, fan_in, fan_out, name, layer_type):
            assert isinstance(name, str)
            assert name not in self.neural_nodes.keys()
            f, param, nodes = layer_type(fan_in, fan_out, name=name, param=dict())
            self.neural_nodes[name] = Model._NNode(layer_type=layer_type, param=param, nodes=nodes)
            return lambda x: f(x)

        def ReLu_Layer(self, fan_in, fan_out, name=None):
            return self._set_layer(fan_in, fan_out, name, NeuralNode._ReLu_Layer)
        
        def Linear_Layer(self, fan_in, fan_out, name=None):
            return self._set_layer(fan_in, fan_out, name, NeuralNode._Linear_Layer)
        
        def _add_to_params(self, p):
            assert p.keys().isdisjoint(self.params.keys())
            self.param = {**self.param, **p} # merge dicts
 
        # following methods     
        # must be implemented by subclasses
        
        def forward(self, x):
            raise NotImplementedError()
    
        def loss(self, x, y):
            raise NotImplementedError()
            
        # example: cross entropy 
        # y should be one hot encoded
        #def loss(self, x, y=None):
        #    n = x.shape[0]
        #    logits = self.logits(x) # has be be defined
        #    # numeric stable max sum exponent   
        #    max_logit = Node(np.max(logits.value, axis=1))
        #    log_softmax = logits - max_logit - (logits - max_logit).exp().sum(axis=1).log()
        #    loss = - y * log_softmax
        #    return loss.sum()/n

##################################################################################################


class Optimizer(object):
    
    def __init__(self, model, x_train=None, y_train=None, hyperparam=dict(), batch_size=128):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size=batch_size
        self.hyperparam = hyperparam
        self._set_param()
        self.grad_stores = [] # list of dicts for momentum, etc. 
        
    def random_batch(self):
        n = self.x_train.shape[0]
        indices = np.random.randint(0, n, size=self.batch_size)
        return Node(self.x_train[indices]), Node(self.y_train[indices])

    def train(self, steps=1000, print_each=100):
        raise NotImplementedError()

    def _train(self, steps=1000, num_grad_stores=0, print_each=100):
        assert num_grad_stores in (0,1,2)
        model = self.model
        if num_grad_stores>0:
            x, y = self.random_batch()
            grad, loss = model.get_grad(x, y)
            self.grad_stores = num_grad_stores * [dict()]
        for grad_store in self.grad_stores:
            for g in grad:
                grad_store[g] = np.zeros_like(grad[g])
                
        param = model.get_param()
        print("iteration\tloss")    
        for i in range(1, steps+1):
            x, y = self.random_batch()
            grad, loss = model.get_grad(x, y)
            if i%print_each==0 or i==1:
                print(i, "\t",loss.value[0,0])       
                
            for g in grad:
                #assert param[g].shape == self.grad_stores[0][g].shape
                #param[g] =
                self._update(param, grad, g, i)
            
            model.set_param(param)
        
        return loss.value




class SGD(Optimizer):
    
    def __init__(self, model, x_train=None, y_train=None, hyperparam=dict(), batch_size=128):
        super(SGD, self).__init__(model, x_train, y_train, hyperparam, batch_size)
        
    def _set_param(self):
        self.alpha = self.hyperparam.get("alpha", 0.001)

    def _update(self, param, grad, g, i):
        param[g] -= self.alpha * grad[g]    
             
    def train(self, steps=1000, print_each=100):
        return self._train(steps, num_grad_stores=1, print_each=print_each)

optimizer = SimpleNamespace()
optimizer.SGD = SGD
     
class SGD_Momentum(Optimizer):
    
    def __init__(self, model, x_train=None, y_train=None, hyperparam=dict(), batch_size=128):
        super(SGD_Momentum, self).__init__(model, x_train, y_train, hyperparam, batch_size)
        
    def _set_param(self):
        self.alpha = self.hyperparam.get("alpha", 0.001)
        self.beta = self.hyperparam.get("beta", 0.9) 
    
    def _update(self, param, grad, g, i):
        gradients = self.grad_stores[0]
        gradients[g] = self.beta * gradients[g] + (1-self.beta) * grad[g]
        #gradients[g] /= (1. - self.beta**i) # bias correction
        param[g] -= self.alpha * gradients[g]    
        #return param[g]
    
    def train(self, steps=1000, print_each=100):
        return self._train(steps, num_grad_stores=1, print_each=print_each)
    

optimizer.SGD_Momentum = SGD_Momentum

class RMS_Prop(Optimizer):
    
    def __init__(self, model, x_train=None, y_train=None, hyperparam=dict(), batch_size=128):
        super(RMS_Prop, self).__init__(model, x_train, y_train, hyperparam, batch_size)
    
    def _set_param(self):
        self.alpha = self.hyperparam.get("alpha", 0.001)
        self.beta2 = self.hyperparam.get("beta2", 0.99) 
        self.epsilon = self.hyperparam.get("epsilon", 10e-8) 
    
    def _update(self, param, grad, g, i):
        squared_gradients = self.grad_stores[0]
        squared_gradients[g] = self.beta2 * squared_gradients[g] + (1-self.beta2) * (grad[g])**2
        squared_gradients[g] /= (1. - self.beta2**i) # bias correction
        param[g] -= self.alpha * grad[g]/np.sqrt(self.grad_stores[0][g]+self.epsilon)   
        #return param[g]
    
    def train(self, steps=1000, print_each=100):
        return self._train(steps, num_grad_stores=1, print_each=print_each)

optimizer.RMS_Prop = RMS_Prop

class Adam(Optimizer):
    
    def __init__(self, model, x_train=None, y_train=None, hyperparam=dict(), batch_size=128):
        raise NotImplementedError() # Not tested
        super(RMS_Prop, self).__init__(model, x_train, y_train, hyperparam, batch_size)
    
    def _set_param(self):
        self.alpha = self.hyperparam.get("alpha", 0.001)
        self.beta2 = self.hyperparam.get("beta1", 0.9) 
        self.beta2 = self.hyperparam.get("beta2", 0.99) 
        self.epsilon = self.hyperparam.get("epsilon", 10e-8) 
    
    def _update(self, param, grad, g, i):
        gradients = self.grad_stores[1]
        gradients = self.grad_stores[0]
        gradients[g] = self.beta * gradients[g] + (1-self.beta) * grad[g]
        #gradients[g] /= (1. - self.beta**i) # bias correction
        
        squared_gradients = self.grad_stores[2]
        squared_gradients[g] = self.beta2 * squared_gradients[g] + (1-self.beta2) * (grad[g])**2
        squared_gradients[g] /= (1. - self.beta2**i) # bias correction
           
        param[g] -= self.alpha * gradients[g]/np.sqrt(squared_gradients[g]+self.epsilon) 
        #return param[g]
    
    def train(self, steps=1000, print_each=100):
        return self._train(steps, num_grad_stores=2, print_each=print_each)

# not tested right now
#optimizer.Adam = Adam

#####################################################################################################

class Helper():

    @staticmethod
    def one_hot_encoding(y, nb_classes):
        m = y.shape[0]
        print(m)
        y_ = np.zeros((m, nb_classes), dtype=int)
        y_[np.arange(m), y.astype(int)] = 1
        return y_
