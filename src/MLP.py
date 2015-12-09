import numpy as np
from Activation import *

from collections import OrderedDict

class MLP_LayerSetting(object):
    def __init__(self,
                 activation):
        self.activation = activation
        
class MLP_HiddenLayer(object):
    def __init__(self,alpha,
                     squared_filter_length_limit,
                     L2_reg,
                     flag_dropout,
                     n_in, n_out, use_bias,
                     dropout_rate,
                     flag_dropout_scaleWeight,
                     layer_setting,
                     rng):
        self.flag_dropout = flag_dropout
        self.alpha = alpha
        self.squared_filter_length_limit = squared_filter_length_limit
        self.L2_reg = L2_reg
        self.n_in = n_in
        self.n_out = n_out
        self.use_bias = use_bias
        self.rng = rng
        self.dropout_rate = dropout_rate
        self.flag_dropout_scaleWeight = flag_dropout_scaleWeight
        
        self.activation = layer_setting.activation
        n_hidden = n_out
        self.params = OrderedDict({})
        self.learning_rate = OrderedDict({})
        self.batch_grad = OrderedDict({})
        
        hiddenLayer = Layer(rng = rng, n_in = n_in,n_out = n_hidden,activation = self.activation)
        self.params['hiddenLayer_W'] = hiddenLayer.W
        self.learning_rate['hiddenLayer_W'] = np.ones_like(hiddenLayer.W, dtype=np.float32)
        self.batch_grad['hiddenLayer_W'] = np.zeros_like(hiddenLayer.W, dtype=np.float32)
        if(use_bias == True):
            self.params['hiddenLayer_b'] = hiddenLayer.b
            self.learning_rate['hiddenLayer_b'] = np.ones_like(hiddenLayer.b, dtype=np.float32)
            self.batch_grad['hiddenLayer_b'] = np.zeros_like(hiddenLayer.b, dtype=np.float32)
        
        
        self.mask = []
        self.input = []
        self.output = []
    
    def _mask_maker(self,x_in):
        mask = self.rng.binomial(n=1, p=1-self.dropout_rate, size=x_in.shape)
        return mask
    
    def encode(self,x_in,flag_train_phase):
        w_scale = 1.0
        if(self.flag_dropout == True) and (flag_train_phase == True):
            mask = self._mask_maker(x_in)
            x_in = x_in * mask
            self.mask.append(mask)
        elif (self.flag_dropout == True) and (flag_train_phase == False):
            w_scale = 1.0-self.dropout_rate
        if(self.flag_dropout_scaleWeight == False): w_scale = 1.0
        
        inside = np.dot(x_in,self.params['hiddenLayer_W']*w_scale)
        if(self.use_bias == True): inside += self.params['hiddenLayer_b']
        hidden = self.activation.encode(inside)
        
        if(flag_train_phase == True):
            self.input.append(x_in)
            self.output.append(hidden)
        
        return hidden
    
    def get_gradient(self,g_uplayer,cmd):
        z = np.asarray(self.input[:])
        #hidden = np.concatenate(self.output[:])
        hidden = np.asarray(self.output[:])
        g_ = {}
        g_hidden = g_uplayer
        
        tmp = g_hidden * self.activation.bp(hidden)
        #print tmp
        g_['hiddenLayer_W'] = z.T.dot(tmp)
        if(self.use_bias == True): g_['hiddenLayer_b'] = np.sum(tmp, axis = 0)
        g_z = tmp.dot(self.params['hiddenLayer_W'].T)
        
        
        if(cmd == 'minus'):
            for param in g_:
                g_[param] = -g_[param]
        
        self.g_ = g_
        
        
        for param in g_:
            self.batch_grad[param] += g_[param]
        
        if(self.flag_dropout == True):
            mask = np.asarray(self.mask[:])
            return g_z * mask
        return g_z
    
    
    def _scale(self,param,squared_filter_length_limit):
        if(squared_filter_length_limit == False):return param
        col_norms = np.sqrt(np.sum(param**2, axis=0))
        desired_norms = np.clip(col_norms, 0, np.sqrt(squared_filter_length_limit))
        scale = desired_norms / (1e-7 + col_norms)
        return param*scale
    
    
    def update_w(self,n):
        for param in self.params:
            old_param = self.params[param]
            if(old_param.ndim == 2):
                grad = self.batch_grad[param] / n + self.L2_reg * old_param
            else:
                grad = self.batch_grad[param] / n
            tmp = self.learning_rate[param] + grad * grad
            lr = self.alpha / (np.sqrt(tmp) + 1.0e-6)
            if(old_param.ndim == 2):
                self.params[param] = self._scale(old_param - lr * grad,self.squared_filter_length_limit)
            else:
                self.params[param] = old_param - lr * grad
            self.learning_rate[param] = tmp
            self.batch_grad[param] = np.zeros_like(old_param,dtype = np.float32)
    def clear_layers(self):
        self.mask = []
        self.output = []
        self.input= []
       
        
        
        
        