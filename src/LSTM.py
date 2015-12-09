import numpy as np
from Activation import *

from collections import OrderedDict

class LSTM_LayerSetting(object):
    def __init__(self,
                 gate_activation,
                 cell_activation):
        self.gate_activation = gate_activation
        self.cell_activation = cell_activation
        
class LSTM_HiddenLayer(object):
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
        
        self.gate_activation = layer_setting.gate_activation
        self.cell_activation = layer_setting.cell_activation
        n_hidden = n_out
        self.params = OrderedDict({})
        self.learning_rate = OrderedDict({})
        self.batch_grad = OrderedDict({})
        
        self.input_inputGate = Layer(rng = rng, n_in = n_in,n_out = n_hidden,activation = self.gate_activation)
        self.params['input_inputGate_W'] = self.input_inputGate.W
        self.learning_rate['input_inputGate_W'] = np.ones((n_in, n_hidden), dtype=np.float32)
        self.batch_grad['input_inputGate_W'] = np.zeros((n_in, n_hidden), dtype=np.float32)
        
        self.input_forgetGate = Layer(rng = rng, n_in = n_in,n_out = n_hidden,activation = self.gate_activation)
        self.params['input_forgetGate_W'] = self.input_forgetGate.W
        self.learning_rate['input_forgetGate_W'] = np.ones((n_in, n_hidden), dtype=np.float32)
        self.batch_grad['input_forgetGate_W'] = np.zeros((n_in, n_hidden), dtype=np.float32)
        
        self.input_outputGate = Layer(rng = rng, n_in = n_in,n_out = n_hidden,activation = self.gate_activation)
        self.params['input_outputGate_W'] = self.input_outputGate.W
        self.learning_rate['input_outputGate_W'] = np.ones((n_in, n_hidden), dtype=np.float32)
        self.batch_grad['input_outputGate_W'] = np.zeros((n_in, n_hidden), dtype=np.float32)
        
        self.hidden_inputGate = Layer(rng = rng, n_in = n_hidden,n_out = n_hidden,activation = self.gate_activation)
        self.params['hidden_inputGate_W'] = self.hidden_inputGate.W
        self.learning_rate['hidden_inputGate_W'] = np.ones((n_hidden, n_hidden), dtype=np.float32)
        self.batch_grad['hidden_inputGate_W'] = np.zeros((n_hidden, n_hidden), dtype=np.float32)
        
        self.hidden_forgetGate = Layer(rng = rng, n_in = n_hidden,n_out = n_hidden,activation = self.gate_activation)
        self.params['hidden_forgetGate_W'] = self.hidden_forgetGate.W
        self.learning_rate['hidden_forgetGate_W'] = np.ones((n_hidden, n_hidden), dtype=np.float32)
        self.batch_grad['hidden_forgetGate_W'] = np.zeros((n_hidden, n_hidden), dtype=np.float32)
        
        self.hidden_outputGate = Layer(rng = rng, n_in = n_hidden,n_out = n_hidden,activation = self.gate_activation)
        self.params['hidden_outputGate_W'] = self.hidden_outputGate.W
        self.learning_rate['hidden_outputGate_W'] = np.ones((n_hidden, n_hidden), dtype=np.float32)
        self.batch_grad['hidden_outputGate_W'] = np.zeros((n_hidden, n_hidden), dtype=np.float32)
        
        #diagonal matrix
        self.cell_inputGate = Layer(rng = rng, n_in = n_hidden,n_out = n_hidden,activation = self.gate_activation)
        self.params['cell_inputGate_W'] = self.cell_inputGate.W[0]
        self.learning_rate['cell_inputGate_W'] = np.ones((n_hidden, ), dtype=np.float32)
        self.batch_grad['cell_inputGate_W'] = np.zeros((n_hidden, ), dtype=np.float32)
        
        self.cell_forgetGate = Layer(rng = rng, n_in = n_hidden,n_out = n_hidden,activation = self.gate_activation)
        self.params['cell_forgetGate_W'] = self.cell_forgetGate.W[0]
        self.learning_rate['cell_forgetGate_W'] = np.ones((n_hidden, ), dtype=np.float32)
        self.batch_grad['cell_forgetGate_W'] = np.zeros((n_hidden, ), dtype=np.float32)
        
        self.cell_outputGate = Layer(rng = rng, n_in = n_hidden,n_out = n_hidden,activation = self.gate_activation)
        self.params['cell_outputGate_W'] = self.cell_outputGate.W[0]
        self.learning_rate['cell_outputGate_W'] = np.ones((n_hidden, ), dtype=np.float32)
        self.batch_grad['cell_outputGate_W'] = np.zeros((n_hidden, ), dtype=np.float32)
        ####
        
        self.input_cell = Layer(rng = rng, n_in=n_in, n_out=n_hidden,activation = self.cell_activation)
        self.params['input_cell_W'] = self.input_cell.W
        self.learning_rate['input_cell_W'] = np.ones((n_in, n_hidden), dtype=np.float32)
        self.batch_grad['input_cell_W'] = np.zeros((n_in, n_hidden), dtype=np.float32)
        
        self.hidden_cell = Layer(rng = rng, n_in = n_hidden,n_out = n_hidden,activation = self.cell_activation)
        self.params['hidden_cell_W'] = self.hidden_cell.W
        self.learning_rate['hidden_cell_W'] = np.ones((n_hidden, n_hidden), dtype=np.float32)
        self.batch_grad['hidden_cell_W'] = np.zeros((n_hidden, n_hidden), dtype=np.float32)
        
        # init with random !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if(use_bias == True):
            self.params['inputGate_b'] = np.zeros(n_hidden, dtype=np.float32)
            self.learning_rate['inputGate_b'] = np.ones(n_hidden, dtype=np.float32)
            self.batch_grad['inputGate_b'] = np.zeros(n_hidden, dtype=np.float32)
            
            self.params['forgetGate_b'] = np.zeros(n_hidden, dtype=np.float32)
            self.learning_rate['forgetGate_b'] = np.ones(n_hidden, dtype=np.float32)
            self.batch_grad['forgetGate_b'] = np.zeros(n_hidden, dtype=np.float32)
            
            self.params['outputGate_b'] = np.zeros(n_hidden, dtype=np.float32)
            self.learning_rate['outputGate_b'] = np.ones(n_hidden, dtype=np.float32)
            self.batch_grad['outputGate_b'] = np.zeros(n_hidden, dtype=np.float32)
            
            self.params['cell_b'] = np.zeros(n_hidden, dtype=np.float32)
            self.learning_rate['cell_b'] = np.ones(n_hidden, dtype=np.float32)
            self.batch_grad['cell_b'] = np.zeros(n_hidden, dtype=np.float32)
        
        
        
        
        self.init_hidden = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_hidden)),
                    high=np.sqrt(6. / (n_hidden)),
                    size=(n_hidden)), dtype=np.float32)
        self.init_cell = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_hidden)),
                    high=np.sqrt(6. / (n_hidden)),
                    size=(n_hidden)), dtype=np.float32)
        #self.init_pre_hidden = self.init_hidden
        #self.init_pre_cell = self.init_cell
        self.mask = []
        self.cell = []
        self.output = []
        self.input_gate = []
        self.forget_gate = []
        self.output_gate = []
        self.input = []
        self.pre_hidden = self.init_hidden
        self.pre_cell = self.init_cell
    
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
        
        pre_hidden = self.pre_hidden
        pre_cell = self.pre_cell
        
        inside = np.dot(x_in,self.params['input_inputGate_W']*w_scale) + \
                          np.dot(pre_hidden,self.params['hidden_inputGate_W']*w_scale) + \
                          np.dot(pre_cell,np.diag(self.params['cell_inputGate_W']*w_scale))
        if(self.use_bias == True): inside += self.params['inputGate_b']
        input_gate = self.gate_activation.encode(inside)
        
        inside = np.dot(x_in,self.params['input_forgetGate_W']*w_scale) + \
                           np.dot(pre_hidden,self.params['hidden_forgetGate_W']*w_scale) + \
                           np.dot(pre_cell,np.diag(self.params['cell_forgetGate_W']*w_scale))
        if(self.use_bias == True): inside += self.params['forgetGate_b']
        forget_gate = self.gate_activation.encode(inside)
        
        inside = np.dot(x_in,self.params['input_cell_W']*w_scale) + \
                        np.dot(pre_hidden,self.params['hidden_cell_W']*w_scale)
        if(self.use_bias == True): inside += self.params['cell_b']
        cur_cell = forget_gate * pre_cell + \
                    input_gate * self.cell_activation.encode(inside)
        
        inside = np.dot(x_in,self.params['input_outputGate_W']*w_scale) + \
                           np.dot(pre_hidden,self.params['hidden_outputGate_W']*w_scale) + \
                           np.dot(cur_cell,np.diag(self.params['cell_outputGate_W']*w_scale))
        if(self.use_bias == True): inside += self.params['outputGate_b']
        output_gate = self.gate_activation.encode(inside)
        
        cur_hidden = output_gate * self.cell_activation.encode(cur_cell)###
        
        self.pre_hidden = cur_hidden
        self.pre_cell = cur_cell
        if(flag_train_phase == True):
            self.output.append(cur_hidden)
            self.cell.append(cur_cell)
            self.input_gate.append(input_gate)
            self.forget_gate.append(forget_gate)
            self.output_gate.append(output_gate)
            self.input.append(x_in)
            
        return cur_hidden
    
    def get_gradient(self,g_uplayer,cmd):
        
        cell = np.asarray(self.cell[:])
        input_gate = np.asarray(self.input_gate[:])
        forget_gate = np.asarray(self.forget_gate[:])
        output_gate = np.asarray(self.output_gate[:])
        z = np.asarray(self.input[:])
        
        pre_cell = self.init_cell
        pre_hidden = self.init_hidden
        pre_cell = [pre_cell]+self.cell[:-1]
        pre_hidden = [pre_hidden]+self.output[:-1]
        pre_cell = np.asarray(pre_cell)
        pre_hidden = np.asarray(pre_hidden)
        
        #print pre_hidden.shape
        g_ = {}
        g_hidden = g_uplayer
        cell_activation_cell = self.cell_activation.encode(cell)
        g_output_gate = cell_activation_cell * g_hidden
        #g_cell = output_gate * (1 - cell_activation_cell * cell_activation_cell) * g_hidden
        g_cell = output_gate * self.cell_activation.bp(cell_activation_cell) * g_hidden
        
        g_forget_gate = pre_cell * g_cell
        inside = np.dot(z,self.params['input_cell_W']) + np.dot(pre_hidden,self.params['hidden_cell_W'])
        if(self.use_bias == True): inside += self.params['cell_b']
        hiddenlayer_output = self.cell_activation.encode(inside) 
        g_input_gate = hiddenlayer_output * g_cell
        g_hiddenLayer_output = input_gate * g_cell
        
        tmp = g_hiddenLayer_output * self.cell_activation.bp(hiddenlayer_output)
        g_['input_cell_W'] = z.T.dot(tmp)
        g_['hidden_cell_W'] = pre_hidden.T.dot(tmp)
        if(self.use_bias == True): g_['cell_b'] = np.sum(tmp, axis = 0)
        g_z = tmp.dot(self.params['input_cell_W'].T)
        
        tmp = g_output_gate * self.gate_activation.bp(output_gate)
        g_['input_outputGate_W'] = z.T.dot(tmp)
        g_['hidden_outputGate_W'] = pre_hidden.T.dot(tmp)
        g_['cell_outputGate_W'] = np.diag(cell.T.dot(tmp))
        if(self.use_bias == True): g_['outputGate_b'] = np.sum(tmp, axis = 0)
        g_z += tmp.dot(self.params['input_outputGate_W'].T)
        
        tmp = g_forget_gate * self.gate_activation.bp(forget_gate)
        g_['input_forgetGate_W'] = z.T.dot(tmp)
        g_['hidden_forgetGate_W'] = pre_hidden.T.dot(tmp)
        g_['cell_forgetGate_W'] = np.diag(pre_cell.T.dot(tmp))
        if(self.use_bias == True): g_['forgetGate_b'] = np.sum(tmp, axis = 0)
        g_z += tmp.dot(self.params['input_forgetGate_W'].T)
        
        tmp = g_input_gate * self.gate_activation.bp(input_gate)
        g_['input_inputGate_W'] = z.T.dot(tmp)
        g_['hidden_inputGate_W'] = pre_hidden.T.dot(tmp)
        g_['cell_inputGate_W'] = np.diag(pre_cell.T.dot(tmp))
        if(self.use_bias == True): g_['inputGate_b'] = np.sum(tmp, axis = 0)
        g_z += tmp.dot(self.params['input_inputGate_W'].T)
        
        if(cmd == 'minus'):
            for param in g_:
                g_[param] = -g_[param]
        self.g_ = g_
        
        
        for param in g_:
            self.batch_grad[param] += g_[param]
        
        if(self.flag_dropout == True):
            #mask = np.asarray(self.mask,dtype = np.float32)
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
        #clear
        self.clear_layers()
    def clear_layers(self):
        self.mask = []
        self.cell = []
        self.output = []
        self.input_gate = []
        self.forget_gate = []
        self.output_gate = []
        self.input= []
        self.pre_cell = self.init_cell
        self.pre_hidden = self.init_hidden
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        