import numpy as np
from Activation import *
class GRU_HiddenLayer(object):
    def __init__(self,alpha,
                     squared_filter_length_limit,
                     L2_reg,
                     dropout,
                     n_in, n_out, use_bias,
                     dropout_rate,
                     rng,
                     gate_activation = 'sigmoid',
                     cell_activation = 'tanh',):
        self.dropout = dropout
        self.alpha = alpha
        self.squared_filter_length_limit = squared_filter_length_limit
        self.L2_reg = L2_reg
        self.n_in = n_in
        self.n_out = n_out
        self.use_bias = use_bias
        self.rng = rng
        self.dropout_rate = dropout_rate
        if(gate_activation == 'sigmoid'):self.gate_activation = Sigmoid()
        elif(gate_activation == 'tanh'):self.gate_activation = Tanh()
        elif(gate_activation == 'relu'):self.gate_activation = ReLU()
        elif(gate_activation == 'linear'):self.gate_activation = Linear()
        if(cell_activation == 'sigmoid'):self.cell_activation = Sigmoid()
        elif(cell_activation == 'tanh'):self.cell_activation = Tanh()
        elif(cell_activation == 'relu'):self.cell_activation = ReLU()
        elif(cell_activation == 'linear'):self.cell_activation = Linear()
        n_hidden = n_out
        self.params = {}
        self.learning_rate = {}
        self.batch_grad = {}
        self.batch_size = None
        
        self.input_inputGate = Layer(rng = rng, n_in = n_in,n_out = n_hidden,activation = gate_activation)
        self.params['input_inputGate_W'] = self.input_inputGate.W
        self.learning_rate['input_inputGate_W'] = np.ones((n_in, n_hidden), dtype=np.float32)
        self.batch_grad['input_inputGate_W'] = np.zeros((n_in, n_hidden), dtype=np.float32)
        
        self.input_forgetGate = Layer(rng = rng, n_in = n_in,n_out = n_hidden,activation = gate_activation)
        self.params['input_forgetGate_W'] = self.input_forgetGate.W
        self.learning_rate['input_forgetGate_W'] = np.ones((n_in, n_hidden), dtype=np.float32)
        self.batch_grad['input_forgetGate_W'] = np.zeros((n_in, n_hidden), dtype=np.float32)
        
        self.input_outputGate = Layer(rng = rng, n_in = n_in,n_out = n_hidden,activation = gate_activation)
        self.params['input_outputGate_W'] = self.input_outputGate.W
        self.learning_rate['input_outputGate_W'] = np.ones((n_in, n_hidden), dtype=np.float32)
        self.batch_grad['input_outputGate_W'] = np.zeros((n_in, n_hidden), dtype=np.float32)
        
        self.hidden_inputGate = Layer(rng = rng, n_in = n_hidden,n_out = n_hidden,activation = gate_activation)
        self.params['hidden_inputGate_W'] = self.hidden_inputGate.W
        self.learning_rate['hidden_inputGate_W'] = np.ones((n_hidden, n_hidden), dtype=np.float32)
        self.batch_grad['hidden_inputGate_W'] = np.zeros((n_hidden, n_hidden), dtype=np.float32)
        
        self.hidden_forgetGate = Layer(rng = rng, n_in = n_hidden,n_out = n_hidden,activation = gate_activation)
        self.params['hidden_forgetGate_W'] = self.hidden_forgetGate.W
        self.learning_rate['hidden_forgetGate_W'] = np.ones((n_hidden, n_hidden), dtype=np.float32)
        self.batch_grad['hidden_forgetGate_W'] = np.zeros((n_hidden, n_hidden), dtype=np.float32)
        
        self.hidden_outputGate = Layer(rng = rng, n_in = n_hidden,n_out = n_hidden,activation = gate_activation)
        self.params['hidden_outputGate_W'] = self.hidden_outputGate.W
        self.learning_rate['hidden_outputGate_W'] = np.ones((n_hidden, n_hidden), dtype=np.float32)
        self.batch_grad['hidden_outputGate_W'] = np.zeros((n_hidden, n_hidden), dtype=np.float32)
        
        #diagonal matrix
        self.cell_inputGate = Layer(rng = rng, n_in = n_hidden,n_out = n_hidden,activation = gate_activation)
        self.params['cell_inputGate_W'] = self.cell_inputGate.W[0]
        self.learning_rate['cell_inputGate_W'] = np.ones((n_hidden, ), dtype=np.float32)
        self.batch_grad['cell_inputGate_W'] = np.zeros((n_hidden, ), dtype=np.float32)
        
        self.cell_forgetGate = Layer(rng = rng, n_in = n_hidden,n_out = n_hidden,activation = gate_activation)
        self.params['cell_forgetGate_W'] = self.cell_forgetGate.W[0]
        self.learning_rate['cell_forgetGate_W'] = np.ones((n_hidden, ), dtype=np.float32)
        self.batch_grad['cell_forgetGate_W'] = np.zeros((n_hidden, ), dtype=np.float32)
        
        self.cell_outputGate = Layer(rng = rng, n_in = n_hidden,n_out = n_hidden,activation = gate_activation)
        self.params['cell_outputGate_W'] = self.cell_outputGate.W[0]
        self.learning_rate['cell_outputGate_W'] = np.ones((n_hidden, ), dtype=np.float32)
        self.batch_grad['cell_outputGate_W'] = np.zeros((n_hidden, ), dtype=np.float32)
        ####
        
        self.input_cell = Layer(rng = rng, n_in=n_in, n_out=n_hidden,activation = cell_activation)
        self.params['input_cell_W'] = self.input_cell.W
        self.learning_rate['input_cell_W'] = np.ones((n_in, n_hidden), dtype=np.float32)
        self.batch_grad['input_cell_W'] = np.zeros((n_in, n_hidden), dtype=np.float32)
        
        self.hidden_cell = Layer(rng = rng, n_in = n_hidden,n_out = n_hidden,activation = cell_activation)
        self.params['hidden_cell_W'] = self.hidden_cell.W
        self.learning_rate['hidden_cell_W'] = np.ones((n_hidden, n_hidden), dtype=np.float32)
        self.batch_grad['hidden_cell_W'] = np.zeros((n_hidden, n_hidden), dtype=np.float32)
        
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
        self.hidden = []
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
        self.batch_size = x_in.shape[0]
        w_scale = 1.0
        if(self.dropout == True) and (flag_train_phase == True):
            mask = self._mask_maker(x_in)
            x_in = x_in * mask
            self.mask.append(mask)
        elif (self.dropout == True) and (flag_train_phase == False):
            w_scale = 1.0-self.dropout_rate
            
            
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
        
        cur_hidden = output_gate * self.gate_activation.encode(cur_cell)
        
        self.pre_hidden = cur_hidden
        self.pre_cell = cur_cell
        if(flag_train_phase == True):
            self.hidden.append(cur_hidden)
            self.cell.append(cur_cell)
            self.input_gate.append(input_gate)
            self.forget_gate.append(forget_gate)
            self.output_gate.append(output_gate)
            self.input.append(x_in)
            
        return cur_hidden
    
    def get_gradient(self,g_uplayer,cmd):
        hidden = np.asarray(self.hidden)
        cell = np.asarray(self.cell)
        input_gate = np.asarray(self.input_gate)
        forget_gate = np.asarray(self.forget_gate)
        output_gate = np.asarray(self.output_gate)
        z = np.asarray(self.input)
        
        l_sen = len(g_uplayer)/self.batch_size
        pre_cell = [self.init_cell for i in xrange(self.batch_size)]
        pre_hidden = [self.init_hidden for i in xrange(self.batch_size)]
        pre_cell = pre_cell+self.cell[:len(g_uplayer)-self.batch_size]
        pre_hidden = pre_hidden+self.hidden[:len(g_uplayer)-self.batch_size]
        
        pre_cell = np.asarray(pre_cell)
        pre_hidden = np.asarray(pre_hidden)
        
        g_ = {}
        g_hidden = g_uplayer
        g_hidden = np.asarray(g_hidden)
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
        
        for param in g_:
            self.batch_grad[param] += g_[param]
        
        if(self.dropout == True):
            mask = np.asarray(self.mask,dtype = np.float32)
            return g_z * mask
        return g_z
    
    
    def _scale(self,param,squared_filter_length_limit):
        if(squared_filter_length_limit == False):return param
        col_norms = np.sqrt(np.sum(param**2, axis=0))
        desired_norms = np.clip(col_norms, 0, np.sqrt(squared_filter_length_limit))
        scale = desired_norms / (1e-7 + col_norms)
        return param*scale
    
    
    def update_w(self):
        for param in self.params:
            old_param = self.params[param]
            if(old_param.ndim == 2):
                grad = self.batch_grad[param] / self.batch_size + self.L2_reg * old_param
            else:
                grad = self.batch_grad[param] / self.batch_size
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
        self.hidden = []
        self.input_gate = []
        self.forget_gate = []
        self.output_gate = []
        self.input= []
        
        

class GRU(object):

    def __init__(self, 
                 alpha,
                 squared_filter_length_limit,
                 L2_reg,
                 dropout,
                 layer_sizes,
                 dropout_rates,
                 use_bias,
                 rng):
        weight_matrix_sizes = zip(layer_sizes[:-1], layer_sizes[1:])
        self.layers = []
        layer_counter = 0        
        for n_in, n_out in weight_matrix_sizes:
            self.layers.append(GRU_HiddenLayer(alpha=alpha,
                                                 squared_filter_length_limit=squared_filter_length_limit,
                                                 L2_reg = L2_reg,
                                                 dropout=dropout,
                                                 n_in=n_in, n_out=n_out, use_bias=use_bias,
                                                 dropout_rate=dropout_rates[layer_counter],
                                                 rng=rng))
            layer_counter += 1
    def encode(self,x,flag):
        if(np.abs(flag - 1.0)<1e-5):
            flag_train_phase = True
        else:
            flag_train_phase = False
        for layer in self.layers:
            x = layer.encode(x,flag_train_phase)
        return x
    
    def get_gradient(self,g_uplayer,cmd):
        for layer in reversed(self.layers):
            g_uplayer = layer.get_gradient(g_uplayer,cmd)
        return g_uplayer
    
    def update_w(self):
        for layer in self.layers:
            layer.update_w()
    
    def clear_layers(self):
        for layer in self.layers:
            layer.clear_layers()
        
        
        
        