import numpy as np
import sys
import time
from collections import OrderedDict
from Activation import ReLU,Sigmoid,Tanh,Add_HiddenLayer
class CWS_Layer(object):
    def __init__(self, rng, n_in, n_out, W=None, b=None, activation = 'tanh'):
        multi = 1.0
        if(activation == 'sigmoid'):multi = 4.0
        if W is None:
            W = np.asarray(rng.uniform(
                    low=-multi * np.sqrt(6. / (n_in + n_out)),
                    high=multi * np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=np.float32)

        if b is None:
            b = np.zeros((n_out,), dtype=np.float32)

        self.W = W
        self.b = b
class CWS(object):
    
    def __init__(self,
                 alpha,
                 squared_filter_length_limit,
                 batch_size,
                 n_epochs,
                 seg_result_file,
                 L2_reg,
                 HINGE_reg,
                 wordVecLen,
                 preWindowSize,
                 surWindowSize,
                 flag_dropout,
                 flag_dropout_scaleWeight,
                 layer_sizes,
                 dropout_rates,
                 layer_types,
                 layer_setting,
                 data,
                 use_bias,
                 use_bigram_feature,
                 random_seed):
        self.rng = np.random.RandomState(random_seed)
        weight_matrix_sizes = zip(layer_sizes[:-1], layer_sizes[1:])
        self.layers = []
        self.layer_sizes = layer_sizes
        self.flag_dropout_scaleWeight = flag_dropout_scaleWeight
        layer_counter = 0        
        for n_in, n_out in weight_matrix_sizes:
            self.layers.append(Add_HiddenLayer(alpha=alpha,
                                                 squared_filter_length_limit=squared_filter_length_limit,
                                                 L2_reg = L2_reg,
                                                 flag_dropout=flag_dropout[layer_counter],
                                                 n_in=n_in, n_out=n_out, use_bias=use_bias,
                                                 dropout_rate=dropout_rates[layer_counter],
                                                 flag_dropout_scaleWeight=flag_dropout_scaleWeight,
                                                 layer_setting = layer_setting[layer_counter],
                                                 rng=self.rng,
                                                 layer_type = layer_types[layer_counter]))
        
        self.alpha = alpha
        self.squared_filter_length_limit=squared_filter_length_limit
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.wordVecLen = wordVecLen
        self.seg_result_file = seg_result_file
        self.L2_reg = L2_reg
        self.HINGE_reg = HINGE_reg
        self.preWindowSize = preWindowSize
        self.surWindowSize = surWindowSize
        self.use_bias = use_bias
        self.use_bigram_feature = use_bigram_feature
        self.data = data
        
        self.params = OrderedDict({})
        self.learning_rate = OrderedDict({})
        self.batch_grad = OrderedDict({})
        self.outputLayer = CWS_Layer(rng = self.rng, n_in=layer_sizes[-1], n_out = 4)
        self.params['outputLayer_W'] = self.outputLayer.W
        self.batch_grad['outputLayer_W'] = np.zeros_like(self.outputLayer.W, dtype=np.float32)
        self.learning_rate['outputLayer_W'] = np.ones_like(self.outputLayer.W, dtype=np.float32)
        if(use_bias == True):
            self.params['outputLayer_b'] = self.outputLayer.b
            self.batch_grad['outputLayer_b'] = np.zeros((4,), dtype=np.float32)
            self.learning_rate['outputLayer_b'] = np.ones((4,), dtype=np.float32)
        self.params['unigram_table'] = data.unigram_table
        self.batch_grad['unigram_table'] = np.zeros(data.unigram_table.shape, dtype=np.float32)
        self.learning_rate['unigram_table'] = np.ones(data.unigram_table.shape, dtype=np.float32)
        if(self.use_bigram_feature == True):
            self.params['bigram_table'] = data.bigram_table
            self.batch_grad['bigram_table'] = np.zeros(data.bigram_table.shape, dtype=np.float32)
            self.learning_rate['bigram_table'] = np.ones(data.bigram_table.shape, dtype=np.float32)
        A = self.rng.normal(loc = 0.0, scale = 0.01, size=(4,4))
        self.params['A'] = np.asarray(A, dtype = np.float32)
        self.batch_grad['A'] = np.zeros(A.shape, dtype=np.float32)
        self.learning_rate['A'] = np.ones(A.shape, dtype=np.float32)
        
        self.INF = 1000000000.0
        self.output = []
        self.dict_unigram_Id = {}
        self.dict_bigram_Id = {}
        self.id_x = []
        
    def encode(self,x,flag):
        for layer in self.layers:
            x = layer.encode(x,flag)#flag == 1 means training,need drop out if flag_dropout==True
        output = np.dot(x, self.params['outputLayer_W'])
        if(self.use_bias == True):
            output += self.params['outputLayer_b']
        return output
    def convert_wordId2wordVec(self,x_in):
        vec = np.concatenate(self.params['unigram_table'][x_in])
        if(self.use_bigram_feature==True):
            bigram = zip(x_in[:-1],x_in[1:])
            for id_bigram in bigram:
                vec = np.concatenate([vec,self.params['bigram_table'][self.data.bigram2id(id_bigram)]])
        return vec
        
    def one_step(self, x_in,cur_label,dp_pre,flag):
        ret1 = [None, None, None, None]
        ret2 = [-1, -1, -1, -1]
        self.id_x.append(x_in)
        x = self.convert_wordId2wordVec(x_in)
        
        val = [[0,1],[2,3],[0,1],[2,3]]
        
        output = self.encode(x,flag)
        self.output.append(output)
        
        for k in xrange(4):
            if dp_pre[k] is None:
                continue
            for j in val[k]:
                loss = 0.0
                if(cur_label!=j):
                    loss = self.HINGE_reg
                if ret1[j] is None:
                    ret1[j] = dp_pre[k] + output[j] + self.params['A'][k][j] + flag*loss
                    ret2[j] = k
                else:
                    if(dp_pre[k] + output[j] + self.params['A'][k][j] + flag * loss > ret1[j]):
                        ret1[j] = dp_pre[k] + output[j] + self.params['A'][k][j] + flag * loss
                        ret2[j] = k
        return ret1 + ret2

    def decode_fun(self,sentence, ans, flag):
        BOSid = self.data.dic_c2idx['<BOS>']
        EOSid = self.data.dic_c2idx['<EOS>']
        ll = len(sentence)
        result = []
        init_v = [0.0, -self.INF, -self.INF, -self.INF,-1,-1,-1,-1]
        for i in xrange(ll):
            x = []
            for j in xrange(self.preWindowSize):
                pos = i - self.preWindowSize + 1 + j
                if(pos < 0):
                    x.append(BOSid)
                    continue
                x.append(sentence[pos])
            #x.append(sentence[i])
            for j in xrange(self.surWindowSize):
                pos = i + (j + 1)
                if(pos >= ll):
                    x.append(EOSid)
                    continue
                x.append(sentence[pos])
            
            init_v = self.one_step(x,ans[i],init_v,flag)
            result.append(init_v)
        return result
    
    def get_best(self,sentence, ans, flag):
        self.clear_layers()
        result = self.decode_fun(sentence, ans, flag)
        sen_len = len(ans)
        ret = [-1 for i in xrange(sen_len)]
        Max = None
        pos = 0
        cur = sen_len - 1
        for i in xrange(4):
            if result[cur][i] is None:
                continue
            if result[cur][i] > Max or Max is None:
                Max = result[cur][i]
                pos = i
        ret[sen_len - 1] = pos
        while cur != 0:
            pos = result[cur][pos + 4]
            cur -= 1
            ret[cur] = pos
        
        cost = 0
        for i in xrange(sen_len):
            output = self.output[i]
            y_pred_in = ret
            y_ans_in = ans
            if(i > 0):
                cost += output[int(y_pred_in[i])] + self.params['A'][y_pred_in[i-1]][y_pred_in[i]] - output[int(y_ans_in[i])] - self.params['A'][y_ans_in[i-1]][y_ans_in[i]]
            else:
                cost += output[int(y_pred_in[i])] + self.params['A'][0][y_pred_in[i]] - output[int(y_ans_in[i])] - self.params['A'][0][y_ans_in[i]]
        #print cost
        return (ret,cost)
    
    def get_gradient(self,sentence,y,cmd):
        l_sen = len(sentence)
        g_outputLayer_W = np.zeros((self.layer_sizes[-1],4))
        g_outputLayer_b = np.zeros((4,))
        for i in xrange(l_sen):
            g_outputLayer_W[:,y[i]] += self.layers[-1].output[i]
            g_outputLayer_b[y[i]] += 1
        g_z = []
        for i in xrange(l_sen):
            g_z.append(((self.params['outputLayer_W'].T)[y[i]]))
        
        for layer in reversed(self.layers):
            g_z = layer.get_gradient(g_z,cmd)
        
        
        g_transMatrix = 1
        if(cmd == 'minus'):
            g_outputLayer_W = -g_outputLayer_W
            g_outputLayer_b = -g_outputLayer_b
            g_z = -g_z
            g_transMatrix = -1
        
        self.batch_grad['outputLayer_W'] += g_outputLayer_W
        if(self.use_bias == True):
            self.batch_grad['outputLayer_b'] += g_outputLayer_b
        
        
        for i in xrange(l_sen):
            cur_y = y[i]
            pre_y = 0
            if(i>0):pre_y = y[i-1]
            nn = 0
            wordVecLen = self.wordVecLen
            for mx in self.id_x[i]:
                self.dict_unigram_Id[mx] = 1
                self.batch_grad['unigram_table'][mx] += g_z[i][nn*wordVecLen:(nn+1)*wordVecLen]
                nn += 1
            if(self.use_bigram_feature == False):continue
            bigram = zip(self.id_x[i][:-1],self.id_x[i][1:])
            for (m1,m2) in bigram:
                id_bigram = self.data.bigram2id((m1,m2))
                self.dict_bigram_Id[id_bigram] = 1
                self.batch_grad['bigram_table'][id_bigram] += g_z[i][nn*wordVecLen:(nn+1)*wordVecLen]
                nn += 1
            self.batch_grad['A'][pre_y][cur_y] += g_transMatrix
        
    def _scale(self,param,squared_filter_length_limit):
        if(squared_filter_length_limit == False):return param
        col_norms = np.sqrt(np.sum(param**2, axis=0))
        desired_norms = np.clip(col_norms, 0, np.sqrt(squared_filter_length_limit))
        scale = desired_norms / (1e-7 + col_norms)
        return param*scale
    def modify_batch_grad(self,sentence,pred,label):
        y = pred
        self.get_gradient(sentence,y,'plus')
        y = label
        self.get_gradient(sentence,y,'minus')
    def update_w(self,n):
        
        for layer in self.layers:
            layer.update_w(n)
        #need update weith here
        for param in self.params:
            if(param == 'unigram_table') or (param == 'bigram_table'):
                if(param == 'unigram_table'):
                    dict_gram_Id = self.dict_unigram_Id
                elif(param == 'bigram_table'):
                    dict_gram_Id = self.dict_bigram_Id
                for idx in dict_gram_Id:
                    old_param = self.params[param][idx]
                    grad = self.batch_grad[param][idx] / n + self.L2_reg * old_param
                    tmp = self.learning_rate[param][idx] + grad * grad
                    lr = self.alpha / (np.sqrt(tmp) + 1.0e-6)
                    self.params[param][idx] = self._scale(old_param - lr * grad,self.squared_filter_length_limit)
                    self.learning_rate[param][idx] = tmp
                    self.batch_grad[param][idx] = np.zeros_like(self.params[param][idx], dtype=np.float32)
                continue
            
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
        self.clear_layers()
    
    def clear_layers(self):
        for layer in self.layers:
            layer.clear_layers()
        self.output = []
        self.id_x = []
            
        
    
    def fit(self):
        M = len(self.data.data_train)
        batch_num = M / self.batch_size
        print 'Start training...'
        sys.stdout.flush()
        best_validation_F = -np.inf
        best_iter = 0
        for epoch in xrange(self.n_epochs):
            (self.data.data_train,self.data.label_train) = self.data.shuffle(self.data.data_train,self.data.label_train)
            ae_costs = 0.0
            start_time = time.time()
            for batch in xrange(batch_num + 1):
                start = batch * self.batch_size
                end = min((batch + 1)*self.batch_size, M)
                if(end<=start):continue
                self.dict_bigram_Id = {}
                self.dict_unigram_Id = {}
                for index in xrange(start, end):
                    (y_pred,cost) = self.get_best(self.data.data_train[index], self.data.label_train[index], 1.0)
                    ae_costs += cost 
                    self.modify_batch_grad(self.data.data_train[index], y_pred, self.data.label_train[index])
                            
                if (batch+1) % 50 == 0:
                    print '50 batches proccessed'
                    print '%d done!' % end
                    print 'Training at batch %d, cost = %f' % (epoch*(batch_num+1)+batch + 1, ae_costs/((batch+1)*self.batch_size))
                    sys.stdout.flush()
                                    
                self.update_w(end - start)
               
                
                

            cur_cost = ae_costs/M
            print 'Training at epoch %d, cost = %f' % (epoch + 1, cur_cost)
            if (epoch+1) % 10 == 0 :
                self.test(self.data.data_train, self.data.label_train, "Train")
            (seg, eval_res) = self.test(self.data.data_dev, self.data.label_dev, "Dev")
            if(best_validation_F < eval_res[2]):
                best_iter = epoch + 1
                best_validation_F = eval_res[2]
            print 'Current best_dev_F is %.2f, at %d epoch'%(best_validation_F,best_iter)
            print 'Testing...'
            sys.stdout.flush()
            (seg, eval_res) = self.test(self.data.data_test, self.data.label_test, 'Test')
            print 'Saving test result for %d_th epoch' % (epoch+1)
            sys.stdout.flush()
            suffix = '_%d' % (epoch+1)
            local_seg_result_file = self.seg_result_file + suffix 
            fw = open(local_seg_result_file, 'w')
            for line in seg:
                for word in line:
                    fw.write(word.encode('utf-8') + ' ')
                fw.write('\n')
            fw.close()
            
            end_time = time.time()
            minu = int((end_time - start_time)/60)
            sec = (end_time - start_time) - 60 * minu
            print 'Time: %d min %.2f sec' % (minu, sec)
            sys.stdout.flush()

    def test(self, data, label, flag):
        res = []
        ans = []
        M = len(data)
        for index in xrange(M):
            (y_pred,cost) = self.get_best(data[index], label[index], 0.0)
            res.append(y_pred)
        pred_seg = self.get_seg(data, res, self.data.dic_idx2c)
        ans_seg = self.get_seg(data, label, self.data.dic_idx2c)
        eval_res = self.evaluate(ans_seg, pred_seg)
        print '%s: P = %f    R=%f    F=%f' % (flag, eval_res[0], eval_res[1], eval_res[2])
        sys.stdout.flush()
        return (pred_seg, eval_res)

    def get_seg(self, data, label, ids):
        ret = []
        for i in xrange(len(data)):
            line = []
            word = u''
            sen = data[i]
            res = label[i]
            for j in xrange(len(sen)):
                if abs(res[j]) <= 1.0e-6:
                    word = ids[sen[j]]
                    line.append(word)
                    word = u''
                elif abs(res[j] - 2.0) <= 1.0e-6:
                    word += ids[sen[j]]
                    line.append(word)
                    word = u''
                else:
                    word += ids[sen[j]]
            if len(word) != 0:
                line.append(word)
            ret.append(line)
        return ret

    def evaluate(self, ans, res):
        right = 0
        wrong = 0
        tot_right = 0
        for i in range(0,len(res)):
            line1 = res[i]
            line2 = ans[i]
            res1 = []
            res2 = []
            j = 0
            k = 0
            for word in line1:
                l = len(word)
                res1.append(l)
                for j in range(1,l):
                    res1.append(-1)
            for word in line2:
                l = len(word)
                res2.append(l)
                for j in range(1,l):
                    res2.append(-1)
            for j in range(0,len(res1)):
                if res1[j] == -1:
                    continue
                if res1[j] == res2[j]:
                    right += 1
                else:
                    wrong += 1
            tot_right += len(line2)
            #print 'right=%d' % right
            #print 'wrong=%d' % wrong
        p = (1.0*right/(right+wrong))
        r = (1.0*right/tot_right)
        f = (2*p*r/(p+r))
        return (p, r, f)
        