import numpy as np
from collections import OrderedDict

class Data(object):
    def __init__(self,path_lookup_table,wordVecLen,path_train_data,path_test_data,
                 flag_random_lookup_table,dic_label,use_bigram_feature,random_seed,flag_toy_data,
                 path_dev_data = None):
        self.rng = np.random.RandomState(random_seed)
        self.dic_c2idx = {}
        self.dic_idx2c = {}
        self.wordVecLen = wordVecLen
        f = open(path_lookup_table,'r')
        li = f.readline()
        li = li.split()
        n_dict = int(li[0])
        self.n_unigram = n_dict
        v_lt = self.rng.normal(loc = 0.0, scale = 0.01, size=(n_dict, wordVecLen))
        #lookup_table = np.zeros([n_dict, 25],dtype = np.float32)
        self.unigram_table = np.asarray(v_lt, dtype = np.float32)
        n_dim = int(li[1])
        for i in xrange(n_dict):
            li = f.readline()
            li = unicode(li,'utf-8')
            li = li.split()
            if (len(li)!=wordVecLen+1):
                continue
            self.dic_c2idx[li[0]] = i
            self.dic_idx2c[i] = li[0]
            if(flag_random_lookup_table == True):continue
            for j in xrange(n_dim):
                self.unigram_table[i][j] = float(li[j+1])
        f.close()
        if(use_bigram_feature == True):
            v_lt = self.rng.normal(loc = 0.0, scale = 0.01, size=(n_dict * n_dict, wordVecLen))
            self.bigram_table = np.asarray(v_lt, dtype = np.float32)
            for i in xrange(n_dict):
                for j in xrange(n_dict):
                    self.bigram_table[i*n_dict + j] = 0.5 * (self.unigram_table[i] + self.unigram_table[j])
        
        
        
        
        data_train = []
        data_sentence = []
        label_train = []
        label_sentence = []
        #f = open('pkutrain_noNUMENG.utf8', 'r')
        f = open(path_train_data, 'r')
        li = f.readlines()
        f.close()
    
        for line in li:
            #print line
            line = unicode(line,'utf-8')
            line_t = line.split()
            if(len(line_t)==0):
                if(len(data_sentence) == 0):
                    continue
                data_train.append(data_sentence)
                label_train.append(label_sentence)
                data_sentence = []
                label_sentence = []
                continue
            ch = line_t[0]
            if(self.dic_c2idx.get(ch) == None):
                ch = self.dic_c2idx['<OOV>']
            else:
                ch = self.dic_c2idx[ch]
            data_sentence += [ch]
            label_sentence += [dic_label[line_t[1]]]
        if(path_dev_data == None):
            l_len = len(data_train)
            thr = int(l_len * 0.9)
            data_dev = data_train[thr:]
            label_dev = label_train[thr:]
            data_train = data_train[:thr]
            label_train = label_train[:thr]
        else:
            data_dev = []
            data_sentence = []
            label_dev = []
            label_sentence = []
            #f = open('pkutrain_noNUMENG.utf8', 'r')
            f = open(path_dev_data, 'r')
            li = f.readlines()
            f.close()
        
            for line in li:
                line = unicode(line,'utf-8')
                line_t = line.split()
                if(len(line_t)==0):
                    if(len(data_sentence) == 0):
                        continue
                    data_dev.append(data_sentence)
                    label_dev.append(label_sentence)
                    data_sentence = []
                    label_sentence = []
                    continue
                ch = line_t[0]
                if(self.dic_c2idx.get(ch) == None):
                    ch = self.dic_c2idx['<OOV>']
                else:
                    ch = self.dic_c2idx[ch]
                data_sentence += [ch]
                label_sentence += [dic_label[line_t[1]]]
        
        data_test = []
        label_test = []
        data_sentence = []
        label_sentence = []
        #f = open('pkutest_noNUMENG.utf8', 'r')
        f = open(path_test_data, 'r')
        li = f.readlines()
        f.close()
        for line in li:
            line = unicode(line,'utf-8')
            line_t = line.split()
            if(len(line_t)==0):
                if(len(data_sentence) == 0):
                    continue
                data_test.append(data_sentence)
                label_test.append(label_sentence)
                data_sentence = []
                label_sentence = []
                continue
            ch = line_t[0]
            if(self.dic_c2idx.get(ch) == None):
                ch = self.dic_c2idx['<OOV>']
            else:
                ch = self.dic_c2idx[ch]
            data_sentence += [ch]
            label_sentence += [dic_label[line_t[1]]]
            
        if(flag_toy_data == False):pass
        else:
            l_len = len(data_train)
            thr = int(l_len * flag_toy_data)
            data_train = data_train[:thr]
            label_train = label_train[:thr]
            
            data_dev = data_train[:]
            label_dev = label_train[:]
            l_len = len(data_test)
            thr = int(l_len * flag_toy_data)
            data_test = data_test[:thr]
            label_test = label_test[:thr]
        
        self.data_train = data_train
        self.label_train = label_train
        self.data_dev = data_dev
        self.label_dev = label_dev
        self.data_test = data_test
        self.label_test = label_test
                
    def shuffle(self,data_in,label_in):
        l_len = len(data_in)
        permu = range(l_len)
        self.rng.shuffle(permu)
        data_out = []
        label_out = []
        for i in xrange(l_len):
            data_out.append(data_in[permu[i]])
            label_out.append(label_in[permu[i]])
        return (data_out,label_out)
    
    
    
    def bigram2id(self,id_bigram):
        (m1,m2) = id_bigram
        return m1*self.n_unigram+m2
    
    def display(self,sentences):
        n = len(sentences)
        for sentence in sentences:
            s = ''
            for ch in sentence:
                s += (self.dic_idx2c[ch].encode('utf-8') + ' ')
            print s,'\n'

        
        
        
        
        
        