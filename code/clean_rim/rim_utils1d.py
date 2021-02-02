import numpy as np
import tensorflow as tf
from convolutional_recurrent import ConvLSTM3DCell
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose, MaxPool3D, AveragePooling3D, Dense, Conv1D, LSTMCell
import tensorflow_probability as tfp

from tensorflow.python.keras.layers import Dropout
l2reg = tf.keras.regularizers.L2
l1reg = tf.keras.regularizers.L1

import sys
sys.path.append('../../utils/')
import tools
from flowpm.utils import r2c3d, c2r3d


@tf.function
def tfpowerspec(x, boxsize):
    nc = x.shape[-1]
    kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=False)
    kmesh = sum((kk / boxsize * nc)**2 for kk in kvec)**0.5
    kmesh = np.expand_dims(kmesh, 0).astype(float32)
    kbinmap = np.digitize(kmesh, kedges, right=False).astype(int32)
    kbinmap[kbinmap == kbinmap.max()] = kbinmap.max()-1
    kbinmap -= 1
    kbinmap = tf.constant(kbinmap)
    kbincount = tfp.stats.count_integers(kbinmap)
    tfkmesh = tf.constant(kmesh)
    kvals = tfp.stats.count_integers(kbinmap, weights=tfkmesh) / tf.cast(kbincount, tf.float32)
    kvals = tf.repeat(tf.reshape(kvals, (1, nc)), x.shape[0], 0)
    #
    pmesh = tf.square(tf.abs(r2c3d(x, norm=nc**3)))*boxsize**3
    tfpower = tfp.stats.count_integers(tf.repeat(kbinmap, pmesh.shape[0], 0), weights=pmesh, axis=[1, 2, 3])
    tfpower = tf.reshape(tfpower, (nc, x.shape[0]))
    tfpower = tf.transpose(tfpower)/ tf.cast(kbincount, tf.float32)
    return kvals, tfpower 

@tf.function
def tfvarmap(x, kbinmap):
    nc = x.shape[-1]
    kbincount = tfp.stats.count_integers(kbinmap)
    pmesh = tf.square(tf.abs(r2c3d(x, norm=nc**3)))
    tfpower = tfp.stats.count_integers(tf.repeat(kbinmap, pmesh.shape[0], 0), weights=pmesh, axis=[1, 2, 3])
    tfpower = tf.reshape(tfpower, (nc, x.shape[0]))
    tfpower = tf.transpose(tfpower)/ tf.cast(kbincount, tf.float32)
    return  tfpower 


@tf.function
def whitenps(x, ps, kbinmapflat):
    nc = x.shape[-1]
    pmesh = tf.transpose(tf.gather(tf.transpose(ps), kbinmapflat))
    pmesh = tf.reshape(pmesh, x.shape)            
    xwhite = c2r3d(r2c3d(x, norm=nc**3) / tf.cast(pmesh**0.5, tf.complex64), norm=nc**3)
    return xwhite

@tf.function
def colorps(x, ps, kbinmapflat):
    nc = x.shape[-1]
    pmesh = tf.transpose(tf.gather(tf.transpose(ps), kbinmapflat))
    pmesh = tf.reshape(pmesh, x.shape)            
    xnew = c2r3d(r2c3d(x, norm=nc**3) * tf.cast(pmesh**0.5, tf.complex64), norm=nc**3)
    return xnew



class myRIM1D(tf.keras.Model):

    def __init__(self, cell, output_layer, input_layer, cell1d, output_layer1dconv, output_layer1d, input_layer1d, niter, kbinmap, nc, bs):
        super(myRIM1D, self).__init__()
        self.cell = cell
        self.output_layer = output_layer
        self.input_layer = input_layer
        self.cell1d = cell1d
        self.output_layer1d = output_layer1d
        self.output_layer1dc = output_layer1dconv
        self.input_layer1d = input_layer1d
        self.niter = niter
        self.beta_1, self.beta_2 = 0.9, 0.999
        self.lr = 0.1
        self.eps = 1e-7
        self.kbinmap = tf.constant(kbinmap)
        self.kbinmapflat = tf.constant(kbinmap.flatten())
        self.nc, self.bs = nc, bs
        
            
    def call(self, x_init, y, grad_fn, x_true, grad_args=[], initstates = None, return_steps=False):

        bs, nc = self.bs, self.nc
        #outputs_ta = tf.TensorArray(size=self.niter+1, dtype=tf.float32)
        
        if initstates is None: 
            stateshape = [x_init.shape[0]] + [self.cell1d.units]
            initstates1d = [tf.zeros(stateshape), tf.zeros(stateshape)]
            stateshape = x_init.shape + tuple([self.cell.filters])
            initstates3d = [tf.zeros(stateshape), tf.zeros(stateshape)]
            initstates = [initstates1d, initstates3d]
            
        i = tf.constant(0, dtype=tf.int32)
        curr_state = initstates
        curr_pos = x_init        
        m = tf.zeros_like(x_init)
        v = tf.zeros_like(x_init)
        
        def body(i, pos, states, m, v):  
            states1d, states3d = states
            
            gradient = grad_fn(pos, y, *grad_args)           
            t = tf.cast(i+1, tf.float32)
            m = self.beta_1*m + (1-self.beta_1)*gradient
            v = self.beta_2*v + (1-self.beta_2)*gradient**2
            mc = m/(1-self.beta_1**t)
            vc = v/(1-self.beta_2**t)
            delta = -1.*self.lr*mc/(tf.sqrt(vc) + self.eps)
    
            pspos = tfvarmap(pos,  self.kbinmap)
            psdelta = tfvarmap(delta,  self.kbinmap)                                    
            
            lpspos = tf.math.log(pspos *bs**3) / tf.math.log(10.)
            lpsdelta = tf.math.log(psdelta *bs**3) / tf.math.log(10.)
            poswhite = whitenps(pos, pspos, self.kbinmap)
            deltawhite = whitenps(delta, psdelta, self.kbinmap)

            #concat_input = tf.stack([poswhite, deltawhite], axis=-1)
            concat_input = tf.stack([pos, delta], axis=-1)
            cell_input = self.input_layer(concat_input)                        
            delta_poswhite, new_state3d = self.cell(cell_input, states3d)
            delta_poswhite = self.output_layer(delta_poswhite)[...,0]

            concat_input1d = tf.concat([lpspos[:, :], lpsdelta[:, :]], axis=-1)            
            cell_input1d = self.input_layer1d(concat_input1d)            
            lpsdeltanew, new_state1d = self.cell1d(cell_input1d, states1d)
            lpsdeltanew = self.output_layer1d(lpsdeltanew)
            lpsdeltanew = self.output_layer1dc(tf.expand_dims(lpsdeltanew, -1))[..., 0]
            psdeltanew = tf.exp(lpsdeltanew)
            #psdeltanew = tf.exp(lpsdeltanew * tf.math.log(10.)) / bs**3

#             delta_pos = delta_poswhite + tf.reduce_sum(psdeltanew)*0
            delta_pos = colorps(delta_poswhite, psdeltanew, self.kbinmap)

            new_pos = pos + delta_pos
            new_state = [new_state1d, new_state3d]
            
            return i +1 , new_pos, new_state, m, v
        
        loss  = 0.
        while tf.less(i, tf.constant(self.niter)):
            print(i)
            i, curr_pos, curr_state, m, v =  body(i, curr_pos, curr_state, m, v)
            loss = loss + tf.reduce_mean(tf.square(x_true - curr_pos))
        return curr_pos, loss

        #while tf.less(i, tf.constant(self.niter)):
        #    outputs_ta = outputs_ta.write(i, curr_pos)
        #    i, curr_pos, curr_state, m, v =  body(i, curr_pos, curr_state, m, v)
        #outputs_ta = outputs_ta.write(i, curr_pos)
        #return outputs_ta.stack()




def build_rim1d(params):

    nc = params['nc']
    bs = params['bs']
    kbinmap = params['kbinmap']
    input_layer3d = Conv3D(params['input_size'], kernel_size=params['input_kernel_size'], 
                         trainable=True, padding='SAME', 
                         input_shape=(None, nc, nc, nc, 2), activation=params['input_activation'])


    cell3d = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')
   #cell2 = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')

    output_layer3d = Conv3D(1, kernel_size=params['output_kernel_size'], trainable=True, padding='SAME', 
                          input_shape=(None, nc, nc, nc, params['cell_size']*2), activation=params['output_activation'])
   
    cell_size1d = params['cell_size1d']
    input_size1d = cell_size1d
    cell1d = LSTMCell(cell_size1d)
    input_layer1d = Dense(input_size1d, activation=params['input_activation1d'])
    output_layer1d = Dense(nc, activation=params['output_activation1d'])
    output_layer1dconv = Conv1D(1, kernel_size=1, activation='linear')

    rim1d = myRIM1D(cell3d, output_layer3d, input_layer3d, cell1d, output_layer1dconv, output_layer1d, input_layer1d, params['rim_iter'], kbinmap, nc, bs)


    return rim1d






from itertools import cycle

class RIMHybrid(tf.keras.Model):

    def __init__(self, units, input_size, niter, kbinmap, args, droprate=0.05, regrate=1e-5):
        super(RIMHybrid, self).__init__()
        self.nc, self.bs = args.nc, args.bs
        nc, bs =  args.nc, args.bs
        self.niter = niter
        print("number of iterations : ", niter)
        self.beta_1, self.beta_2 = 0.9, 0.999
        self.lr = 0.1
        self.eps = 1e-7
        self.kbinmap = tf.constant(kbinmap)
        self.kbinmapflat = tf.constant(kbinmap.flatten())

        self.layers_in = [Dense(units, activation='tanh', kernel_regularizer=l2reg(regrate)) for i in range(niter)]
        self.drop_in = [Dropout(rate=droprate) for i in range(niter)]
        self.layers_out = [Dense(2*nc, activation='linear', kernel_regularizer=l2reg(regrate)) for i in range(niter)]
        self.drop_out = [Dropout(rate=droprate) for i in range(niter)]
        self.conv_in = [Conv1D(1, 1, activation='linear') for i in range(niter)]
        self.conv_out = [Conv1D(1, 1) for i in range(niter)]
        self.conv_relu = [Conv1D(1, 1, activation='relu') for i in range(niter)]

        self.layers1d = [[self.layers_in[i], self.layers_out[i], \
                         self.drop_in[i], self.drop_out[i], 
                         self.conv_in[i], self.conv_out[i],
                         self.conv_relu[i]] for i in range(niter)]
        
        cell_size = input_size
        self.input_layer = Conv3D(input_size, kernel_size=5, trainable=True, padding='SAME',
                             input_shape=(None, nc, nc, nc, 2), activation='tanh')

        self.cell = ConvLSTM3DCell(cell_size, kernel_size=5, padding='SAME')
        self.output_layer = Conv3D(1, kernel_size=5, trainable=True, padding='SAME', 
                             input_shape=(None, nc, nc, nc, cell_size))


    def call(self, x_init, y, grad_fn, x_true, grad_args=[], initstates = None, return_steps=False, training=False):
        
        nc, bs = self.nc, self.bs
        
        if initstates is None: 
            initstates1d = tf.zeros([x_init.shape[0], nc])
            stateshape = x_init.shape + tuple([self.cell.filters])
            initstates3d = [tf.zeros(stateshape), tf.zeros(stateshape)]
            initstates = [initstates1d, initstates3d]
            
        curr_state = initstates
        curr_pos = x_init        
        m = tf.zeros_like(x_init)
        v = tf.zeros_like(x_init)
        
        #layers = self.conv_in
        def body(i, pos, states, m, v, ll):  
            states1d, states3d = states

            gradient = grad_fn(pos, y, *grad_args)           
            t = tf.cast(i+1, tf.float32)
            m = self.beta_1*m + (1-self.beta_1)*gradient
            v = self.beta_2*v + (1-self.beta_2)*gradient**2
            mc = m/(1-self.beta_1**t)
            vc = v/(1-self.beta_2**t)
            delta = -1.*self.lr*mc/(tf.sqrt(vc) + self.eps)
            
            pspos = tfvarmap(pos,  self.kbinmap)
            psdelta = tfvarmap(delta,  self.kbinmap)                                                
            lpspos = tf.math.log(pspos *bs**3) / tf.math.log(10.)
            lpsdelta = tf.math.log(psdelta *bs**3) / tf.math.log(10.)

            #print("layers : ", i, ll[0])
            #psdeltanew, new_state1d = psdelta, psdelta
            
            layers_in, layers_out, drop_in, drop_out, conv_in, conv_out, conv_relu = ll
            concat_input1d = tf.concat([lpspos, lpsdelta, states1d], axis=-1)   
            x = conv_in(tf.expand_dims(concat_input1d, -1))[..., 0]
            x = drop_in(layers_in(x), training=training)
            x = drop_out(layers_out(x), training=training)
            x = conv_out(tf.expand_dims(x, -1))[..., 0]
            lpsdeltanew, new_state1d = tf.split(x, 2, axis=-1)
            psdeltanew = conv_relu(tf.expand_dims(lpsdeltanew, -1))[..., 0]


            concat_input3d = tf.stack([pos, delta], axis=-1)
            cell_input = self.input_layer(concat_input3d)                        
            delta_pos3d, new_state3d = self.cell(cell_input, states3d)
            delta_pos3d = self.output_layer(delta_pos3d)[...,0]

            delta_pos = colorps(delta_pos3d, psdeltanew, self.kbinmap)        
            new_pos = pos + delta_pos
            new_state = [new_state1d, new_state3d]
            return i +1 , new_pos, new_state, m, v, ll
        
        i = tf.constant(0, dtype=tf.int32)
        loss  = 0.
        convin = self.conv_in
        for ll in self.layers1d:
            i, curr_pos, curr_state, m, v, _ =  body(i, curr_pos, curr_state, m, v, ll)
            loss = loss + tf.reduce_mean(tf.square(x_true - curr_pos))
        return curr_pos, loss





class RIMHybrid2(tf.keras.Model):
 
    def __init__(self, units, input_size, niter, kbinmap, args, droprate=0.05, regrate=1e-5):
        super(RIMHybrid, self).__init__()
        self.nc, self.bs = args.nc, args.bs
        nc, bs =  args.nc, args.bs
        self.niter = niter
        print("number of iterations : ", niter)
        self.beta_1, self.beta_2 = 0.9, 0.999
        self.lr = 0.1
        self.eps = 1e-7
        self.kbinmap = tf.constant(kbinmap)
        self.kbinmapflat = tf.constant(kbinmap.flatten())
##        self.layers_in = cycle([Dense(units, activation='tanh', kernel_regularizer=l2reg(regrate)) for i in range(niter)])
##        self.drop_in = cycle([Dropout(rate=droprate) for i in range(niter)])
##        self.layers_out = cycle([Dense(2*nc, activation='linear', kernel_regularizer=l2reg(regrate)) for i in range(niter)])
##        self.drop_out = cycle([Dropout(rate=droprate) for i in range(niter)])
##        self.conv_in = cycle(cycle([Conv1D(1, 1, activation='linear') for i in range(niter)]))
##        self.conv_out = cycle([Conv1D(1, 1) for i in range(niter)])
##        self.conv_relu = cycle([Conv1D(1, 1, activation='relu') for i in range(niter)])
##     
        cell_size = input_size
        self.input_layer = Conv3D(input_size, kernel_size=5, trainable=True, padding='SAME',
                             input_shape=(None, nc, nc, nc, 2), activation='tanh')

        self.cell = ConvLSTM3DCell(cell_size, kernel_size=5, padding='SAME')
        #self.cell.build(input_shape=[None, nc, nc, nc, input_size])
        self.output_layer = Conv3D(1, kernel_size=5, trainable=True, padding='SAME', 
                             input_shape=(None, nc, nc, nc, cell_size))
            
    def call(self, x_init, y, grad_fn, x_true, grad_args=[], initstates = None, return_steps=False):

        bs, nc = self.bs, self.nc
        #outputs_ta = tf.TensorArray(size=self.niter+1, dtype=tf.float32)
        
        if initstates is None: 
            #stateshape = [x_init.shape[0]] + [self.cell1d.units]
            #initstates1d = [tf.zeros(stateshape), tf.zeros(stateshape)]
            stateshape = x_init.shape + tuple([self.cell.filters])
            initstates1d = [tf.zeros(stateshape), tf.zeros(stateshape)]
            initstates3d = [tf.zeros(stateshape), tf.zeros(stateshape)]
            initstates = [initstates1d, initstates3d]
            
        i = tf.constant(0, dtype=tf.int32)
        curr_state = initstates
        curr_pos = x_init        
        m = tf.zeros_like(x_init)
        v = tf.zeros_like(x_init)
        
        def body(i, pos, states, m, v):  
            states1d, states3d = states
            
            gradient = grad_fn(pos, y, *grad_args)           
            t = tf.cast(i+1, tf.float32)
            m = self.beta_1*m + (1-self.beta_1)*gradient
            v = self.beta_2*v + (1-self.beta_2)*gradient**2
            mc = m/(1-self.beta_1**t)
            vc = v/(1-self.beta_2**t)
            delta = -1.*self.lr*mc/(tf.sqrt(vc) + self.eps)
    
            pspos = tfvarmap(pos,  self.kbinmap)
            psdelta = tfvarmap(delta,  self.kbinmap)                                    
            
            lpspos = tf.math.log(pspos *bs**3) / tf.math.log(10.)
            lpsdelta = tf.math.log(psdelta *bs**3) / tf.math.log(10.)

            #concat_input = tf.stack([poswhite, deltawhite], axis=-1)
            concat_input = tf.stack([pos, delta], axis=-1)
            cell_input = self.input_layer(concat_input)                        
            #delta_poswhite,  new_state3d = cell_input, states3d
            delta_poswhite, new_state3d = self.cell(cell_input, states3d)
            print(delta_poswhite)
            delta_poswhite = self.output_layer(delta_poswhite)[...,0]

            psdeltanew, new_state1d = psdelta, states1d
#            concat_input1d = tf.concat([lpspos[:, :], lpsdelta[:, :]], axis=-1)            
#            cell_input1d = self.input_layer1d(concat_input1d)            
#            lpsdeltanew, new_state1d = self.cell1d(cell_input1d, states1d)
#            lpsdeltanew = self.output_layer1d(lpsdeltanew)
#            lpsdeltanew = self.output_layer1dc(tf.expand_dims(lpsdeltanew, -1))[..., 0]
#            psdeltanew = tf.exp(lpsdeltanew)
#            #psdeltanew = tf.exp(lpsdeltanew * tf.math.log(10.)) / bs**3

            delta_pos = colorps(delta, psdeltanew, self.kbinmap)

            new_pos = pos + delta_pos
            new_state = [new_state1d, new_state3d]
            
            return i +1 , new_pos, new_state, m, v
        
        loss  = 0.
        while tf.less(i, tf.constant(self.niter)):
            print(i)
            i, curr_pos, curr_state, m, v =  body(i, curr_pos, curr_state, m, v)
            loss = loss + tf.reduce_mean(tf.square(x_true - curr_pos))
        return curr_pos, loss

