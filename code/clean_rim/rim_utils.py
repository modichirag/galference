import numpy as np
import tensorflow as tf
from convolutional_recurrent import ConvLSTM3DCell
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose, MaxPool3D, AveragePooling3D

import sys
sys.path.append('../../utils/')
import tools



class RIM3D_parallel(tf.keras.Model):

    def __init__(self, cell1, cell2, input_layer, input_layer_sub, output_layer_up, output_layer, strides, niter):
        super(RIM3D_parallel, self).__init__()
        self.cell1 = cell1
        self.cell2 = cell2
        self.output_layer = output_layer
        self.output_layer_up = output_layer_up
        self.input_layer = input_layer
        self.input_layer_sub = input_layer_sub
        self.strides = strides
        self.niter = niter
        self.beta_1, self.beta_2 = 0.9, 0.999
        self.lr, self.eps = 0.1, 1e-7
            
            
    def call(self, x_init, y, grad_fn, grad_args=[], initstates = None, return_steps=False):

        outputs_ta = tf.TensorArray(size=self.niter+1, dtype=tf.float32)

        
        if initstates is None: 
            #stateshape = tuple(i//self.strides for i in x_init.shape) + tuple([self.cell1.filters])
            #stateshape = x_init.shape + tuple([self.cell.filters])
            #initstates = [tf.zeros(stateshape), tf.zeros(stateshape)]
            nc2 = int(x_init.shape[1]/self.strides)
            stateshape = (x_init.shape[0], nc2, nc2, nc2, self.cell1.filters)
            initstates1 = [tf.zeros(stateshape), tf.zeros(stateshape)]
            stateshape = x_init.shape + tuple([self.cell2.filters])
            initstates2 = [tf.zeros(stateshape), tf.zeros(stateshape)]
            initstates = [initstates1, initstates2]

        i = tf.constant(0, dtype=tf.int32)
        curr_state = initstates
        curr_pos = x_init        
        m = tf.zeros_like(x_init)
        v = tf.zeros_like(x_init)
        
        def body(i, pos, states, m, v):  
            gradient = grad_fn(pos, y, *grad_args)           
            t = tf.cast(i+1, tf.float32)
            m = self.beta_1*m + (1-self.beta_1)*gradient
            v = self.beta_2*v + (1-self.beta_2)*gradient**2
            mc = m/(1-self.beta_1**t)
            vc = v/(1-self.beta_2**t)
            delta = -1.*self.lr*mc/(tf.sqrt(vc) + self.eps)
            #
            states1, states2 = states
            concat_input = tf.stack([pos, delta], axis=-1)
            #
            cell_input_sub = self.input_layer_sub(concat_input)
            delta_pos1, new_states1 = self.cell1(cell_input_sub, states1)
            delta_pos1 = self.output_layer_up(delta_pos1) 
            #
            cell_input = self.input_layer(concat_input)
            delta_pos2, new_states2 = self.cell2(cell_input, states2)
            #delta_pos2 = self.output_layer(delta_pos2)
            #
            #delta_pos = delta_pos1 + delta_pos2
            delta_pos = tf.concat([delta_pos1, delta_pos2], axis=-1)
            delta_pos = self.output_layer(delta_pos)
            new_pos = pos + delta_pos[..., 0]
            new_states = [new_states1, new_states2]
            return i + 1 , new_pos, new_states, m, v


        while tf.less(i, tf.constant(self.niter)):
            outputs_ta = outputs_ta.write(i, curr_pos)
            i, curr_pos, curr_state, m, v =  body(i, curr_pos, curr_state, m, v)
        outputs_ta = outputs_ta.write(i, curr_pos)
        return outputs_ta.stack()




def build_rim_parallel(params):

    nc = params['nc']
    input_layer = Conv3D(params['input_size'], kernel_size=params['input_kernel_size'], 
                         trainable=True, padding='SAME', 
                         input_shape=(None, nc, nc, nc, 2), activation=params['input_activation'])

    input_layer_sub = Conv3D(params['input_size'], kernel_size=params['input_kernel_size'], 
                             trainable=True, padding='SAME', strides= [params['strides']]*3,
                             input_shape=(None, nc, nc, nc, 2), activation=params['input_activation'])

    cell1 = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')

    output_layer_up = Conv3DTranspose(params['cell_size'], kernel_size=params['middle_kernel_size'], 
                         trainable=True, padding='SAME', strides=[params['strides']]*3, 
                         activation=params['output_activation'])

    cell2 = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')

    output_layer = Conv3D(1, kernel_size=params['output_kernel_size'], trainable=True, padding='SAME', 
                          input_shape=(None, nc, nc, nc, params['cell_size']*2), activation=params['output_activation'])
   
    rim = RIM3D_parallel(cell1, cell2, input_layer, input_layer_sub, output_layer_up, output_layer, strides=params['strides'],
                       niter=params['rim_iter'])

    return rim


class RIM3D_parallel_single(tf.keras.Model):

    def __init__(self, cell1, cell2, input_layer, input_layer_sub, output_layer_up, output_layer, strides, niter):
        super(RIM3D_parallel_single, self).__init__()
        self.cell1 = cell1
        self.cell2 = cell2
        self.output_layer = output_layer
        self.output_layer_up = output_layer_up
        self.input_layer = input_layer
        self.input_layer_sub = input_layer_sub
        self.strides = strides
        self.niter = niter
        self.beta_1, self.beta_2 = 0.9, 0.999
        self.lr, self.eps = 0.1, 1e-7
            
            
    def call(self, x_init, y, grad_fn, x_true, grad_args=[], initstates = None, return_steps=False):


        if initstates is None: 
            #stateshape = tuple(i//self.strides for i in x_init.shape) + tuple([self.cell1.filters])
            #stateshape = x_init.shape + tuple([self.cell.filters])
            #initstates = [tf.zeros(stateshape), tf.zeros(stateshape)]
            nc2 = int(x_init.shape[1]/self.strides)
            stateshape = (x_init.shape[0], nc2, nc2, nc2, self.cell1.filters)
            initstates1 = [tf.zeros(stateshape), tf.zeros(stateshape)]
            stateshape = x_init.shape + tuple([self.cell2.filters])
            initstates2 = [tf.zeros(stateshape), tf.zeros(stateshape)]
            initstates = [initstates1, initstates2]

        i = tf.constant(0, dtype=tf.int32)
        curr_state = initstates
        curr_pos = x_init        
        m = tf.zeros_like(x_init)
        v = tf.zeros_like(x_init)
        
        def body(i, pos, states, m, v):  
            gradient = grad_fn(pos, y, *grad_args)           
            t = tf.cast(i+1, tf.float32)
            m = self.beta_1*m + (1-self.beta_1)*gradient
            v = self.beta_2*v + (1-self.beta_2)*gradient**2
            mc = m/(1-self.beta_1**t)
            vc = v/(1-self.beta_2**t)
            delta = -1.*self.lr*mc/(tf.sqrt(vc) + self.eps)
            #
            states1, states2 = states
            concat_input = tf.stack([pos, delta], axis=-1)
            #
            cell_input_sub = self.input_layer_sub(concat_input)
            delta_pos1, new_states1 = self.cell1(cell_input_sub, states1)
            delta_pos1 = self.output_layer_up(delta_pos1) 
            #
            cell_input = self.input_layer(concat_input)
            delta_pos2, new_states2 = self.cell2(cell_input, states2)
            #delta_pos2 = self.output_layer(delta_pos2)
            #
            #delta_pos = delta_pos1 + delta_pos2
            delta_pos = tf.concat([delta_pos1, delta_pos2], axis=-1)
            delta_pos = self.output_layer(delta_pos)
            new_pos = pos + delta_pos[..., 0]
            new_states = [new_states1, new_states2]
            return i + 1 , new_pos, new_states, m, v

        loss  = 0.
        while tf.less(i, tf.constant(self.niter)):
            i, curr_pos, curr_state, m, v =  body(i, curr_pos, curr_state, m, v)
            loss = loss + tf.reduce_mean(tf.square(x_true - curr_pos))
        return curr_pos, loss




def build_rim_parallel_single(params):

    nc = params['nc']
    input_layer = Conv3D(params['input_size'], kernel_size=params['input_kernel_size'], 
                         trainable=True, padding='SAME', 
                         input_shape=(None, nc, nc, nc, 2), activation=params['input_activation'])

    input_layer_sub = Conv3D(params['input_size'], kernel_size=params['input_kernel_size'], 
                             trainable=True, padding='SAME', strides= [params['strides']]*3,
                             input_shape=(None, nc, nc, nc, 2), activation=params['input_activation'])

    cell1 = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')

    output_layer_up = Conv3DTranspose(params['cell_size'], kernel_size=params['middle_kernel_size'], 
                         trainable=True, padding='SAME', strides=[params['strides']]*3, 
                         activation=params['output_activation'])

    cell2 = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')

    output_layer = Conv3D(1, kernel_size=params['output_kernel_size'], trainable=True, padding='SAME', 
                          input_shape=(None, nc, nc, nc, params['cell_size']*2), activation=params['output_activation'])
   
    rim = RIM3D_parallel_single(cell1, cell2, input_layer, input_layer_sub, output_layer_up, output_layer, strides=params['strides'],
                       niter=params['rim_iter'])

    return rim







class RIM3D_parallel3_single(tf.keras.Model):

    def __init__(self, cells, input_layers, output_layers, strides, niter):
        super(RIM3D_parallel3_single, self).__init__()
        self.cell1, self.cell2, self.cell3 = cells
        self.input_layer1, self.input_layer2, self.input_layer3 = input_layers
        self.output_layer1, self.output_layer2, self.output_layer3 = output_layers
        self.strides = strides
        self.niter = niter
        self.beta_1, self.beta_2 = 0.9, 0.999
        self.lr, self.eps = 0.1, 1e-7
            
            
    def call(self, x_init, y, grad_fn, x_true, grad_args=[], initstates = None, return_steps=False):


        if initstates is None: 
            #stateshape = tuple(i//self.strides for i in x_init.shape) + tuple([self.cell1.filters])
            #stateshape = x_init.shape + tuple([self.cell.filters])
            #initstates = [tf.zeros(stateshape), tf.zeros(stateshape)]
            nc2 = int(x_init.shape[1]/self.strides)
            nc3 = int(x_init.shape[1]/self.strides**2)
            stateshape = (x_init.shape[0], nc3, nc3, nc3, self.cell3.filters)
            initstates3 = [tf.zeros(stateshape), tf.zeros(stateshape)]
            stateshape = (x_init.shape[0], nc2, nc2, nc2, self.cell2.filters)
            initstates2 = [tf.zeros(stateshape), tf.zeros(stateshape)]
            stateshape = x_init.shape + tuple([self.cell1.filters])
            initstates1 = [tf.zeros(stateshape), tf.zeros(stateshape)]
            initstates = [initstates1, initstates2, initstates3]


        i = tf.constant(0, dtype=tf.int32)
        curr_state = initstates
        curr_pos = x_init        
        m = tf.zeros_like(x_init)
        v = tf.zeros_like(x_init)
        
        def body(i, pos, states, m, v):  
            gradient = grad_fn(pos, y, *grad_args)           
            t = tf.cast(i+1, tf.float32)
            m = self.beta_1*m + (1-self.beta_1)*gradient
            v = self.beta_2*v + (1-self.beta_2)*gradient**2
            mc = m/(1-self.beta_1**t)
            vc = v/(1-self.beta_2**t)
            delta = -1.*self.lr*mc/(tf.sqrt(vc) + self.eps)
            #
            states1, states2, states3 = states
            concat_input = tf.stack([pos, delta], axis=-1)
            cell_input1 = self.input_layer1(concat_input)
            cell_input2 = self.input_layer2(concat_input)
            cell_input3 = self.input_layer3(concat_input)
            #
            delta_pos1, new_states1 = self.cell1(cell_input1, states1)
            delta_pos2, new_states2 = self.cell2(cell_input2, states2)
            delta_pos3, new_states3 = self.cell3(cell_input3, states3)
            #
            delta_pos3 = self.output_layer3(delta_pos3)
            delta_pos2 = tf.concat([delta_pos2, delta_pos3], axis=-1)
            delta_pos2 = self.output_layer2(delta_pos2)
            delta_pos1 = tf.concat([delta_pos1, delta_pos2], axis=-1)
            delta_pos = self.output_layer1(delta_pos1)
            #
            new_pos = pos + delta_pos[..., 0]
            new_states = [new_states1, new_states2, new_states3]
            return i + 1 , new_pos, new_states, m, v

        loss  = 0.
        while tf.less(i, tf.constant(self.niter)):
            i, curr_pos, curr_state, m, v =  body(i, curr_pos, curr_state, m, v)
            loss = loss + tf.reduce_mean(tf.square(x_true - curr_pos))
        return curr_pos, loss




def build_rim_parallel3_single(params):

    nc = params['nc']

    cell1 = ConvLSTM3DCell(params['cell_size1'], kernel_size=params['cell_kernel_size1'], padding='SAME')
    cell2 = ConvLSTM3DCell(params['cell_size2'], kernel_size=params['cell_kernel_size2'], padding='SAME')
    cell3 = ConvLSTM3DCell(params['cell_size3'], kernel_size=params['cell_kernel_size3'], padding='SAME')
    cells = [cell1, cell2, cell3]

    input_layer1 = Conv3D(params['input_size1'], kernel_size=params['input_kernel_size1'], 
                         trainable=True, padding='SAME', activation=params['input_activation'])
    input_layer2 = Conv3D(params['input_size2'], kernel_size=params['input_kernel_size2'], strides= [params['strides']]*3,
                         trainable=True, padding='SAME', activation=params['input_activation'])
    input_layer3 = Conv3D(params['input_size3'], kernel_size=params['input_kernel_size3'], strides= [2*params['strides']]*3,
                         trainable=True, padding='SAME', activation=params['input_activation'])
    input_layers = [input_layer1, input_layer2, input_layer3]

    output_layer1 = Conv3D(params['output_size1'], kernel_size=params['output_kernel_size1'], trainable=True, padding='SAME', 
                          activation=params['output_activation'])   
    output_layer2 = Conv3DTranspose(params['output_size2'], kernel_size=params['output_kernel_size2'], trainable=True, padding='SAME', 
                          strides= [params['strides']]*3, activation=params['output_activation'])
    output_layer3 = Conv3DTranspose(params['output_size3'], kernel_size=params['output_kernel_size3'], trainable=True, padding='SAME', 
                          strides= [params['strides']]*3, activation=params['output_activation'])
    output_layers = [output_layer1, output_layer2, output_layer3]
   
    rim = RIM3D_parallel3_single(cells, input_layers, output_layers, strides=params['strides'], niter=params['rim_iter'])

    return rim






#$#
#$#class RIM3D_parallel3_single(tf.keras.Model):
#$#
#$#    def __init__(self, cells, input_layers, output_layers, strides, niter):
#$#        super(RIM3D_parallel3_single, self).__init__()
#$#        self.cell1, self.cell2, self.cell3 = cells
#$#        self.input_layer1, self.input_layer2, self.input_layer3 = input_layers
#$#        self.output_layer1, self.output_layer2, self.output_layer3 = output_layers
#$#        self.strides = strides
#$#        self.niter = niter
#$#        self.beta_1, self.beta_2 = 0.9, 0.999
#$#        self.lr, self.eps = 0.1, 1e-7
#$#            
#$#            
#$#    def call(self, x_init, y, grad_fn, x_true, grad_args=[], initstates = None, return_steps=False):
#$#
#$#
#$#        if initstates is None: 
#$#            #stateshape = tuple(i//self.strides for i in x_init.shape) + tuple([self.cell1.filters])
#$#            #stateshape = x_init.shape + tuple([self.cell.filters])
#$#            #initstates = [tf.zeros(stateshape), tf.zeros(stateshape)]
#$#            nc2 = int(x_init.shape[1]/self.strides)
#$#            nc3 = int(x_init.shape[1]/self.strides**2)
#$#            stateshape = (x_init.shape[0], nc3, nc3, nc3, self.cell3.filters)
#$#            initstates3 = [tf.zeros(stateshape), tf.zeros(stateshape)]
#$#            stateshape = (x_init.shape[0], nc2, nc2, nc2, self.cell2.filters)
#$#            initstates2 = [tf.zeros(stateshape), tf.zeros(stateshape)]
#$#            stateshape = x_init.shape + tuple([self.cell1.filters])
#$#            initstates1 = [tf.zeros(stateshape), tf.zeros(stateshape)]
#$#            initstates = [initstates1, initstates2, initstates3]
#$#
#$#
#$#        i = tf.constant(0, dtype=tf.int32)
#$#        curr_state = initstates
#$#        curr_pos = x_init        
#$#        m = tf.zeros_like(x_init)
#$#        v = tf.zeros_like(x_init)
#$#        
#$#        def body(i, pos, states, m, v):  
#$#            gradient = grad_fn(pos, y, *grad_args)           
#$#            t = tf.cast(i+1, tf.float32)
#$#            m = self.beta_1*m + (1-self.beta_1)*gradient
#$#            v = self.beta_2*v + (1-self.beta_2)*gradient**2
#$#            mc = m/(1-self.beta_1**t)
#$#            vc = v/(1-self.beta_2**t)
#$#            delta = -1.*self.lr*mc/(tf.sqrt(vc) + self.eps)
#$#            #
#$#            states1, states2, states3 = states
#$#            concat_input = tf.stack([pos, delta], axis=-1)
#$#            cell_input1 = self.input_layer1(concat_input)
#$#            cell_input2 = self.input_layer2(cell_input1)
#$#            cell_input3 = self.input_layer3(cell_input2)
#$#            #
#$#            delta_pos1, new_states1 = self.cell1(cell_input1, states1)
#$#            delta_pos2, new_states2 = self.cell2(cell_input2, states2)
#$#            delta_pos3, new_states3 = self.cell3(cell_input3, states3)
#$#            #
#$#            delta_pos3 = self.output_layer3(delta_pos3)
#$#            delta_pos2 = tf.concat([delta_pos2, delta_pos3], axis=-1)
#$#            delta_pos2 = self.output_layer2(delta_pos2)
#$#            delta_pos1 = tf.concat([delta_pos1, delta_pos2], axis=-1)
#$#            delta_pos = self.output_layer1(delta_pos1)
#$#            #
#$#            new_pos = pos + delta_pos[..., 0]
#$#            new_states = [new_states1, new_states2, new_states3]
#$#            return i + 1 , new_pos, new_states, m, v
#$#
#$#        loss  = 0.
#$#        while tf.less(i, tf.constant(self.niter)):
#$#            i, curr_pos, curr_state, m, v =  body(i, curr_pos, curr_state, m, v)
#$#            loss = loss + tf.reduce_mean(tf.square(x_true - curr_pos))
#$#        return curr_pos, loss
#$#
#$#
#$#
#$#
#$#def build_rim_parallel3_single(params):
#$#
#$#    nc = params['nc']
#$#
#$#    cell1 = ConvLSTM3DCell(params['cell_size1'], kernel_size=params['cell_kernel_size1'], padding='SAME')
#$#    cell2 = ConvLSTM3DCell(params['cell_size2'], kernel_size=params['cell_kernel_size2'], padding='SAME')
#$#    cell3 = ConvLSTM3DCell(params['cell_size3'], kernel_size=params['cell_kernel_size3'], padding='SAME')
#$#    cells = [cell1, cell2, cell3]
#$#
#$#    input_layer1 = Conv3D(params['input_size1'], kernel_size=params['input_kernel_size1'], 
#$#                         trainable=True, padding='SAME', activation=params['input_activation'])
#$#    input_layer2 = Conv3D(params['input_size2'], kernel_size=params['input_kernel_size2'], strides= [params['strides']]*3,
#$#                         trainable=True, padding='SAME', activation=params['input_activation'])
#$#    input_layer3 = Conv3D(params['input_size3'], kernel_size=params['input_kernel_size3'], strides= [params['strides']]*3,
#$#                         trainable=True, padding='SAME', activation=params['input_activation'])
#$#    input_layers = [input_layer1, input_layer2, input_layer3]
#$#
#$#    output_layer1 = Conv3D(params['output_size1'], kernel_size=params['output_kernel_size1'], trainable=True, padding='SAME', 
#$#                          activation=params['output_activation'])   
#$#    output_layer2 = Conv3DTranspose(params['output_size2'], kernel_size=params['output_kernel_size2'], trainable=True, padding='SAME', 
#$#                          strides= [params['strides']]*3, activation=params['output_activation'])
#$#    output_layer3 = Conv3DTranspose(params['output_size3'], kernel_size=params['output_kernel_size3'], trainable=True, padding='SAME', 
#$#                          strides= [params['strides']]*3, activation=params['output_activation'])
#$#    output_layers = [output_layer1, output_layer2, output_layer3]
#$#   
#$#    rim = RIM3D_parallel3_single(cells, input_layers, output_layers, strides=params['strides'], niter=params['rim_iter'])
#$#
#$#    return rim
#$#





class myAdam(tf.keras.Model):

    def __init__(self, niter, lr=0.1):
        super(myAdam, self).__init__()
        self.niter = niter
        self.lr = lr
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.eps =  1e-7
            
            
    def call(self, x_init, y, grad_fn, grad_args=[], ):
        
        #outputs_ta = tf.TensorArray(size=self.niter+1, dtype=tf.float32)
            
        i = tf.constant(0, dtype=tf.int32)
        curr_pos = x_init        
        m = tf.zeros_like(x_init)
        v = tf.zeros_like(x_init)
        
        def body(i, pos, m, v):  
            gradient = grad_fn(pos, y, *grad_args)           
            #get_step = self.optimizer.apply_gradients(zip([gradient],[ pos]))

            t = tf.cast(i+1, tf.float32)
            m = self.beta_1*m + (1-self.beta_1)*gradient
            v = self.beta_2*v + (1-self.beta_2)*gradient**2
            mc = m/(1-self.beta_1**t)
            vc = v/(1-self.beta_2**t)
            delta = -1.*self.lr*mc/(np.sqrt(vc) + self.eps)

            new_pos = pos + delta
            return i +1 , new_pos, m, v
        
        while tf.less(i, tf.constant(self.niter)):
            #outputs_ta = outputs_ta.write(i, curr_pos)
            i, curr_pos,  m, v =  body(i, curr_pos,  m, v)
        #outputs_ta = outputs_ta.write(i, curr_pos)
        #return outputs_ta.stack()
        return curr_pos




