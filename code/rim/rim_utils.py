import numpy as np
import tensorflow as tf
from convolutional_recurrent import ConvLSTM3DCell
from tensorflow.python.keras.layers import Conv3D


class RIM3D(tf.keras.Model):

    def __init__(self, cell, output_layer, input_layer, niter):
        super(RIM3D, self).__init__()
        self.cell = cell
        self.output_layer = output_layer
        self.input_layer = input_layer
        self.niter = niter
        self.beta_1, self.beta_2 = 0.9, 0.999
        self.lr, self.eps = 0.1, 1e-7
            
            
    def call(self, x_init, y, grad_fn, grad_args=[], initstates = None, return_steps=False):
        
        outputs_ta = tf.TensorArray(size=self.niter+1, dtype=tf.float32)
        states_ta = tf.TensorArray(size=self.niter+1, dtype=tf.float32)
        
        if initstates is None: 
            stateshape = x_init.shape + tuple([self.cell.filters])
            initstates = [tf.zeros(stateshape), tf.zeros(stateshape)]
    
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

            concat_input = tf.stack([pos, delta], axis=-1)
            cell_input = self.input_layer(concat_input)            
            delta_pos, new_state = self.cell(cell_input, states)
            delta_pos = self.output_layer(delta_pos)[...,0]
            new_pos = pos + delta_pos
            return i +1 , new_pos, new_state, m, v
        
        while tf.less(i, tf.constant(self.niter)):
            outputs_ta = outputs_ta.write(i, curr_pos)
            states_ta = states_ta.write(i, curr_state)
            i, curr_pos, curr_state, m, v =  body(i, curr_pos, curr_state, m, v)
        outputs_ta = outputs_ta.write(i, curr_pos)
        states_ta = states_ta.write(i, curr_state)
        return outputs_ta.stack(), states_ta.stack()




class myAdam(tf.keras.Model):

    def __init__(self, niter):
        super(myAdam, self).__init__()
        self.niter = niter
        self.beta_1, self.beta_2 = 0.9, 0.999
        self.lr, self.eps = 0.1, 1e-7
            
            
    def call(self, x_init, y, grad_fn, grad_args=[], ):
        
        outputs_ta = tf.TensorArray(size=self.niter+1, dtype=tf.float32)
            
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
            outputs_ta = outputs_ta.write(i, curr_pos)
            i, curr_pos,  m, v =  body(i, curr_pos,  m, v)
        outputs_ta = outputs_ta.write(i, curr_pos)
        return outputs_ta.stack()



def build_rim(params):

    nc = params['nc']
    input_layer = Conv3D(params['input_size'], kernel_size=params['input_kernel_size'], 
                         trainable=True, padding='SAME',
                         input_shape=(None, nc, nc, nc, 2))

    cell = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')
    cell.build(input_shape=[None, nc, nc, nc, params['input_size']])

    output_layer = Conv3D(1, kernel_size=params['output_kernel_size'], trainable=True, padding='SAME', 
                         input_shape=(None, nc, nc, nc, params['cell_size']))
   
    rim = RIM3D(cell, output_layer, input_layer, niter=params['rim_iter'])

    return rim
