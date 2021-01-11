import numpy as np
import tensorflow as tf
from convolutional_recurrent import ConvLSTM3DCell
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose, MaxPool3D, AveragePooling3D

import sys
sys.path.append('../../utils/')
import tools


class RIM3D(tf.keras.Model):

    def __init__(self, cell, input_layer, output_layer, niter):
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



def build_rim(params):

    nc = params['nc']
    input_layer = Conv3D(params['input_size'], kernel_size=params['input_kernel_size'], 
                         trainable=True, padding='SAME',
                         input_shape=(None, nc, nc, nc, 2), activation=params['input_activation'])

    cell = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')
    cell.build(input_shape=[None, nc, nc, nc, params['input_size']])

    output_layer = Conv3D(1, kernel_size=params['output_kernel_size'], trainable=True, padding='SAME', 
                          input_shape=(None, nc, nc, nc, params['cell_size']), activation=params['output_activation'])
   
    rim = RIM3D(cell, input_layer, output_layer, niter=params['rim_iter'])

    return rim


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

    #input_layer_sub = MaxPool3D(padding='SAME')
    #input_layer_sub = AveragePooling3D(padding='SAME')
    cell1 = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')
    #cell1.build(input_shape=[None, nc, nc, nc, params['input_size']])

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




    
##def build_rim_series(params):
##
##    nc = params['nc']
##    input_layer = Conv3D(params['input_size'], kernel_size=params['input_kernel_size'], 
##                         trainable=True, padding='SAME', strides= [params['strides']]*3,
##                         input_shape=(None, nc, nc, nc, 2), activation=params['input_activation'])
##
##    cell1 = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')
##    #cell1.build(input_shape=[None, nc, nc, nc, params['input_size']])
##
##    middle_layer = Conv3DTranspose(params['middle_size'], kernel_size=params['middle_kernel_size'], 
##                         trainable=True, padding='SAME', strides=[params['strides']]*3, 
##                         activation=params['input_activation'])
##
##    cell2 = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')
##
##    output_layer = Conv3D(1, kernel_size=params['output_kernel_size'], trainable=True, padding='SAME', 
##                          input_shape=(None, nc, nc, nc, params['cell_size']), activation=params['output_activation'])
##   
##    rim = RIM3D_series(cell1, cell2, input_layer, middle_layer, output_layer, strides=params['strides'],
##                       niter=params['rim_iter'])
##
##    return rim
##




def get_bspline_kernel(x, num_channels, transpose=False, dtype=tf.float32, order=4):
  """Creates a 5x5x5 b-spline kernel.
  Args:
    num_channels: The number of channels of the image to filter.
    dtype: The type of an element in the kernel.
  Returns:
    A tensor of shape `[5, 5, 5, num_channels, num_channels]`.
  """
  in_dim = x.shape[-1]
  if order == 8:
    kernel = np.array(( 1., 8., 28., 56., 70., 56., 28., 8., 1.), dtype=dtype.as_numpy_dtype())
  elif order == 6:
    kernel = np.array(( 1., 6., 15., 20., 15., 6., 1.), dtype=dtype.as_numpy_dtype())
  elif order==2:
    kernel = np.array(( 1., 2., 1.), dtype=dtype.as_numpy_dtype())
  else:
    kernel = np.array(( 1., 4., 6., 4., 1.), dtype=dtype.as_numpy_dtype())
  size = len(kernel)
  kernel = np.einsum('ij,k->ijk', np.outer(kernel, kernel), kernel)
  kernel /= np.sum(kernel)
  kernel = kernel[:, :, :, np.newaxis, np.newaxis]
  kernel = tf.constant(kernel, dtype=dtype) * tf.eye(num_channels, dtype=dtype)

  return kernel

  #fd_dim = mtf.Dimension("fd", size)
  #fh_dim = mtf.Dimension("fh", size)
  #fw_dim = mtf.Dimension("fw", size)
  #if transpose:
  #  return mtf.import_tf_tensor(mesh, kernel, shape=[fd_dim, fh_dim, fw_dim, channels, in_dim])
  #else:
  #  return mtf.import_tf_tensor(mesh, kernel, shape=[fd_dim, fh_dim, fw_dim, in_dim, channels])


def downsample(field, downsampling_factor=2, antialias=True):
  """
  Performs a multiresolution decomposition of the input field.

  The input field will be decomposed into a low resolution approximation,
  and a details component.
  """
  low = field
  for i in range(downsampling_factor):
    kernel = get_bspline_kernel(low, low.shape[-1], order=6)
    low = tf.nn.conv3d(low, kernel, strides=(1,2,2,2,1), padding='SAME')
  if antialias:
    kernel = get_bspline_kernel(low, low.shape[-1], order=2)
    low = tf.nn.conv3d(low, kernel, strides=(1,1,1,1,1), padding='SAME')
  return low

def upsample(low, output_shape, downsampling_factor=2):
  """
  Performs a multiresolution reconstruction of the input field.

  The input field will be decomposed into a low resolution approximation,
  and a details component.
  """
  for i in range(downsampling_factor):
    kernel = get_bspline_kernel(low, low.shape[-1], transpose=True, order=6)
    #kernel = mesh_kernels.get_bspline_kernel(low, mtf.Dimension('out_%d'%i,low.shape[-1].size), transpose=True, order=6)
    high = tf.nn.conv3d_transpose(low, kernel * 2.0**3, strides=(1,2,2,2,1), padding='SAME', output_shape=output_shape)
  return high

def split_scales(field, downsampling_factor=2, antialias=True):
  """
  Performs a multiresolution decomposition of the input field.

  The input field will be decomposed into a low resolution approximation,
  and a details component.
  """
  low = downsample(field, downsampling_factor, antialias)
  high = upsample(low, field.shape, downsampling_factor)
  high = field - high #mtf.reshape(high, field.shape)
  return low, high





class RIM3D_split(tf.keras.Model):

    def __init__(self, cell1, cell2, input_layer, input_layer_sub, output_layer_sub, output_layer, strides, niter):
        super(RIM3D_split, self).__init__()
        self.cell1 = cell1
        self.cell2 = cell2
        self.output_layer = output_layer
        self.output_layer_sub = output_layer_sub
        self.input_layer = input_layer
        self.input_layer_sub = input_layer_sub
        self.strides = strides
        self.niter = niter
        self.beta_1, self.beta_2 = 0.9, 0.999
        self.lr, self.eps = 0.1, 1e-7
            
            
    def call(self, x_init, y, grad_fn, grad_args=[], initstates = None, return_steps=False):

        outputs_ta = tf.TensorArray(size=self.niter+1, dtype=tf.float32)

        
        if initstates is None: 
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
            #low, high = split_scales(pos, 1)
            #lowd, highd = split_scales(delta, 1)
            #concat_input = tf.stack([high, highd], axis=-1)
            #concat_input_sub = tf.stack([low, lowd], axis=-1)
            
            concat_input = tf.stack([pos, delta], axis=-1)
            low, high = split_scales(concat_input, 1)
            concat_input = high
            concat_input_sub = low

            #
            cell_input_sub = self.input_layer_sub(concat_input_sub)
            delta_pos1, new_states1 = self.cell1(cell_input_sub, states1)
            delta_pos1 = self.output_layer_sub(delta_pos1) 
            delta_pos1 = upsample(delta_pos1, pos.shape + [1], 1)
            #
            cell_input = self.input_layer(concat_input)
            delta_pos2, new_states2 = self.cell2(cell_input, states2)
            delta_pos2 = self.output_layer(delta_pos2)
            #
            delta_pos = delta_pos1 + delta_pos2
            #delta_pos = tf.concat([delta_pos1, delta_pos2], axis=-1)
            #delta_pos = self.output_layer(delta_pos)
            new_pos = pos + delta_pos[..., 0]
            new_states = [new_states1, new_states2]
            return i +1 , new_pos, new_states, m, v


        while tf.less(i, tf.constant(self.niter)):
            outputs_ta = outputs_ta.write(i, curr_pos)
            i, curr_pos, curr_state, m, v =  body(i, curr_pos, curr_state, m, v)
        outputs_ta = outputs_ta.write(i, curr_pos)
        return outputs_ta.stack()




def build_rim_split(params):

    nc = params['nc']
    input_layer = Conv3D(params['input_size'], kernel_size=params['input_kernel_size'], trainable=True, padding='SAME', 
                         input_shape=(None, nc, nc, nc, 2), activation=params['input_activation'])

    input_layer_sub = Conv3D(params['input_size'], kernel_size=params['input_kernel_size'], 
                             trainable=True, padding='SAME',  activation=params['input_activation'])

    cell1 = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')

    output_layer_sub = Conv3D(1, kernel_size=params['output_kernel_size'], 
                         trainable=True, padding='SAME', activation=params['output_activation'])

    cell2 = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')

    output_layer = Conv3D(1, kernel_size=params['output_kernel_size'], trainable=True, padding='SAME', 
                          input_shape=(None, nc, nc, nc, params['cell_size']*2), activation=params['output_activation'])
   
    rim = RIM3D_split(cell1, cell2, input_layer, input_layer_sub, output_layer_sub, output_layer, strides=params['strides'],
                       niter=params['rim_iter'])

    return rim




class RIM3D_split_single(tf.keras.Model):

    def __init__(self, cell1, cell2, input_layer, input_layer_sub, output_layer_sub, output_layer, strides, niter):
        super(RIM3D_split, self).__init__()
        self.cell1 = cell1
        self.cell2 = cell2
        self.output_layer = output_layer
        self.output_layer_sub = output_layer_sub
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
            ####
            states1, states2 = states
            
            concat_input = tf.stack([pos, delta], axis=-1)
            low, high = split_scales(concat_input, 1)
            concat_input = high
            concat_input_sub = low

            #
            cell_input_sub = self.input_layer_sub(concat_input_sub)
            delta_pos1, new_states1 = self.cell1(cell_input_sub, states1)
            delta_pos1 = self.output_layer_sub(delta_pos1) 
            delta_pos1 = upsample(delta_pos1, pos.shape + [1], 1)
            #
            cell_input = self.input_layer(concat_input)
            delta_pos2, new_states2 = self.cell2(cell_input, states2)
            delta_pos2 = self.output_layer(delta_pos2)
            #
            delta_pos = delta_pos1 + delta_pos2
            #delta_pos = tf.concat([delta_pos1, delta_pos2], axis=-1)
            #delta_pos = self.output_layer(delta_pos)
            new_pos = pos + delta_pos[..., 0]
            new_states = [new_states1, new_states2]
            return i +1 , new_pos, new_states, m, v

        loss  = 0.
        while tf.less(i, tf.constant(self.niter)):
            i, curr_pos, curr_state, m, v =  body(i, curr_pos, curr_state, m, v)
            loss = loss + tf.reduce_mean(tf.square(x_true - curr_pos))
        return curr_pos, loss




def build_rim_split_single(params):

    nc = params['nc']
    input_layer = Conv3D(params['input_size'], kernel_size=params['input_kernel_size'], 
                         trainable=True, padding='SAME', 
                         input_shape=(None, nc, nc, nc, 2), activation=params['input_activation'])

    input_layer_sub = Conv3D(params['input_size'], kernel_size=params['input_kernel_size'], 
                             trainable=True, padding='SAME',  activation=params['input_activation'])

    cell1 = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')


    output_layer_sub = Conv3D(1, kernel_size=params['output_kernel_size'], 
                         trainable=True, padding='SAME', activation=params['output_activation'])

    cell2 = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')

    output_layer = Conv3D(1, kernel_size=params['output_kernel_size'], trainable=True, padding='SAME', 
                          input_shape=(None, nc, nc, nc, params['cell_size']*2), activation=params['output_activation'])
   
    rim = RIM3D_split(cell1, cell2, input_layer, input_layer_sub, output_layer_sub, output_layer, strides=params['strides'],
                       niter=params['rim_iter'])

    return rim

