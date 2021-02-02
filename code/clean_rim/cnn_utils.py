import numpy as np
import tensorflow as tf
from convolutional_recurrent import ConvLSTM3DCell
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose
from tensorflow.python.keras.layers import MaxPool3D, AveragePooling3D, Dropout, BatchNormalization
from tensorflow.python.keras.layers import ReLU, LeakyReLU
from tensorflow.python.keras.activations import tanh, linear

import sys
sys.path.append('../../utils/')
import tools






class SimpleUNet(tf.keras.Model):

    def __init__(self, channels, dropout=0.95, kernel_size=3, strides=2):
        super(SimpleUNet, self).__init__()
        #self.nlayers = nlayers
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout=dropout
        self.strides = strides
        #self.activation = LeakyReLU()
        self.activation = tanh
        self._build()


    def _build(self):

        c = self.channels
        ksize = self.kernel_size
        self.l10, self.l11, self.l1 = Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear'), \
                     Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear'), \
                     Conv3D(c, kernel_size=1, padding='SAME', activation='linear')
        self.l12, self.l13, self.l1new = Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear'), \
                     Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear'), \
                     Conv3D(c, kernel_size=1, padding='SAME', activation='linear')
        #self.b10, self.b11, self.b1 = [BatchNormalization()]*3
        self.b10, self.b11, self.b1 = [linear]*3

        self.l20, self.l21, self.l2 = Conv3D(2*c, kernel_size=ksize, padding='SAME', activation='linear'), \
                     Conv3D(2*c, kernel_size=ksize, padding='SAME', activation='linear'), \
                     Conv3D(2*c, kernel_size=1, padding='SAME', activation='linear')
        self.l22, self.l23, self.l2new = Conv3D(2*c, kernel_size=ksize, padding='SAME', activation='linear'), \
                     Conv3D(2*c, kernel_size=ksize, padding='SAME', activation='linear'), \
                     Conv3D(2*c, kernel_size=1, padding='SAME', activation='linear')
        #self.b20, self.b21, self.b2 = [BatchNormalization()]*3
        self.b20, self.b21, self.b2 = [linear]*3

        
        
        self.l3in = Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear')
        self.l30, self.l31, self.l3 = Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear'), \
                     Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear'), \
                     Conv3D(c, kernel_size=1, padding='SAME', activation='linear')            
        self.l32, self.l33, self.l3new = Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear'), \
                     Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear'), \
                     Conv3D(c, kernel_size=1, padding='SAME', activation='linear')            
        #self.b30, self.b31, self.b3 = [BatchNormalization()]*3
        self.b30, self.b31, self.b3 = [linear]*3

        self.sub = Conv3D(2*c, kernel_size=ksize, padding='SAME', activation='linear', strides=[self.strides]*3)
        self.sup = Conv3DTranspose(c, kernel_size=ksize, padding='SAME', activation='linear', strides=[self.strides]*3)
        self.out0 = Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear')
        #self.outb = BatchNormalization()
        self.outb = linear
        self.out1 = Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear')
        

    def call(self, xinp, training=False):


        x0 = xinp
        #
        x = self.activation(self.b10(self.l10(x0)))
        x = self.activation(self.b11(self.l11(x)))
        x = self.b1(self.l1(x))
        x0 = x+x0 
        x = self.activation((self.l12(x0)))
        x = self.activation((self.l13(x)))
        x = (self.l1new(x))
        x = x+x0

        y = self.activation(x)
          
        x0 = self.sub(y)
        #
        x = self.activation(self.b20(self.l20(x0)))
        x = self.activation(self.b21(self.l21(x)))
        x = self.b2(self.l2(x))
        x0 = x+x0
        x = self.activation((self.l22(x0)))
        x = self.activation((self.l23(x)))
        x = (self.l2new(x))
        x = x+x0
        ysub = self.activation(x)

        x0 = self.sup(ysub)
        x0 = tf.concat([x0, y], axis=-1)
        x0 = (self.l3in(x0))
        #
        x = self.activation(self.b30(self.l30(x0)))
        x = self.activation(self.b31(self.l31(x)))
        x = self.b3(self.l3(x))
        x0 = x+x0
        x = self.activation((self.l32(x0)))
        x = self.activation((self.l33(x)))
        x = self.b3(self.l3new(x))
        x = x+x0
        y = self.activation(x)
     
        ##
        x = self.activation(self.outb(self.out0(y)))
        x = self.out1(x)
        #
        return x
        




###        x0 = xinp
###        #
###        x = Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear')(x0)
###        x = BatchNormalization()(x)
###        x = self.activation(x)
###        x = Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear')(x)
###        x = BatchNormalization()(x)
###        x = self.activation(x)
###        xr = Conv3D(c, kernel_size=1, padding='SAME', activation='linear')(x0)
###        xr = BatchNormalization()(xr)
###        x = x+xr
###        y = self.activation(x)
###          
###        x0 = Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear', strides=[self.strides]*3)(y)
###        #
###        x = Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear')(x0)
###        x = BatchNormalization()(x)
###        x = self.activation(x)
###        x = Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear')(x)
###        x = BatchNormalization()(x)
###        x = self.activation(x)
###        xr = Conv3D(c, kernel_size=1, padding='SAME', activation='linear')(x0)
###        xr = BatchNormalization()(xr)
###        x = x+xr
###        x = self.activation(x)
###        
###        x0 = Conv3DTranspose(c, kernel_size=ksize, padding='SAME', activation='linear', strides=[self.strides]*3)(x)
###        x0 = tf.concat([x0, y], axis=-1)
###        #print(x0.shape)
###        x = Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear')(x0)
###        x = BatchNormalization()(x)
###        x = self.activation(x)
###        x = Conv3D(c, kernel_size=ksize, padding='SAME', activation='linear')(x)
###        x = BatchNormalization()(x)
###        x = self.activation(x)
###        xr = Conv3D(c, kernel_size=1, padding='SAME', activation='linear')(x0)
###        xr = BatchNormalization()(xr)
###        x = x+xr
###        y = self.activation(x)
###        #
###        x0 = Conv3D(1, kernel_size=ksize, padding='SAME', activation='linear')(y)
###        x0 = self.activation(x0)
###        x = Conv3D(1, kernel_size=ksize, padding='SAME', activation='linear')(x0)
###        #
###        return x
###        
###
###
###
