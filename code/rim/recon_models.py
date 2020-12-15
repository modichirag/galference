import numpy as np
import os, sys
import math, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
import tensorflow as tf
import tensorflow_probability as tfp

import flowpm
from astropy.cosmology import Planck15
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d

sys.path.append('../utils/')
import tools

cosmology=Planck15
np.random.seed(100)


class Recon_DM():
    
    def __init__(self, nc, bs, a0=0.1, af=1.0, nsteps=5, nbody=True, lpt_order=2, anneal=True):
        self.nc = nc
        self.bs = bs
        self.a0, self.af, self.nsteps = a0, af, nsteps
        self.stages = np.linspace(a0, af, nsteps, endpoint=True)
        self.nbody = nbody
        self.lpt_order = lpt_order
        self.anneal = True
        self.klin = np.loadtxt('../../data/Planck15_a1p00.txt').T[0].astype(np.float32)
        self.plin = np.loadtxt('../../data/Planck15_a1p00.txt').T[1].astype(np.float32)
        self.ipklin = iuspline(self.klin, self.plin)
        # Compute necessary Fourier kernels
        self.kvec = tools.fftk((nc, nc, nc), boxsize=bs, symmetric=False)
        self.kmesh = (sum(k**2 for k in self.kvec)**0.5).astype(np.float32)
        self.priorwt = self.ipklin(self.kmesh)
        self.R0 = tf.constant(0.)

    @tf.function
    def forward(self, linear):
        if self.nbody:
            print('Nobdy sim')
            state = lpt_init(linear, a0=self.a0, order=self.lpt_order)
            final_state = nbody(state,  self.stages, self.nc)
        else:
            print('ZA/2LPT sim')
            final_state = lpt_init(linear, a0=self.af, order=self.lpt_order)

        tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
        return tfinal_field


    @tf.function
    def loss_fn(self, linear, data, Rsm=tf.constant(0.)):
        """
        """
        bs, nc = self.bs, self.nc
        linear = tf.reshape(linear, data.shape)
        final_field = self.forward(linear)
        residual = final_field - data
        base = residual

        if self.anneal :
            print("\nAdd annealing section to graph\n")
            Rsmsq = tf.multiply(Rsm*bs/nc, Rsm*bs/nc)
            smwts = tf.exp(tf.multiply(-self.kmesh**2, Rsmsq))
            basek = r2c3d(base, norm=nc**3)
            basek = tf.multiply(basek, tf.cast(smwts, tf.complex64))
            base = c2r3d(basek, norm=nc**3)   

        chisq = tf.multiply(base, base)
        chisq = tf.reduce_sum(chisq)
        #chisq = tf.multiply(chisq, 1/nc**3, name='chisq')

        #Prior
        lineark = r2c3d(linear, norm=nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_sum(tf.multiply(priormesh, 1/self.priorwt))
        #prior = tf.multiply(prior, 1/nc**3, name='prior')
        #
        loss = chisq + prior
        return loss


    def reconstruct(self, data, RRs=[1.0, 0.0], niter=100, lr=0.01, x_init=None):

        print('reconstructing')
        bs, nc = self.bs, self.nc

        @tf.function
        def grad(x, Rsm):
            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = self.loss_fn(x, data, Rsm)
            grad = tape.gradient(loss, x)
            return grad

            
        # Create an optimizer for Adam.
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        ##Reconstruction
        if x_init is None: 
            x_init = np.random.normal(0, 1, nc**3).reshape(data.shape).astype(np.float32) 
        linear = tf.Variable(name='linmesh', shape=data.shape, dtype=tf.float32,
                                 initial_value=x_init, trainable=True)


        for iR, RR in enumerate(RRs):
            print('For smoothing scale : ', RR)
            for i in range(niter):
                grads = grad([linear], tf.constant(RR, dtype=tf.float32))
                opt.apply_gradients(zip(grads, [linear]))
            minic = linear.numpy().reshape(data.shape)
        #
        print('\nminimized\n')
        minfin = self.forward(tf.constant(minic, dtype=tf.float32)).numpy()
        
        return minic, minfin
