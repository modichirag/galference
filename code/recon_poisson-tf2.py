import numpy as np
import os, sys
import math, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from matplotlib import pyplot as plt

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf
import tensorflow_probability as tfp
import mesh_tensorflow as mtf

import flowpm
import flowpm.mesh_ops as mpm
import flowpm.mtfpm as mtfpm
import flowpm.mesh_utils as mesh_utils
from astropy.cosmology import Planck15
from flowpm.tfpm import PerturbationGrowth
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d

sys.path.append('../utils/')
import tools
import diagnostics as dg

import contextlib
import functools

import scipy.optimize as sopt
##


cosmology=Planck15
np.random.seed(100)
#tf.random.set_random_seed(100)
cscratch = "../figs_recon/"


nc, bs = 64, 200
a0, a, nsteps = 0.1, 1.0, 5
stages = np.linspace(a0, a, nsteps, endpoint=True)
anneal = True
lr = 0.01
niter = 200
plambda = 0.1

klin = np.loadtxt('..//data/Planck15_a1p00.txt').T[0].astype(np.float32)
plin = np.loadtxt('..//data/Planck15_a1p00.txt').T[1].astype(np.float32)
ipklin = iuspline(klin, plin)
# Compute necessary Fourier kernels
kvec = tools.fftk((nc, nc, nc), boxsize=nc, symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
priorwt = ipklin(kmesh)


fpath = "./tmp/poisson-tf2-%d-p%0.2f/"%(nc, plambda)
for ff in [fpath]:
#for ff in [fpath, fpath + '/figs']:
    try: os.makedirs(ff)
    except Exception as e: print (e)



dtype=tf.float32


def pm(linear):
    state = lpt_init(linear, a0=0.1, order=1)
    final_state = nbody(state,  stages, nc)
    tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
    return tfinal_field


def main():

    startw = time.time()


    ic = np.load('../data/poisson_L%04d_N%03d/ic.npy'%(bs, nc))
    fin = np.load('../data/poisson_L%04d_N%03d/final.npy'%(bs, nc))
    data = np.load('../data/poisson_L%04d_N%03d/psample_%0.2f.npy'%(bs, nc, plambda))



    def recon_prototype(linear, Rsm=0):
        """
        """

        linear = tf.reshape(linear, data.shape)
        #loss = tf.reduce_sum(tf.square(linear - minimum)) 
        final_field = pm(linear)
        base = final_field

        if anneal:
            print('\nAdd annealing graph\n')
            Rsmsq = tf.multiply(Rsm*bs/nc, Rsm*bs/nc)
            smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
            basek = r2c3d(base, norm=nc**3)
            basek = tf.multiply(basek, tf.cast(smwts, tf.complex64))
            base = c2r3d(basek, norm=nc**3)   

        galmean = tfp.distributions.Poisson(rate = plambda * (1 + base))
        sample = galmean.sample()
        logprob = -tf.reduce_sum(galmean.log_prob(data))
        #logprob = tf.multiply(logprob, 1/nc**3, name='logprob')


        #Prior
        lineark = r2c3d(linear, norm=nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
        prior = tf.multiply(prior, 1/nc**3, name='prior')
        #
        loss = logprob + prior

        return loss
#


    x0 = np.random.normal(0, 1, nc**3).reshape(fin.shape).astype(np.float32)
    linear = tf.Variable(name='linmesh', shape=(1, nc, nc, nc), dtype=tf.float32,
                             initial_value=x0, trainable=True)
    # Create an optimizer.
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    vars = [linear]
    

    @tf.function
    def get_grad(x, RR):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = recon_prototype(x, RR)
        grad = tape.gradient(loss, x)
        return grad

    ##Reconstruction


    RRs = [2, 1, 0.5, 0]
    for iR, RR in enumerate(RRs):

        for i in range(niter):
            grads = get_grad(vars, RR)
            opt.apply_gradients(zip(grads, vars))
            
        ###
        minic = linear.numpy().reshape(data.shape)
        print(minic.shape)
        print('\nminimized\n')
        minfin = pm(tf.constant(minic, dtype=tf.float32)).numpy()
        dg.saveimfig("-R%d"%RR, [minic, minfin], [ic, fin], fpath+'')
        dg.save2ptfig("-R%d"%RR, [minic, minfin], [ic, fin], fpath+'', bs)
##


if __name__ == "__main__":
  #tf.app.run(main=main)
  main()

  
