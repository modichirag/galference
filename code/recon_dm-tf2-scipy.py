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
niter = 200

klin = np.loadtxt('..//data/Planck15_a1p00.txt').T[0].astype(np.float32)
plin = np.loadtxt('..//data/Planck15_a1p00.txt').T[1].astype(np.float32)
ipklin = iuspline(klin, plin)
# Compute necessary Fourier kernels
kvec = tools.fftk((nc, nc, nc), boxsize=nc, symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
priorwt = ipklin(kmesh)


fpath = "./tmp/tf2-scipy-%d/"%nc
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


    # Run normal flowpm to generate data
    try:
        ic, fin = np.load(fpath + 'ic.npy'), np.load(fpath + 'final.npy')
        print('Data loaded')
    except Exception as e:
        print('Exception occured', e)
        tfic = linear_field(nc, bs, ipklin, batch_size=1, seed=100, dtype=dtype)
        tfinal_field = pm(tfic)
        ic, fin = tfic.numpy(), tfinal_field.numpy()
        np.save(fpath + 'ic', ic)
        np.save(fpath + 'final', fin)

    print('\ndata constructed\n')
    

    noise = np.random.normal(0, 1, nc**3).reshape(fin.shape).astype(np.float32)
    data_noised = fin + noise
    data = data_noised



    def recon_prototype(linear, Rsm=0):
        """
        """

        linear = tf.reshape(linear, data.shape)
        #loss = tf.reduce_sum(tf.square(linear - minimum)) 
        final_field = pm(linear)
        #state = lpt_init(linear, a0=0.1, order=1)
        #final_state = nbody(state,  stages, nc)
        #final_field = cic_paint(tf.zeros_like(linear), final_state[0])

        residual = final_field - data.astype(np.float32)
        base = residual

        if anneal :
            print("\nAdd annealing section to graph\n")
            Rsmsq = tf.multiply(Rsm*bs/nc, Rsm*bs/nc)
            smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
            basek = r2c3d(base, norm=nc**3)
            basek = tf.multiply(basek, tf.cast(smwts, tf.complex64))
            base = c2r3d(basek, norm=nc**3)   
    ###
        chisq = tf.multiply(base, base)
        chisq = tf.reduce_sum(chisq)
        chisq = tf.multiply(chisq, 1/nc**3, name='chisq')

        #Prior
        lineark = r2c3d(linear, norm=nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
        prior = tf.multiply(prior, 1/nc**3, name='prior')
        #
        loss = chisq + prior

        return loss

    @tf.function
    def val_and_grad(x, Rsm):
        print("val and grad : ", x.shape)
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = recon_prototype(x, Rsm)
        grad = tape.gradient(loss, x)
        return loss, grad


    def func(x, RR):
        return [vv.numpy().astype(np.float64)  for vv in val_and_grad(tf.constant(x, dtype=tf.float32), Rsm=RR)] # 



    ##Reconstruction
    x0 = np.random.normal(0, 1, nc**3).reshape(fin.shape).astype(np.float32) 

    RRs = [2, 1, 0.5, 0]
    for iR, RR in enumerate(RRs):

        print("start shape : ", x0.shape)
        results = sopt.minimize(fun=func, x0=x0, args = RR, jac=True, method='L-BFGS-B', 
                                options={'maxiter':niter, 'ftol': 2.220446049250313e-09, 'gtol': 1e-10, 'eps':1e-10})
        #options={'disp': None, 'maxcor': 10,  'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None}

        print(results)
        ###
        minic = results.x.reshape(data.shape)
        print(minic.shape)
        print('\nminimized\n')
        minfin = pm(tf.constant(minic, dtype=tf.float32)).numpy()
        dg.saveimfig("-R%d"%RR, [minic, minfin], [ic, fin], fpath+'')
        dg.save2ptfig("-R%d"%RR, [minic, minfin], [ic, fin], fpath+'', bs)
        ###
        x0 = minic
    exit(0)



if __name__ == "__main__":
  #tf.app.run(main=main)
  main()

  
