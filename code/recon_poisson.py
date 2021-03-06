import numpy as np
import os, sys
import math, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from matplotlib import pyplot as plt
import argparse

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf
import tensorflow_probability as tfp

import sys
#sys.path.pop(6)
import flowpm
from astropy.cosmology import Planck15
from flowpm.tfpm import PerturbationGrowth
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d


sys.path.append('../')
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

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nc', type=int, default=32, help='Grid size')
parser.add_argument('--bs', type=float, default=100, help='Box Size')
parser.add_argument('--niter', type=int, default=200, help='Number of iterations/Max iterations')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='adam', help='Which optimizer to use')
parser.add_argument('--plambda', type=float, default=0.10, help='Poisson probability')
parser.add_argument('--nbody', type=str2bool, default=True, help='Number of simulationss')
parser.add_argument('--lpt_order', type=int, default=1, help='Order of LPT Initial conditions')
parser.add_argument('--nsteps', type=int, default=3, help='Number of time steps')

args = parser.parse_args()


##Change things here
nc, bs = args.nc, args.bs
niter = args.niter
plambda = args.plambda
optimizer = args.optimizer
lr = args.lr
a0, af, nsteps = 0.1, 1.0, args.nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
anneal = True
RRs = [2, 1, 0.5, 0]



#Read in constants
klin = np.loadtxt('..//data/Planck15_a1p00.txt').T[0].astype(np.float32)
plin = np.loadtxt('..//data/Planck15_a1p00.txt').T[1].astype(np.float32)
ipklin = iuspline(klin, plin)
# Compute necessary Fourier kernels
kvec = tools.fftk((nc, nc, nc), boxsize=bs, symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
priorwt = ipklin(kmesh)


#fpath = "./tmp/poisson-%s-%d-p%0.2f/"%(optimizer, nc, plambda)
fpath = "./tmp/poisson-%d-p%0.2f/"%(nc, plambda)
for ff in [fpath]:
#for ff in [fpath, fpath + '/figs']:
    try: os.makedirs(ff)
    except Exception as e: print (e)



dtype=tf.float32


@tf.function
def pm(linear):
    if args.nbody:
        print('Nobdy sim')
        state = lpt_init(linear, a0=a0, order=args.lpt_order)
        final_state = nbody(state,  stages, nc)
    else:
        print('ZA/2LPT sim')
        final_state = lpt_init(linear, a0=af, order=args.lpt_order)
    tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
    return tfinal_field


def main():

    startw = time.time()

    if args.nbody: dpath = '/project/projectdirs/m3058/chmodi/rim-data/poisson_L%04d_N%03d_T%02d_p%03/'%(bs, nc, nsteps, plambda*100)
    else: dpath = '/project/projectdirs/m3058/chmodi/rim-data/poisson_L%04d_N%03d_LPT%d_p%03d/'%(bs, nc, args.lpt_order, plambda*100)
    ic, fin, data = np.load(dpath + '%04d.npy'%0)
    ic, fin, data = np.expand_dims(ic, 0), np.expand_dims(fin, 0), np.expand_dims(data, 0)
    print(ic.shape, fin.shape, data.shape)


    check = pm(tf.constant(ic)).numpy()
    print(fin/check)
    #ic = np.load('../data/poisson_L%04d_N%03d/ic.npy'%(bs, nc))
    #fin = np.load('../data/poisson_L%04d_N%03d/final.npy'%(bs, nc))
    #data = np.load('../data/poisson_L%04d_N%03d/psample_%0.2f.npy'%(bs, nc, plambda))


    
    @tf.function
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
        logprob = tf.multiply(logprob, 1/nc**3, name='logprob')


        #Prior
        lineark = r2c3d(linear, norm=nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
        prior = tf.multiply(prior, 1/nc**3, name='prior')
        #
        loss = logprob + prior

        return loss



    #Loop it Reconstruction
    ##Reconstruction
    x0 = np.random.normal(0, 1, nc**3).reshape(fin.shape).astype(np.float32) 
    linear = tf.Variable(name='linmesh', shape=(1, nc, nc, nc), dtype=tf.float32,
                             initial_value=x0, trainable=True)



    ##
    for iR, RR in enumerate(RRs):

        @tf.function
        def val_and_grad(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = recon_prototype(x, tf.constant(RR, dtype=tf.float32))
            grad = tape.gradient(loss, x)
            return loss, grad

        @tf.function
        def grad(x):
            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = recon_prototype(x, tf.constant(RR, dtype=tf.float32))
            grad = tape.gradient(loss, x)
            return grad


        start = time.time()

        #
        if optimizer == 'scipy-lbfgs':
            def func(x):
                return [vv.numpy().astype(np.float64)  for vv in val_and_grad(x=tf.constant(x, dtype=tf.float32))] # 

            results = sopt.minimize(fun=func, x0=x0, jac=True, method='L-BFGS-B', 
                                    #tol=1e-10, 
                                    options={'maxiter':niter, 'ftol': 1e-12, 'gtol': 1e-12, 'eps':1e-12})
                                    #options={'maxiter':niter})
            print(results)
            minic = results.x.reshape(data.shape)
       
        #
        elif optimizer == 'tf2-lbfgs':
            
            @tf.function
            def min_lbfgs(x0):
                return tfp.optimizer.lbfgs_minimize( val_and_grad, initial_position=x0, tolerance=1e-10, max_iterations=niter)
                
            results = min_lbfgs(x0.flatten())
            print(results)
            minic = results.position.numpy().reshape(data.shape)

        #    
        elif optimizer == 'adam':
            
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
            for i in range(niter):
                grads = grad([linear])
                opt.apply_gradients(zip(grads, [linear]))
            minic = linear.numpy().reshape(data.shape)

        #
        print('\nminimized\n')
        print("Time taken for maxiter %d : "%niter, time.time()-start)

        minfin = pm(tf.constant(minic, dtype=tf.float32)).numpy()
        dg.saveimfig("-R%d"%RR, [minic, minfin], [ic, fin], fpath+'')
        dg.save2ptfig("-R%d"%RR, [minic, minfin], [ic, fin], fpath+'', bs)
        ###
        x0 = minic


    exit(0)



if __name__ == "__main__":
  #tf.app.run(main=main)
  main()

  
