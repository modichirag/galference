##make a toy model halo data which is essentially a possion sample of the density field
##N_halo = Poisson(\lambda * (1 + \delta))

import numpy as np
import os
import math
import tensorflow as tf
import tensorflow_probability as tfp
import argparse


import sys
#sys.path.pop(6)
sys.path.append('../')
sys.path.append('../utils/')
import flowpm
from astropy.cosmology import Planck15
from flowpm.tfpm import PerturbationGrowth
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d

from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from matplotlib import pyplot as plt
from utils import tools

cosmology=Planck15

cosmology=Planck15
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nc', type=int, default=32, help='Grid size')
parser.add_argument('--bs', type=float, default=100, help='Box Size')
parser.add_argument('--niter', type=int, default=200, help='Number of iterations/Max iterations')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='adam', help='Which optimizer to use')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--nsteps', type=int, default=3, help='Number of time steps')
parser.add_argument('--nsims', type=int, default=2, help='Number of simulationss')
parser.add_argument('--plambda', type=float, default=0.01, help='Multiplicative factor of Poisson lambda')
parser.add_argument('--Rsm', type=float, default=0.0, help="smoothing length of the gaussian for poisson sampling")

args = parser.parse_args()


##Change things here
nc, bs = args.nc, args.bs
niter = args.niter
optimizer = args.optimizer
lr = args.lr
batch_size = args.batch_size
a0, a, nsteps = 0.1, 1.0,  args.nsteps
stages = np.linspace(a0, a, nsteps, endpoint=True)
nsims = args.nsims
plambda = args.plambda
Rsm = args.Rsm

#Read in constants
klin = np.loadtxt('..//data/Planck15_a1p00.txt').T[0].astype(np.float32)
plin = np.loadtxt('..//data/Planck15_a1p00.txt').T[1].astype(np.float32)
ipklin = iuspline(klin, plin)

#smooth to regularize?
kk = tools.fftk([nc, nc, nc], bs, symmetric=False)
kmesh = sum(k**2 for k in kk)**0.5
smwts = tf.exp(tf.multiply(-kmesh**2, Rsm**2))


##Folder
##save and diagnostics
fpath  = '../data/rim-data/poisson_L%04d_N%03d_T%02d_p%03d/'%(bs, nc, nsteps, plambda*100)
try : os.makedirs(fpath)
except Exception as e : print(e)



@tf.function()
def pm_poisson():
    print("PM graph")
    linear = flowpm.linear_field(nc, bs, ipklin, batch_size=batch_size)
    state = lpt_init(linear, a0=0.1, order=1)
    final_state = nbody(state,  stages, nc)
    tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
    base = tfinal_field
    if Rsm != 0:
         basek = r2c3d(tfinal_field, norm=nc**3)
         basek = tf.multiply(basek, tf.cast(smwts, tf.complex64))
         base = c2r3d(basek, norm=nc**3)

    galmean = tfp.distributions.Poisson(rate = plambda * (1 + base))
    result = galmean.sample()
    return linear, tfinal_field, result, base

    
def main():

      
    np.random.seed(100)


    for i in range(nsims):
        
        a,b,c,d = pm_poisson()
        a, b, c = a.numpy(), b.numpy(), c.numpy()
        tosave = np.concatenate([a, b, c], axis=0)
        np.save(fpath + '%04d'%i, tosave)
        
        if i == 0 or i == nsims-1:
            print('\nMin and Max Halo sampled\n', c.min(), c.max())
            print('\nHalo number desnity ; %0.2e\n'%(c.sum()/bs**3))
            fig, ax = plt.subplots(1, 3, figsize=(10, 4))
            ax[0].imshow(a[0].sum(axis=0))
            ax[1].imshow(b[0].sum(axis=0))
            ax[2].imshow(c[0].sum(axis=0))
            plt.savefig(fpath + 'im-%d.png'%i)
            plt.close()

            a += 1.
            k, pi = tools.power(a[0], boxsize=bs)
            k, pf = tools.power(b[0], boxsize=bs)
            k, ph = tools.power(c[0], boxsize=bs)
            k, pxf = tools.power(b[0], f2=c[0], boxsize=bs)
            k, pxi = tools.power(a[0], f2=c[0], boxsize=bs)
            plt.plot(k, pi, label='IC')
            plt.plot(k, pf, label='Final')
            plt.plot(k, ph, label='Poisson')
            plt.loglog()
            plt.grid(which = 'both')
            plt.legend()
            plt.ylim(10, 5e4)
            plt.savefig(fpath + 'power-%d.png'%i)
            plt.close()

            plt.plot(k, pxf/(pf*ph)**0.5, label='Final')
            plt.plot(k, pxi/(pi*ph)**0.5, label='IC')
            plt.semilogx()
            plt.grid(which = 'both')
            plt.legend()
            plt.ylim(-0.1, 1.1)
            plt.savefig(fpath + 'rcc-%d.png'%i)
            plt.close()


    exit(0)

if __name__ == "__main__":
    main()
