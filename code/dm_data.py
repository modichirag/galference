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
#import flowpm.mesh_ops as mpm
#import flowpm.mtfpm as mtfpm
#import flowpm.mesh_utils as mesh_utils
import flowpm
from astropy.cosmology import Planck15
from flowpm.tfpm import PerturbationGrowth
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d

from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from matplotlib import pyplot as plt
from utils import tools

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


#Read in constants
klin = np.loadtxt('..//data/Planck15_a1p00.txt').T[0].astype(np.float32)
plin = np.loadtxt('..//data/Planck15_a1p00.txt').T[1].astype(np.float32)
ipklin = iuspline(klin, plin)


fpath = '../data/rim-data/L%04d_N%03d_T%02d/'%(bs, nc, nsteps)
try: os.makedirs(fpath)
except: print('Folder exists')


@tf.function()
def pm():
    print("PM graph")
    linear = flowpm.linear_field(nc, bs, ipklin, batch_size=batch_size)
    state = lpt_init(linear, a0=0.1, order=1)
    final_state = nbody(state,  stages, nc)
    tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
    return linear, tfinal_field


def main():

      
    #tf.random.set_random_seed(100)
    np.random.seed(100)

    ##Graph
    #pt = PerturbationGrowth(cosmology, a=[a], a_normalize=1.0)
    # Generate a batch of 3D initial conditions

    for i in range(nsims):
        
        ic, fin = pm()
        ic, fin = ic.numpy(), fin.numpy()
        tosave = np.concatenate([ic, fin], axis=0)
        np.save(fpath + '%04d'%i, tosave)
        
        if i == 0 or i == nsims-1:
            fig, ax = plt.subplots(1, 2, figsize=(7, 4))
            ax[0].imshow(ic[0].sum(axis=0))
            ax[1].imshow(fin[0].sum(axis=0))
            plt.savefig('%d.png'%i)

    exit(0)

if __name__ == "__main__":
  main()
