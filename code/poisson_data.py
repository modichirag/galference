##make a toy model halo data which is essentially a possion sample of the density field
##N_halo = Poisson(\lambda * (1 + \delta))

import numpy as np
import os
import math
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tf.disable_v2_behavior()
#import mesh_tensorflow as mtf


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
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from matplotlib import pyplot as plt
from utils import tools

cosmology=Planck15


tf.flags.DEFINE_integer("nc", 64, "Size of the cube")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")
tf.flags.DEFINE_float("box_size", 200, "Box Size")
tf.flags.DEFINE_integer("nx", 1, "# blocks along x")
tf.flags.DEFINE_integer("ny", 1, "# blocks along y")
tf.flags.DEFINE_integer("dsample", 2, "downsampling factor")
tf.flags.DEFINE_integer("hsize", 16, "halo size")
tf.flags.DEFINE_float("a0", 0.1, "initial scale factor")
tf.flags.DEFINE_float("af", 1.0, "final scale factor")
tf.flags.DEFINE_integer("nsteps", 5, "Number of time steps")
tf.flags.DEFINE_float("plambda", 0.01, "Multiplicative factor of Poisson lambda")
#tf.flags.DEFINE_string("mesh_shape", "row:nx;col:ny", "mesh shape")

FLAGS = tf.flags.FLAGS

    
def main(_):

      
    tf.random.set_random_seed(100)
    np.random.seed(100)

    mesh_shape = [("row", FLAGS.nx), ("col", FLAGS.ny)]

    layout_rules = [("nx_lr", "row"), ("ny_lr", "col"),
                    ("nx", "row"), ("ny", "col"),
                    ("ty_lr", "row"), ("tz_lr", "col"),
                    ("nx_block","row"), ("ny_block","col")]

                    

    mesh_size = FLAGS.nx * FLAGS.ny # mesh_shape.size
    mesh_devices = [""] * mesh_size
    #mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
    #  mesh_shape, layout_rules, mesh_devices)
   

    ## Create computational graphs and some initializations   
    #graph = mtf.Graph()
    #mesh = mtf.Mesh(graph, "nbody_mesh")

    # Compute a few things first, using simple tensorflow
    a0=FLAGS.a0
    a=FLAGS.af
    nsteps=FLAGS.nsteps
    bs, nc = FLAGS.box_size, FLAGS.nc
    klin = np.loadtxt('../data/Planck15_a1p00.txt').T[0]
    plin = np.loadtxt('../data/Planck15_a1p00.txt').T[1]
    ipklin = iuspline(klin, plin)
    stages = np.linspace(a0, a, nsteps, endpoint=True)

    #pt = PerturbationGrowth(cosmology, a=[a], a_normalize=1.0)
    # Generate a batch of 3D initial conditions
    initial_conditions = flowpm.linear_field(FLAGS.nc,          # size of the cube
                                             FLAGS.box_size,         # Physical size of the cube
                                             ipklin,      # Initial power spectrum
                                             batch_size=FLAGS.batch_size)

    state = lpt_init(initial_conditions, a0=a0, order=1) 
    #final_state = state
    final_state = nbody(state,  stages, nc)
    tfinal_field = cic_paint(tf.zeros_like(initial_conditions), final_state[0])

    plambda = FLAGS.plambda
    galmean = tfp.distributions.Poisson(rate = plambda * (1 + tfinal_field))
    result = galmean.sample()

    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)) as sess:
        a,b,c = sess.run([initial_conditions, tfinal_field, result])

    print('\nHalo sampled\n', c.sum(), c.min(), c.max(), c.mean())


    ##save and diagnostics
    try : os.makedirs('../data/poisson_N%03d'%nc)
    except Exception as e : print(e)
    
    np.save('../data/poisson_N%03d/ic'%nc, a)
    np.save('../data/poisson_N%03d/final'%nc, b)
    np.save('../data/poisson_N%03d/psample_%0.2f'%(nc, plambda), c)
    
    k, pi = tools.power(a[0], boxsize=bs)
    k, pf = tools.power(b[0], boxsize=bs)
    k, ph = tools.power(c[0], boxsize=bs)
    k, pxf = tools.power(b[0], f2=c[0], boxsize=bs)
    k, pxi = tools.power(a[0], f2=c[0], boxsize=bs)
    plt.plot(k, pi, label='IC')
    plt.plot(k, pf, label='Final')
    plt.plot(k, ph, label='Halo')
    plt.loglog()
    plt.grid(which = 'both')
    plt.legend()
    plt.ylim(10, 2e4)
    plt.savefig('../data/poisson_N%03d/fig_power_l%0.2f.png'%(nc, plambda))
    plt.close()

    plt.plot(k, pxf/(pf*ph)**0.5, label='Final')
    plt.plot(k, pxi/(pi*ph)**0.5, label='IC')
    plt.semilogx()
    plt.grid(which = 'both')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.savefig('../data/poisson_N%03d/fig_rcc_l%0.2f.png'%(nc, plambda))
    plt.close()
    


    plt.figure(figsize=(15,4))
    plt.subplot(131)
    plt.imshow(a[0].sum(axis=2))
    plt.title('Initial Conditions')

    plt.subplot(132)
    plt.imshow(b[0].sum(axis=2))
    plt.title('TensorFlow (single GPU)')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(c[0].sum(axis=2))
    plt.title('Mesh TensorFlow Single')
    plt.colorbar()

    plt.savefig("../data/poisson_N%03d/fig_field_l%0.2f.png"%(nc, plambda))

    exit(0)

if __name__ == "__main__":
  tf.app.run(main=main)
