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


#tf.flags.DEFINE_integer("gpus_per_node", 8, "Number of GPU on each node")
#tf.flags.DEFINE_integer("gpus_per_task", 8, "Number of GPU in each task")
#tf.flags.DEFINE_integer("tasks_per_node", 1, "Number of task in each node")
#
tf.flags.DEFINE_integer("nc", 64, "Size of the cube")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")
tf.flags.DEFINE_float("box_size", 200, "Batch Size")
tf.flags.DEFINE_float("a0", 0.1, "initial scale factor")
tf.flags.DEFINE_float("af", 1.0, "final scale factor")
tf.flags.DEFINE_integer("nsteps", 2, "Number of time steps")
tf.flags.DEFINE_bool("nbody", True, "Do nbody evolution")
tf.flags.DEFINE_string("suffix", "-lbfgs", "suffix for the folder name")
tf.flags.DEFINE_bool("anneal", True, "Anneal")
tf.flags.DEFINE_float("lr", 0.01, "Learning rate")
tf.flags.DEFINE_integer("niter", 100, "Number of iterations")

FLAGS = tf.flags.FLAGS

nc, bs = FLAGS.nc, FLAGS.box_size
a0, a, nsteps =FLAGS.a0, FLAGS.af, FLAGS.nsteps

klin = np.loadtxt('..//data/Planck15_a1p00.txt').T[0].astype(np.float32)
plin = np.loadtxt('..//data/Planck15_a1p00.txt').T[1].astype(np.float32)
ipklin = iuspline(klin, plin)
# Compute necessary Fourier kernels
#kvec = flowpm.kernels.fftk((nc, nc, nc), symmetric=False)
kvec = tools.fftk((nc, nc, nc), boxsize=nc, symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
priorwt = ipklin(kmesh)
stages = np.linspace(a0, a, nsteps, endpoint=True)


fpath = "./tmp/L%04d_N%03d"%(bs, nc)
if FLAGS.anneal: fpath = fpath + '-anneal'
fpath = fpath + '%s/'%FLAGS.suffix
print(fpath)
for ff in [fpath]:
#for ff in [fpath, fpath + '/figs']:
    try: os.makedirs(ff)
    except Exception as e: print (e)



def make_val_and_grad_fn(value_fn):
  @functools.wraps(value_fn)
  def val_and_grad(x):
    return tfp.math.value_and_gradient(value_fn, x)
  return val_and_grad


@contextlib.contextmanager
def timed_execution():
  t0 = time.time()
  yield
  dt = time.time() - t0
  print('Evaluation took: %f seconds' % dt)


def np_value(tensor):
  """Get numpy value out of possibly nested tuple of tensors."""
  if isinstance(tensor, tuple):
    return type(tensor)(*(np_value(t) for t in tensor))
  else:
    #return tensor.numpy()
    return tensor.value()

def run(optimizer):
  """Run an optimizer and measure it's evaluation time."""
  optimizer()  # Warmup.
  with timed_execution():
    result = optimizer()
  return np_value(result)


dtype=tf.float32

startw = time.time()



# Compute a few things first, using simple tensorflow
a0=FLAGS.a0
a=FLAGS.af
nsteps=FLAGS.nsteps
bs, nc = FLAGS.box_size, FLAGS.nc
klin = np.loadtxt('../data/Planck15_a1p00.txt').T[0]
plin = np.loadtxt('../data/Planck15_a1p00.txt').T[1]
ipklin = iuspline(klin, plin)
stages = np.linspace(a0, a, nsteps, endpoint=True)

#tf.reset_default_graph()
# Run normal flowpm to generate data
try:
    ic, fin = np.load(fpath + 'ic.npy'), np.load(fpath + 'final.npy')
    print('Data loaded')
except Exception as e:
    print('Exception occured', e)
    tfic = linear_field(FLAGS.nc, FLAGS.box_size, ipklin, batch_size=1, seed=100, dtype=dtype)
    if FLAGS.nbody:
        state = lpt_init(tfic, a0=0.1, order=1)
        final_state = nbody(state,  stages, FLAGS.nc)
    else:
        final_state = lpt_init(tfic, a0=stages[-1], order=1)
    tfinal_field = cic_paint(tf.zeros_like(tfic), final_state[0])
    with tf.Session() as sess:
        ic, fin  = sess.run([tfic, tfinal_field])
    np.save(fpath + 'ic', ic)
    np.save(fpath + 'final', fin)


################################################################
#tf.reset_default_graph()
print('ic constructed')

noise = np.random.normal(0, 1, nc**3).reshape(fin.shape).astype(np.float32)
data_noised = fin + noise
data = data_noised

minimum = data.copy()
start = noise.copy().flatten().astype(np.float32)


def recon_prototype(linear):
    """
    """

    linear = tf.reshape(linear, minimum.shape)
    #loss = tf.reduce_sum(tf.square(linear - minimum)) 

    state = lpt_init(linear, a0=0.1, order=1)
    final_state = nbody(state,  stages, FLAGS.nc)
    final_field = cic_paint(tf.zeros_like(linear), final_state[0])

    residual = final_field - data.astype(np.float32)
    base = residual
    Rsm = tf.placeholder(tf.float32, name='smoothing')
    if FLAGS.anneal :
        print("\nAdd annealing section to graph\n")
        Rsmsq = tf.multiply(Rsm*bs/nc, Rsm*bs/nc)
        smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
        basek = r2c3d(base, norm=nc**3)
        basek = tf.multiply(basek, tf.cast(smwts, tf.complex64))
        base = c2r3d(basek, norm=nc**3)   
#
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
def val_and_grad(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = recon_prototype(x)
    grad = tape.gradient(loss, x)
    return loss, grad

def func(x):
    return [vv.numpy().astype(np.float64)  for vv in val_and_grad(tf.constant(x, dtype=tf.float32))]


results = sopt.minimize(fun=func, x0=start, jac=True, method='L-BFGS-B', options={'maxiter':50})

print(results)
minimum = results.position
print(minimum)
print('\nminimized\n')

tf.reset_default_graph()

tfic = linear_field(FLAGS.nc, FLAGS.box_size, ipklin, batch_size=1, seed=100, dtype=dtype)*0 + minimum.reshape(data_noised.shape)
state = lpt_init(tfic, a0=0.1, order=1)
final_state = nbody(state,  stages, FLAGS.nc)
tfinal_field = cic_paint(tf.zeros_like(tfic), final_state[0])
with tf.Session() as sess:
    minic, minfin  = sess.run([tfic, tfinal_field])

dg.saveimfig(0, [minic, minfin], [ic, fin], fpath+'')
dg.save2ptfig(0, [minic, minfin], [ic, fin], fpath+'', bs)


##  
##
##def main(_):
##
##    dtype=tf.float32
##
##    startw = time.time()
##
##    tf.random.set_random_seed(100)
##    np.random.seed(100)
##
##    
##    # Compute a few things first, using simple tensorflow
##    a0=FLAGS.a0
##    a=FLAGS.af
##    nsteps=FLAGS.nsteps
##    bs, nc = FLAGS.box_size, FLAGS.nc
##    klin = np.loadtxt('../data/Planck15_a1p00.txt').T[0]
##    plin = np.loadtxt('../data/Planck15_a1p00.txt').T[1]
##    ipklin = iuspline(klin, plin)
##    stages = np.linspace(a0, a, nsteps, endpoint=True)
##
##    #tf.reset_default_graph()
##    # Run normal flowpm to generate data
##    try:
##        ic, fin = np.load(fpath + 'ic.npy'), np.load(fpath + 'final.npy')
##        print('Data loaded')
##    except Exception as e:
##        print('Exception occured', e)
##        tfic = linear_field(FLAGS.nc, FLAGS.box_size, ipklin, batch_size=1, seed=100, dtype=dtype)
##        if FLAGS.nbody:
##            state = lpt_init(tfic, a0=0.1, order=1)
##            final_state = nbody(state,  stages, FLAGS.nc)
##        else:
##            final_state = lpt_init(tfic, a0=stages[-1], order=1)
##        tfinal_field = cic_paint(tf.zeros_like(tfic), final_state[0])
##        with tf.Session() as sess:
##            ic, fin  = sess.run([tfic, tfinal_field])
##        np.save(fpath + 'ic', ic)
##        np.save(fpath + 'final', fin)
##
##
##    ################################################################
##    #tf.reset_default_graph()
##    print('ic constructed')
##
##    noise = np.random.normal(0, 1, nc**3).reshape(fin.shape).astype(np.float32)
##    data_noised = fin + noise
##    data = data_noised
##
##    minimum = data.copy()
##    start = noise.copy().flatten().astype(np.float32)
##
##
##    def recon_prototype(linear):
##        """
##        """
##        
##        linear = tf.reshape(linear, minimum.shape)
##        #loss = tf.reduce_sum(tf.square(linear - minimum)) 
##
##        state = lpt_init(linear, a0=0.1, order=1)
##        final_state = nbody(state,  stages, FLAGS.nc)
##        final_field = cic_paint(tf.zeros_like(linear), final_state[0])
##
##        residual = final_field - data.astype(np.float32)
##        base = residual
##        Rsm = tf.placeholder(tf.float32, name='smoothing')
##        if FLAGS.anneal :
##            print("\nAdd annealing section to graph\n")
##            Rsmsq = tf.multiply(Rsm*bs/nc, Rsm*bs/nc)
##            smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
##            basek = r2c3d(base, norm=nc**3)
##            basek = tf.multiply(basek, tf.cast(smwts, tf.complex64))
##            base = c2r3d(basek, norm=nc**3)   
##    #
##        chisq = tf.multiply(base, base)
##        chisq = tf.reduce_sum(chisq)
##        chisq = tf.multiply(chisq, 1/nc**3, name='chisq')
##
##        #Prior
##        lineark = r2c3d(linear, norm=nc**3)
##        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
##        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
##        prior = tf.multiply(prior, 1/nc**3, name='prior')
##        #
##        loss = chisq + prior
##
##        return loss
##
##    @tf.function
##    def val_and_grad(x):
##        with tf.GradientTape() as tape:
##            tape.watch(x)
##            loss = recon_prototype(x)
##        grad = tape.gradient(loss, x)
##        return loss, grad
##
##    def func(x):
##        return [vv.numpy().astype(np.float64)  for vv in val_and_grad(tf.constant(x, dtype=tf.float32))]
##    
##
##    results = sopt.minimize(fun=func, x0=start, jac=True, method='L-BFGS-B', options={'maxiter':50})
##
##    print(results)
##    minimum = results.position
##    print(minimum)
##    print('\nminimized\n')
##
##    tf.reset_default_graph()
##
##    tfic = linear_field(FLAGS.nc, FLAGS.box_size, ipklin, batch_size=1, seed=100, dtype=dtype)*0 + minimum.reshape(data_noised.shape)
##    state = lpt_init(tfic, a0=0.1, order=1)
##    final_state = nbody(state,  stages, FLAGS.nc)
##    tfinal_field = cic_paint(tf.zeros_like(tfic), final_state[0])
##    with tf.Session() as sess:
##        minic, minfin  = sess.run([tfic, tfinal_field])
##
##    dg.saveimfig(0, [minic, minfin], [ic, fin], fpath+'')
##    dg.save2ptfig(0, [minic, minfin], [ic, fin], fpath+'', bs)
##    
##    
####
##    exit(0)
##
##if __name__ == "__main__":
##  tf.app.run(main=main)
##
##  
