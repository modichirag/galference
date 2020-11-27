import numpy as np
import os, sys
import math, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from matplotlib import pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
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

##


cosmology=Planck15
np.random.seed(100)
tf.random.set_random_seed(200)
cscratch = "../figs_recon/"

#
tf.flags.DEFINE_integer("nc", 64, "Size of the cube")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")
tf.flags.DEFINE_float("box_size", 200, "Batch Size")
tf.flags.DEFINE_float("a0", 0.1, "initial scale factor")
tf.flags.DEFINE_float("af", 1.0, "final scale factor")
tf.flags.DEFINE_integer("nsteps", 3, "Number of time steps")
tf.flags.DEFINE_bool("nbody", True, "Do nbody evolution")
tf.flags.DEFINE_string("suffix", "-lbfgs", "suffix for the folder name")
tf.flags.DEFINE_bool("anneal", True, "Anneal")
tf.flags.DEFINE_float("lr", 0.01, "Learning rate")
tf.flags.DEFINE_integer("niter", 100, "Number of iterations")
tf.flags.DEFINE_float("plambda", 0.10, "Lambda of poisson probability")

FLAGS = tf.flags.FLAGS

nc, bs = FLAGS.nc, FLAGS.box_size
a0, a, nsteps =FLAGS.a0, FLAGS.af, FLAGS.nsteps
plambda = FLAGS.plambda

klin = np.loadtxt('..//data/Planck15_a1p00.txt').T[0].astype(np.float32)
plin = np.loadtxt('..//data/Planck15_a1p00.txt').T[1].astype(np.float32)
ipklin = iuspline(klin, plin)
# Compute necessary Fourier kernels
#kvec = flowpm.kernels.fftk((nc, nc, nc), symmetric=False)
kvec = tools.fftk((nc, nc, nc), boxsize=nc, symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
priorwt = ipklin(kmesh)
stages = np.linspace(a0, a, nsteps, endpoint=True)
print(stages)


fpath = "./tmp/poisson_L%04d_N%03d_p%0.02f"%(bs, nc, plambda)
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


def pmgraph(lin):
    state = lpt_init(lin, a0=0.1, order=1)
    final_state = nbody(state,  stages, FLAGS.nc)
    final_field = cic_paint(tf.zeros_like(lin), final_state[0])
    return final_field

##############################################

def main(_):

    dtype=tf.float32

    startw = time.time()

    tf.random.set_random_seed(100)
    np.random.seed(100)

    
    # Compute a few things first, using simple tensorflow
    a0=FLAGS.a0
    a=FLAGS.af
    nsteps=FLAGS.nsteps
    bs, nc = FLAGS.box_size, FLAGS.nc
    klin = np.loadtxt('../data/Planck15_a1p00.txt').T[0]
    plin = np.loadtxt('../data/Planck15_a1p00.txt').T[1]
    ipklin = iuspline(klin, plin)
    stages = np.linspace(a0, a, nsteps, endpoint=True)

    tf.reset_default_graph()
    # Run normal flowpm to generate data
    ic = np.load('../data/poisson_L%04d_N%03d/ic.npy'%(bs, nc))
    fin = np.load('../data/poisson_L%04d_N%03d/final.npy'%(bs, nc))
    data = np.load('../data/poisson_L%04d_N%03d/psample_%0.2f.npy'%(bs, nc, plambda))


    k, pic = tools.power(ic[0]+1, boxsize=bs)
    k, pfin = tools.power(fin[0], boxsize=bs)
    plt.plot(k, pic)
    plt.plot(k, pfin)
    plt.loglog()
    plt.grid(which='both')
    plt.savefig('pklin.png')
    plt.close()

    print(pic)
    print(pfin)
    #sys.exit(-1)

    ################################################################
    tf.reset_default_graph()
    print('ic constructed')

    #noise = np.random.normal(0, 1, nc**3).reshape(fin.shape).astype(np.float32)*1
    #data_noised = fin + noise
    #data = data_noised

    startpos = np.random.normal(0, 1, nc**3).reshape(fin.shape).astype(np.float32)*1
    startpos = startpos.flatten()

    x0 = tf.placeholder(dtype=tf.float32, shape=data.flatten().shape, name='initlin')
    xlin = tf.placeholder(dtype=tf.float32, shape=data.shape, name='linfield')
    Rsm = tf.placeholder(tf.float32, name='smoothing')



    def recon_prototype(linearflat):
        """
        """       
        linear = tf.reshape(linearflat, data.shape)
        #

        #loss = tf.reduce_sum(tf.square(linear - minimum)) 
        state = lpt_init(linear, a0=0.1, order=1)
        final_state = nbody(state,  stages, FLAGS.nc)
        final_field = cic_paint(tf.zeros_like(linear), final_state[0])
        #final_field = pmgraph(linear)
        base = final_field

        if FLAGS.anneal:
            Rsmsq = tf.multiply(Rsm*bs/nc, Rsm*bs/nc)
            smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
            basek = r2c3d(base, norm=nc**3)
            basek = tf.multiply(basek, tf.cast(smwts, tf.complex64))
            base = c2r3d(basek, norm=nc**3)   

        galmean = tfp.distributions.Poisson(rate = plambda * (1 + base))
        logprob = -tf.reduce_sum(galmean.log_prob(data))
        #logprob = tf.multiply(logprob, 1/nc**3, name='logprob')
        
        #Prior
        lineark = r2c3d(linear, norm=nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
        #prior = tf.multiply(prior, 1/nc**3, name='prior')
        #
        loss = logprob + prior

        grad = tf.gradients(loss, linearflat)
        print(grad)
        return loss, grad[0]


    @tf.function
    def min_lbfgs():
        return tfp.optimizer.lbfgs_minimize(
            #make_val_and_grad_fn(recon_prototype),
            recon_prototype,
            initial_position=x0,
            tolerance=1e-10,
            max_iterations=FLAGS.niter)

    tfinal_field = pmgraph(xlin)

    RRs = [2.0, 1.0, 0.0]
    start0 = time.time()
    with tf.Session() as sess:

        for iR, RR in enumerate(RRs):
            
            start = time.time()
            results = sess.run(min_lbfgs(), {Rsm:RR, x0:startpos})        
            print("\n")
            print(results)
            print("\n")
            startpos = results.position
            print(startpos)
            print("\nTime taken for %d iterations: "%FLAGS.niter, time.time()-start)

            minic = startpos.reshape(data.shape)
            minfin  = sess.run(tfinal_field, {xlin:minic})
            dg.saveimfig("R%d"%RR, [minic, minfin], [ic, fin], fpath+'')
            dg.save2ptfig("R%d"%RR, [minic, minfin], [ic, fin], fpath+'', bs)
            
            np.save(fpath + 'recon-icR%d'%RR, minic)
            np.save(fpath + 'recon-finalR%d'%RR, minfin)

    #tf.reset_default_graph()
    print("\n")
    print('\nminimized\n')
    print('\nTotal time taken %d iterations: '%(len(RRs)*FLAGS.niter), time.time()-start0)

    #tfic = linear_field(FLAGS.nc, FLAGS.box_size, ipklin, batch_size=1, seed=100, dtype=dtype)*0 + minimum.reshape(data_noised.shape)
    #state = lpt_init(tfic, a0=0.1, order=1)
    #final_state = nbody(state,  stages, FLAGS.nc)
    #tfinal_field = cic_paint(tf.zeros_like(tfic), final_state[0])


    
##
    exit(0)

if __name__ == "__main__":
  tf.app.run(main=main)

  
