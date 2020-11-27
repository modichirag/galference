import numpy as np
import os, sys
import math, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from matplotlib import pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import tensorflow_addons as tfa
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
tf.flags.DEFINE_integer("nsteps", 5, "Number of time steps")
tf.flags.DEFINE_bool("nbody", True, "Do nbody evolution")
tf.flags.DEFINE_string("suffix", "", "suffix for the folder name")
tf.flags.DEFINE_bool("anneal", False, "Anneal")
tf.flags.DEFINE_float("lr", 0.01, "Learning rate")
tf.flags.DEFINE_integer("niter", 200, "Number of iterations")
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


    ################################################################
    tf.reset_default_graph()
    print('ic constructed')

    startpos = np.random.normal(0, 1, nc**3).reshape(fin.shape).astype(np.float32)*1
    startpos = startpos.flatten()


    ##
    Rsm = tf.placeholder(tf.float32, name='smoothing')
    def recon_prototype(x0=None):
        """
        """       
#        linear = tf.get_variable('linmesh', shape=(1, nc, nc, nc), dtype=tf.float32,
#                             initializer=tf.random_normal_initializer(), trainable=True)
        if x0 is None:
            linear = tf.get_variable('linmesh', shape=(1, nc, nc, nc), dtype=tf.float32,
                             initializer=tf.random_normal_initializer(), trainable=True)
        else:
            linear = tf.get_variable('linmesh', shape=(1, nc, nc, nc), dtype=tf.float32,
                             initializer=tf.constant_initializer(x0), trainable=True)

        state = lpt_init(linear, a0=0.1, order=1)
        final_state = nbody(state,  stages, FLAGS.nc)
        final_field = cic_paint(tf.zeros_like(linear), final_state[0])
        base = final_field

        if FLAGS.anneal:
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
        #prior = tf.multiply(prior, 1/nc**3, name='prior')
        #
        loss = logprob + prior

        #opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)

        #step = tf.Variable(0, trainable=False)
        #schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        #    [10000, 15000], [1e-0, 1e-1, 1e-2])
        ## lr and wd can be a function or a tensor
        #lr = 1e-1 * schedule(step)
        #wd = lambda: 1e-4 * schedule(step)
        #opt = tfa.optimizers.AdamW(learning_rate=FLAGS.lr, weight_decay=1e-1)

        # Compute the gradients for a list of variables.
        grads_and_vars = opt.compute_gradients(loss, [linear])
        print("\ngradients : ", grads_and_vars)
        update_ops = opt.apply_gradients(grads_and_vars)

        return linear, sample, update_ops, loss, logprob, prior
 

    linear, sample, update_ops, loss, chisq, prior = recon_prototype()


    with tf.Session() as sess:
                    
        #sess.run(tf_linear_op, feed_dict={input_field:np.random.normal(size=ic.size).reshape(ic.shape)})
        sess.run(tf.global_variables_initializer())
        ic0, sampl0 = sess.run([linear, sample], {Rsm:0.})
        dg.saveimfig('-init', [ic0, sampl0], [ic, data], fpath)
        start = time.time()


        titer = 10
        niter = FLAGS.niter + 1
        iiter = 0
        
        start0 = time.time()
        RRs = [2, 1, 0.5, 0]
        lrs = np.array([0.1, 0.1, 0.1, 0.1, 0.1])*2
        for iR, zlR in enumerate(zip(RRs, lrs)):
            RR, lR = zlR
            for ff in [fpath + '/figs-R%02d'%(10*RR)]:
                try: os.makedirs(ff)
                except Exception as e: print (e)
                
            for i in range(niter):
                iiter +=1
                sess.run(update_ops, {Rsm:RR})
                #
                if (i%titer == 0):
                    end = time.time()
                    print('Iter : ', i)
                    print('Time taken for %d iterations: '%titer, end-start)
                    start = end
                    print(sess.run([loss, chisq, prior], {Rsm:RR}))
                if (i%(2*titer) == 0):
                    ic1, samp1 = sess.run([linear, sample], {Rsm:RR})                
                    dg.saveimfig(i, [ic1, samp1], [ic, data], fpath+'/figs-R%02d'%(10*RR))
                    dg.save2ptfig(i, [ic1, samp1], [ic, data], fpath+'/figs-R%02d'%(10*RR), bs)

        ic1, samp1 = sess.run([linear, sample], {Rsm:0.})
        print('Total time taken for %d iterations is : '%iiter, time.time()-start0)
        
    dg.saveimfig(i, [ic1, samp1], [ic, data], fpath)
    dg.save2ptfig(i, [ic1, samp1], [ic, data], fpath, bs)

    np.save(fpath + 'ic_recon', ic1)
    np.save(fpath + 'final_recon', samp1)
    print('Total wallclock time is : ', time.time()-start0)


    
##
    exit(0)

if __name__ == "__main__":
  tf.app.run(main=main)

  
