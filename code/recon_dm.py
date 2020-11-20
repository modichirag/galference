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
##


cosmology=Planck15
np.random.seed(100)
tf.random.set_random_seed(200)
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
tf.flags.DEFINE_integer("nsteps", 5, "Number of time steps")
tf.flags.DEFINE_bool("nbody", True, "Do nbody evolution")
tf.flags.DEFINE_string("suffix", "", "suffix for the folder name")
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


def recon_prototype(data, anneal=FLAGS.anneal, nc=FLAGS.nc, bs=FLAGS.box_size, batch_size=FLAGS.batch_size,
                        a0=FLAGS.a0, a=FLAGS.af, nsteps=FLAGS.nsteps, dtype=tf.float32):
    """
    Prototype of function computing LPT deplacement.

    Returns output tensorflow and mesh tensorflow tensors
    """
    if dtype == tf.float32:
        npdtype = "float32"
        cdtype = tf.complex64
    elif dtype == tf.float64:
        npdtype = "float64"
        cdtype = tf.complex128
    print(dtype, npdtype)
    
    #graph = mtf.Graph()
    #mesh = mtf.Mesh(graph, "my_mesh")

    linear = tf.get_variable('linmesh', shape=(1, nc, nc, nc), dtype=tf.float32,
                             initializer=tf.random_normal_initializer(), trainable=True)
    
    state = lpt_init(linear, a0=0.1, order=1)
    final_state = nbody(state,  stages, FLAGS.nc)
    final_field = cic_paint(tf.zeros_like(linear), final_state[0])

    residual = final_field - data.astype(np.float32)
    base = residual
    ##Anneal
    Rsm = tf.placeholder(tf.float32, name='smoothing')
    if anneal :
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

    #opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)

    # Compute the gradients for a list of variables.
    grads_and_vars = opt.compute_gradients(loss, [linear])
    print("\ngradients : ", grads_and_vars)
    update_ops = opt.apply_gradients(grads_and_vars)
    
    #optimizer = tf.keras.optimizers.Adam(0.01)        
    #var_grads = tf.gradients([loss], [linear])


    #update_ops = optimizer.apply_gradients(var_grads, linear)
    #update_ops = optimizer.apply_gradients(zip(var_grads, [linear]))
    #update_ops = None
    #lr = tf.placeholder(tf.float32, shape=())
    #update_op = mtf.assign(fieldvar, fieldvar - var_grads[0]*lr)

    return linear, final_field, update_ops, loss, chisq, prior, Rsm


##############################################

def main(_):

    dtype=tf.float32

    startw = time.time()

    tf.random.set_random_seed(100)
    np.random.seed(100)

    
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

    tf.reset_default_graph()
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

        
    tf.reset_default_graph()
    print('ic constructed')

    noise = np.random.normal(0, 1, nc**3).reshape(fin.shape)
    data_noised = fin + noise

    linear, final_field, update_ops, loss, chisq, prior, Rsm = recon_prototype(data_noised)

    #initial_conditions = recon_prototype(mesh, fin, nc=FLAGS.nc,  batch_size=FLAGS.batch_size, dtype=dtype)

    # Lower mesh computation



    with tf.Session() as sess:
                    
        #ic_check, fin_check = sess.run([tf_initc, tf_final])
        #sess.run(tf_linear_op, feed_dict={input_field:ic})
        #ic_check, fin_check = sess.run([linear, final_field])
        #dg.saveimfig('-check', [ic_check, fin_check], [ic, fin], fpath)
        #dg.save2ptfig('-check', [ic_check, fin_check], [ic, fin], fpath, bs)

        #sess.run(tf_linear_op, feed_dict={input_field:np.random.normal(size=ic.size).reshape(ic.shape)})
        sess.run(tf.global_variables_initializer())
        ic0, fin0 = sess.run([linear, final_field])
        dg.saveimfig('-init', [ic0, fin0], [ic, fin], fpath)
        start = time.time()


        titer = 20
        niter = FLAGS.niter + 1
        iiter = 0
        
        start0 = time.time()
        RRs = [2, 1, 0.5, 0]
        lrs = np.array([0.1, 0.1, 0.1, 0.1, 0.1])*2
        #lrs = [0.1, 0.05, 0.01, 0.005, 0.001]
        for iR, zlR in enumerate(zip(RRs, lrs)):
            RR, lR = zlR
            for ff in [fpath + '/figs-R%02d'%(10*RR)]:
                try: os.makedirs(ff)
                except Exception as e: print (e)
                
            for i in range(niter):
                iiter +=1
                sess.run(update_ops, {Rsm:RR})
                print(sess.run([loss, chisq, prior], {Rsm:RR}))
                if (i%titer == 0):
                    end = time.time()
                    print('Iter : ', i)
                    print('Time taken for %d iterations: '%titer, end-start)
                    start = end

                    ##
                    #ic1, fin1, cc, pp = sess.run([tf_initc, tf_final, tf_chisq, tf_prior], {R0:RR})
                    #ic1, fin1, cc, pp = sess.run([tf_initc, tf_final, tf_chisq, tf_prior], {R0:RR})
                    ic1, fin1 = sess.run([linear, final_field])
                    #print('Chisq and prior are : ', cc, pp)
                    
                    dg.saveimfig(i, [ic1, fin1], [ic, fin], fpath+'/figs-R%02d'%(10*RR))
                    dg.save2ptfig(i, [ic1, fin1], [ic, fin], fpath+'/figs-R%02d'%(10*RR), bs)
            #dg.saveimfig(i*(iR+1), [ic1, fin1], [ic, fin], fpath+'/figs')
            #dg.save2ptfig(i*(iR+1), [ic1, fin1], [ic, fin], fpath+'/figs', bs)

        ic1, fin1 = sess.run([linear, final_field])
        print('Total time taken for %d iterations is : '%iiter, time.time()-start0)
        
    dg.saveimfig(i, [ic1, fin1], [ic, fin], fpath)
    dg.save2ptfig(i, [ic1, fin1], [ic, fin], fpath, bs)

    np.save(fpath + 'ic_recon', ic1)
    np.save(fpath + 'final_recon', fin1)
    print('Total wallclock time is : ', time.time()-start0)

    
##
    exit(0)

if __name__ == "__main__":
  tf.app.run(main=main)

  
