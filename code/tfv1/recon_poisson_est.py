import numpy as np
import os, sys
import math, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from matplotlib import pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp

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
tf.flags.DEFINE_integer("nsteps", 3, "Number of time steps")
tf.flags.DEFINE_bool("nbody", True, "Do nbody evolution")
tf.flags.DEFINE_string("suffix", "", "suffix for the folder name")

#pyramid flags
tf.flags.DEFINE_integer("dsample", 2, "downsampling factor")
tf.flags.DEFINE_integer("hsize", 32, "halo size")

#mesh flags
tf.flags.DEFINE_integer("nx", 1, "# blocks along x")
tf.flags.DEFINE_integer("ny", 1, "# blocks along y")
#tf.flags.DEFINE_string("mesh_shape", "row:16", "mesh shape")
#tf.flags.DEFINE_string("layout", "nx:b1", "layout rules")
tf.flags.DEFINE_string("output_file", "timeline", "Name of the output timeline file")

FLAGS = tf.flags.FLAGS

nc, bs = FLAGS.nc, FLAGS.box_size
a0, a, nsteps =FLAGS.a0, FLAGS.af, FLAGS.nsteps

klin = np.loadtxt('..//data/Planck15_a1p00.txt').T[0].astype(np.float32)
plin = np.loadtxt('..//data/Planck15_a1p00.txt').T[1].astype(np.float32)
ipklin = iuspline(klin, plin)
# Compute necessary Fourier kernels
kvec = flowpm.kernels.fftk((nc, nc, nc), symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
priorwt = ipklin(kmesh)
stages = np.linspace(a0, a, nsteps, endpoint=True)


fpath = "./tmp/"
#if FLAGS.nbody: fpath = cscratch + "nbody_%d_nx%d_ny%d_mesh%s/"%(nc, FLAGS.nx, FLAGS.ny, FLAGS.suffix)
#else: fpath = cscratch + "lpt_%d_nx%d_ny%d_mesh%s/"%(nc, FLAGS.nx, FLAGS.ny, FLAGS.suffix)
print(fpath)
for ff in [fpath, fpath + '/figs']:
    try: os.makedirs(ff)
    except Exception as e: print (e)


def recon_prototype(data, anneal=True, nc=FLAGS.nc, bs=FLAGS.box_size, batch_size=FLAGS.batch_size,
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
    #def anneal
        Rsmsq = tf.multiply(Rsm*bs/nc, Rsm*bs/nc)
        smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
        basek = r2c3d(base, norm=nc**3)
        basek = tf.multiply(basek, tf.cast(smwts, tf.complex64))
        base = c2r3d(basek, norm=nc**3)
    

    chisq = tf.multiply(base, base)
    chisq = tf.reduce_sum(chisq)
    #chisq = tf.multiply(chisq, 1/nc**3, name='chisq')

    #Prior
    lineark = r2c3d(linear, norm=nc**3)
    priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
    prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
    #prior = tf.multiply(prior, 1/nc**3, name='prior')
    #
    
    loss = chisq + prior

##    #optimizer = tf.optimize.AdamWeightDecayOptimizer(0.01)        
##    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
##
##    # Compute the gradients for a list of variables.
##    grads_and_vars = opt.compute_gradients(loss, [linear])
##    print("\ngradients : ", grads_and_vars)
##    update_ops = opt.apply_gradients(grads_and_vars)
##    
##    #optimizer = tf.keras.optimizers.Adam(0.01)        
##    #var_grads = tf.gradients([loss], [linear])
##
##
##    #update_ops = optimizer.apply_gradients(var_grads, linear)
##    #update_ops = optimizer.apply_gradients(zip(var_grads, [linear]))
##    #update_ops = None
##    #lr = tf.placeholder(tf.float32, shape=())
##    #update_op = mtf.assign(fieldvar, fieldvar - var_grads[0]*lr)
##
    return linear, final_field, loss, chisq, prior



############################################





def model_fn(features, labels, mode, params):
    """The model_fn argument for creating an Estimator."""

    #tf.logging.info("features = %s labels = %s mode = %s params=%s" %
    #              (features, labels, mode, params))

    global_step = tf.train.get_global_step()
    graph = tf.Graph()

    data = features['data']
    R0 = features['R0']*1.
    x0 = features['x0']
    print("\nR0 in the model function : %0.1f\n"%R0)
    fieldvar, final_field, loss, chisq, prior = recon_model(mesh, data, R0, x0)
    

    ##
    if mode == tf.estimator.ModeKeys.TRAIN:
        var_grads = tf.gradients(
            [loss], [v.outputs[0] for v in graph.trainable_variables])

#        nyq = np.pi*nc/bs
#        def _cwise_highpass(kfield, kx, ky, kz):
#            kx = tf.reshape(kx, [-1, 1, 1])
#            ky = tf.reshape(ky, [1, -1, 1])
#            kz = tf.reshape(kz, [1, 1, -1])
#            kk = (kx / bs * nc)**2 + (ky/ bs * nc)**2 + (kz/ bs * nc)**2
#            wts = tf.cast(tf.exp(- kk* (R0*bs/nc + 1/nyq)**2), kfield.dtype)
#            return kfield * (1-wts)
#        
#        k_dims_pr = [d.shape[0] for d in kv]
#        k_dims_pr = [k_dims_pr[2], k_dims_pr[0], k_dims_pr[1]]
#        cgrads = mesh_utils.r2c3d(var_grads[0], k_dims_pr, dtype=tf.complex64)
#        cgrads = mtf.cwise(_cwise_highpass, [cgrads] + kv, output_dtype=tf.complex64)
#        var_grads = [mesh_utils.c2r3d(cgrads, var_grads[0].shape[-3:], dtype=tf.float32)]
#        update_ops = [mtf.assign(fieldvar, fieldvar - var_grads[0]*0.2)]

        optimizer = tf.optimize.AdamOptimizer(features['lr'])
        update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

#

    start = time.time()
    #lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    #print("\nTime taken for lowering is : %0.3f"%(time.time()-start))
    #restore_hook = mtf.MtfRestoreHook(lowering)
    
    ##Predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.summary.scalar("loss", toss)
        tf.summary.scalar("chisq", chisq)
        tf.summary.scalar("prior", prior)
        predictions = {
            "ic": fieldvar,
            "data": data,
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            #prediction_hooks=[restore_hook],
            export_outputs={
                "data": tf.estimator.export.PredictOutput(predictions) #TODO: is classify a keyword?
            })

    ##Train
    if mode == tf.estimator.ModeKeys.TRAIN:
        #tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
        update_ops.append(tf.assign_add(global_step, 1))
        train_op = tf.group(update_ops)
        saver = tf.train.Saver(
            tf.global_variables(),
            sharded=True,
            max_to_keep=1,
            keep_checkpoint_every_n_hours=2,
            defer_build=False, save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        saver_listener = mtf.MtfCheckpointSaverListener(lowering)
        saver_hook = tf.train.CheckpointSaverHook(
            fpath,
            save_steps=1000,
            saver=saver,
            listeners=[saver_listener])
        
        logging_hook = tf.train.LoggingTensorHook({"loss" : loss, 
                                                   "chisq" : chisq,
                                                   "prior" : prior}, every_n_iter=10)

        # Name tensors to be logged with LoggingTensorHook.
        tf.identity(loss, "loss")
        tf.identity(prior, "prior")
        tf.identity(chisq, "chisq")
        
        # Save accuracy scalar to Tensorboard output.
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("chisq", chisq)
        tf.summary.scalar("prior", prior)

        # restore_hook must come before saver_hook
        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op,
            training_chief_hooks=[restore_hook, saver_hook, logging_hook])

    ##Eval
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=tf_loss,
            evaluation_hooks=[restore_hook],
            eval_metric_ops={
                "loss": tf_loss,
                "chisq" : tf_chisq,
                #tf.metrics.accuracy(
                #    labels=labels, predictions=tf.argmax(tf_logits, axis=1)),
            })
    


#############################################
#

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

    linear, final_field, update_ops, loss, chisq, prior, Rsm = recon_prototype(fin)

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
        niter = 201
        iiter = 0
        
        start0 = time.time()
        RRs = [4, 2, 1, 0.5, 0]
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
            dg.saveimfig(i*(iR+1), [ic1, fin1], [ic, fin], fpath+'/figs')
            dg.save2ptfig(i*(iR+1), [ic1, fin1], [ic, fin], fpath+'/figs', bs)

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

  
