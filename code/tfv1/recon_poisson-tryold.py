import numpy as np
import numpy
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from time import time
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
#from tensorflow.contrib.opt import ScipyOptimizerInterface
import tensorflow_hub as hub

sys.path.append('../flowpm/')
#from background import *
#import tfpm 
#import tfpmfuncs as tfpf
#from tfpmconfig import Config
from flowpm.tfpm import PerturbationGrowth
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d
from flowpm import tfpm
import flowpm

sys.path.append('../utils/')
import tools
#from standardrecon import standardinit
import diagnostics as dg
#import reconmodels as rmods
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline




#cosmology=Planck15
np.random.seed(100)
tf.random.set_random_seed(200)
cscratch = "../figs_recon/"


tf.flags.DEFINE_integer("nc", 64, "Size of the cube")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size")
tf.flags.DEFINE_float("box_size", 200, "Batch Size")
tf.flags.DEFINE_float("a0", 0.1, "initial scale factor")
tf.flags.DEFINE_float("af", 1.0, "final scale factor")
tf.flags.DEFINE_integer("nsteps", 5, "Number of time steps")
tf.flags.DEFINE_bool("nbody", True, "Do nbody evolution")
tf.flags.DEFINE_string("suffix", "-sm4", "suffix for the folder name")
tf.flags.DEFINE_float("plambda", 0.1, "Multiplicative factor of Poisson lambda")
tf.flags.DEFINE_string("output_file", "timeline", "Name of the output timeline file")

FLAGS = tf.flags.FLAGS

nc, bs = FLAGS.nc, FLAGS.box_size
a0, a, nsteps =FLAGS.a0, FLAGS.af, FLAGS.nsteps
stages = np.linspace(a0, a, nsteps, endpoint=True)
klin = np.loadtxt('../data/Planck15_a1p00.txt').T[0]
plin = np.loadtxt('..//data/Planck15_a1p00.txt').T[1]
ipklin = iuspline(klin, plin)
if FLAGS.nbody: fpath = cscratch + "nbody_N%03d_single_poisson_p%0.2f%s/"%(nc, FLAGS.plambda, FLAGS.suffix) # 
else: fpath = cscratch + "lpt_N%03d_single_poisson_p%0.2f%s/"%(nc, FLAGS.plambda, FLAGS.suffix) # 
print(fpath)
for ff in [fpath, fpath + '/figs']:
    try: os.makedirs(ff)
    except Exception as e: print (e)



def recon_model(data, sigma=0.01**0.5, maxiter=100, anneal=False, dataovd=False, gtol=1e-5):

    #bs, nc = config['boxsize'], config['nc']
    kvec = flowpm.kernels.fftk([nc, nc, nc], symmetric=False)
    kmesh = sum(kk**2 for kk in kvec)**0.5
    priorwt = ipklin(kmesh) * bs ** -3 
    
    g = tf.Graph()

    with g.as_default():
        
        initlin = tf.placeholder(tf.float32, data.shape, name='initlin')
        linear = tf.get_variable('linmesh', shape=(nc, nc, nc), 
                             initializer=tf.random_normal_initializer(), trainable=True)
        initlin_op = linear.assign(initlin, name='initlin_op')
        #PM
        icstate = tfpm.lptinit(linear, FLAGS.a0, name='icstate')
        fnstate = tfpm.nbody(icstate, stages, nc, name='fnstate')
        final = tf.zeros_like(linear)
        final = cic_paint(final, fnstate[0], name='final')
        if dataovd:
            print('\Converting final density to overdensity because data is that\n')
            fmean = tf.reduce_mean(final)
            final = tf.multiply(final, 1/ fmean)
            final = final - 1
        #
        #Prior
        lineark = r2c3d(linear, norm=nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
        prior = tf.multiply(prior, 1/nc**3, name='prior')

        likelihood = tf.subtract(final, data)
        likelihood = tf.multiply(likelihood, 1/sigma)
        #galmean = tfp.distributions.Poisson(rate = plambda * (1 + finalfield))
        #logprob = galmean.log_prob(data)

        ##Anneal
        Rsm = tf.placeholder(tf.float32, name='smoothing')
        if anneal :
            print('\nAdding annealing part to graph\n')
            Rsm = tf.multiply(Rsm, bs/nc)
            Rsmsq = tf.multiply(Rsm, Rsm)
            smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
            likelihood = tf.squeeze(likelihood)
            likelihoodk = r2c3d(likelihood, norm=nc**3)
            likelihoodk = tf.multiply(likelihoodk, tf.cast(smwts, tf.complex64))
            residual = c2r3d(likelihoodk, norm=nc**3)
        else:
            residual = tf.identity(likelihood)
            
        chisq = tf.multiply(residual, residual)
        chisq = tf.reduce_sum(chisq)
        chisq = tf.multiply(chisq, 1/nc**3, name='chisq')

        loss = tf.add(chisq, prior, name='loss')
        
        #optimizer = ScipyOptimizerInterface(loss, var_list=[linear], method='L-BFGS-B', 
        #                                    options={'maxiter': maxiter, 'gtol':gtol})

        optimizer = tf.optimize.AdamWeightDecayOptimizer(0.01)        
        var_grads = tf.gradients(
            [loss], [linear])

        update_ops = optimizer.apply_grads(var_grads, linear)

        
        tf.add_to_collection('inits', [initlin_op, initlin])
        #tf.add_to_collection('opt', optimizer)
        tf.add_to_collection('opt', update_ops)
        tf.add_to_collection('diagnostics', [prior, chisq, loss])
        tf.add_to_collection('reconpm', [linear, final, fnstate])
        tf.add_to_collection('data', data)
    return g




def loss_callback(var, literals, nprint=50, nsave=50, maxiter=500, t0=time()):
    losses = literals['losses']
    loss = var[0]
    reconmeshes = var[1]
    nit = len(losses) %(maxiter*2)
    losses.append(loss)


    if nit % nprint == 0:
        print('Time taken for iterations %d = '%nit, time() - t0)
        print(nit, " - Loss, chisq, prior, grad : ", loss)

        fname = optfolder + '/%d.png'%nit
        stime = time()
        #dg.savehalofig(literals['truemeshes'], reconmeshes[0], fname, literals['hgraph'], boxsize=bs, title='%s'%loss)
        dg.makefig(literals['truemeshes'], reconmeshes, fname, boxsize=bs, title='%s'%loss)    
        print('Time taken to make figure = ', time()-stime)
        
    if nit % nsave == 0:
        np.save(optfolder + '/iter%d.f4'%nit, reconmeshes)
        np.savetxt(optfolder + '/losses.txt', np.array(losses))






########################



if __name__=="__main__":

    #
    maxiter = 500
    gtol = 1e-8
    sigma = 1**0.5
    nprint, nsave = 20, 40
    R0s = [4, 2, 1, 0]
    

    ofolder = fpath

    tf.reset_default_graph()
    # Run normal flowpm to generate data
    plambda = FLAGS.plambda
    ic, fin, data = np.load('../data/poisson_N%03d/ic.npy'%nc), np.load('../data/poisson_N%03d/final.npy'%nc), \
                    np.load('../data/poisson_N%03d/psample_%0.2f.npy'%(nc, plambda))
    #ic = np.expand_dims(ic, 0)
    #fin = np.expand_dims(fin, 0)
    #data = np.expand_dims(data, 0)
    ic = np.squeeze(ic)
    fin = np.squeeze(fin)
    data = np.squeeze(data)
    print('Data loaded')

    
    truemeshes = [ic, fin, data]
    np.save(ofolder + '/truth.f4', ic)
    np.save(ofolder + '/final.f4', fin)
    np.save(ofolder + '/data.f4', data)

    ###
    #Do reconstruction here
    print('\nDo reconstruction\n')

    recong = recon_model(data, sigma=0.01**0.5, maxiter=100, anneal=False, dataovd=False, gtol=1e-5)
    #
    
    initval = None
    #initval = np.random.normal(1, 0.5, size=nc**3).reshape(nc, nc, nc).astype(config['dtype'])#truth
    #initval = standardinit(config, data, hposd, final, R=8)
    #initval = tools.readbigfile(dpath + ftype%(bs, nc, 900, step) + 'mesh/s/')
    #initval = np.ones((nc, nc, nc))
    #initval = truth.copy()


    losses = []
    literals = {'losses':losses, 'truemeshes':truemeshes, 'bs':bs, 'nc':nc}
    tstart = time()
    lcallback = lambda x: loss_callback(x, literals=literals, nprint=nprint, nsave=nsave, maxiter=maxiter, t0=tstart)
    
    with tf.Session(graph=recong) as session:
        g = session.graph
        session.run(tf.global_variables_initializer())
        linmesh = g.get_tensor_by_name("linmesh:0")
        final = g.get_tensor_by_name("final:0")
        samples = tf.squeeze(g.get_tensor_by_name("samples:0"))
        optimizer = g.get_collection_ref('opt')[0]
        loss = g.get_tensor_by_name('loss:0')
        chisq = g.get_tensor_by_name('chisq:0')
        grad = tf.norm(tf.gradients(loss, linmesh))
        prior = g.get_tensor_by_name('prior:0')
        Rsm = g.get_tensor_by_name('smoothing:0')

        if initval is not None:
            print('Do init')
            initlinop = g.get_operation_by_name('initlin_op')
            initlin = g.get_tensor_by_name('initlin:0')
            session.run(initlinop, {initlin:initval})


        def checkiter(mode, optfolder, R0=0):
            print('\nChecking mode = %s\n'%mode)
            meshs, meshf, meshd = session.run([linmesh, final, samples], {Rsm:R0})
            title = session.run([loss, chisq, prior, grad], {Rsm:R0})
            np.save(optfolder + '/%s%d.f4'%(mode, R0), meshs) 
            dg.makefig(literals['truemeshes'], [meshs, meshf, meshd], optfolder+'%s%d.png'%(mode, R0), boxsize=bs, title='%s'%title)

            
        if anneal:

            for R0 in R0s:
                optfolder = ofolder + "/R%02d/"%(R0*10)
                try: os.makedirs(optfolder)
                except:pass
                print('\nAnneal for Rsm = %0.2f\n'%R0)
                print('Output in ofolder = \n%s'%optfolder)

                checkiter('init', optfolder, R0=R0)
                #
                for i in range(20):
                    iiter +=1
                    sess.run(optimizer, {lr:lR, R0:RR})
#                    if (i%titer == 0):
#                        end = time.time()
#                        print('Iter : ', i)
#                        print('Time taken for %d iterations: '%titer, end-start)
#                        start = end
#
#                        ##
#                        ic1, fin1, cc, pp = sess.run([tf_initc, tf_final, tf_chisq, tf_prior], {R0:RR})
#                        print('Chisq and prior are : ', cc, pp)
#
#                        dg.saveimfig(i, [ic1, fin1], [ic, fin], fpath+'/figs-R%02d'%(10*RR))
#                        dg.save2ptfig(i, [ic1, fin1], [ic, fin], fpath+'/figs-R%02d'%(10*RR), bs)
#                dg.saveimfig(i*(iR+1), [ic1, fin1], [ic, fin], fpath+'/figs')
#                dg.save2ptfig(i*(iR+1), [ic1, fin1], [ic, fin], fpath+'/figs', bs)
#
                #optimizer.minimize(session, {Rsm:R0}, loss_callback=lcallback, fetches=[[[loss, chisq, prior, grad], \
                #                                                           [linmesh, final, samples]]])
                #
                checkiter('recon', optfolder, R0=R0)

                
        else:
            optfolder = ofolder
            try: os.makedirs(optfolder)
            except:pass
            print('\nNo annealing\n')
            print('Output in ofolder = \n%s'%optfolder)
            
            checkiter('init', optfolder, R0=0)
            ##
            optimizer.minimize(session, loss_callback=lcallback, fetches=[[[loss, chisq, prior, grad], \
                                                                           [linmesh, final, samples]]])
            checkiter('recon', optfolder, R0=0)
            #


