""" Implementation of Cosmic RIM estimator"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, sys, argparse, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from rim_utils import build_rim_split, build_rim_parallel, myAdam
from recon_models import Recon_DM

import flowpm
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d
sys.path.append('../../utils/')
import tools



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
parser.add_argument('--nc', type=int, default=16, help='Grid size')
parser.add_argument('--bs', type=float, default=100, help='Box Size')
parser.add_argument('--nsteps', type=int, default=3, help='')
parser.add_argument('--niter', type=int, default=200, help='Number of iterations/Max iterations')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='adam', help='Which optimizer to use')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--nsims', type=int, default=100, help='Number of simulations')
parser.add_argument('--nbody', type=str2bool, default=True, help='Number of simulationss')
parser.add_argument('--lpt_order', type=int, default=1, help='Order of LPT Initial conditions')
parser.add_argument('--input_size', type=int, default=16, help='Input layer channel size')
parser.add_argument('--cell_size', type=int, default=16, help='Cell channel size')
parser.add_argument('--rim_iter', type=int, default=10, help='Optimization iteration')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--suffix', type=str, default='', help='Suffix for folder pathname')
parser.add_argument('--batch_in_epoch', type=int, default=20, help='Number of batches in epochs')
parser.add_argument('--trainiter', type=int, default=102, help='Which iteration number to plot')
parser.add_argument('--parallel', type=str2bool, default=True, help='Mode')



args = parser.parse_args()


nc, bs = args.nc, args.bs
niter = args.niter
optimizer = args.optimizer
lr = args.lr
a0, af, nsteps = 0.1, 1.0,  args.nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
#anneal = True
#RRs = [2, 1, 0.5, 0]

#
klin = np.loadtxt('../../data/Planck15_a1p00.txt').T[0]
plin = np.loadtxt('../../data//Planck15_a1p00.txt').T[1]
ipklin = iuspline(klin, plin)
# Compute necessary Fourier kernels                                                                                                                          
kvec = tools.fftk((nc, nc, nc), boxsize=nc, symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
priorwt = ipklin(kmesh)



#RIM params
params = {}
params['input_size'] = args.input_size
params['cell_size'] = args.cell_size
params['strides'] = 2
params['middle_size'] = args.input_size // params['strides']  #lets divide by strides
params['cell_kernel_size'] = 5
params['input_kernel_size'] = 5
params['middle_kernel_size'] = 5
params['output_kernel_size'] = 5
params['rim_iter'] = args.rim_iter
params['input_activation'] = 'tanh'
params['output_activation'] = 'linear'
params['nc'] = nc


adam = myAdam(params['rim_iter'])
adam10 = myAdam(10*params['rim_iter'])
fid_recon = Recon_DM(nc, bs, a0=a0, af=af, nsteps=nsteps, nbody=args.nbody, lpt_order=args.lpt_order, anneal=True)

suffpath = '' + args.suffix
if args.nbody: ofolder = './models/L%04d_N%03d_T%02d%s/'%(bs, nc, nsteps, suffpath)
else: ofolder = './models/L%04d_N%03d_LPT%d%s/'%(bs, nc, args.lpt_order, suffpath)
print(ofolder)



def get_data(nsims=args.nsims):
    #if args.nbody: dpath = '/project/projectdirs/m3058/chmodi/rim-data/L%04d_N%03d_T%02d/'%(bs, nc, nsteps)
    #else: dpath = '/project/projectdirs/m3058/chmodi/rim-data/L%04d_N%03d_LPT%d/'%(bs, nc, args.lpt_order)
    if args.nbody: dpath = '../../data/rim-data/L%04d_N%03d_T%02d/'%(bs, nc, nsteps)
    else: dpath = '../../data/rim-data/L%04d_N%03d_LPT%d/'%(bs, nc, args.lpt_order)
    alldata = np.array([np.load(dpath + '%04d.npy'%i) for i in range(nsims)]).astype(np.float32)
    traindata, testdata = alldata[:int(0.9*nsims)], alldata[int(0.9*nsims):]
    return traindata, testdata

@tf.function()
def pm_data(dummy):
    print("PM graph")
    linear = flowpm.linear_field(nc, bs, ipklin, batch_size=args.batch_size)
    if args.nbody:
        print('Nobdy sim')
        state = lpt_init(linear, a0=a0, order=args.lpt_order)
        final_state = nbody(state,  stages, nc)
    else:
        print('ZA/2LPT sim')
        final_state = lpt_init(linear, a0=af, order=args.lpt_order)
    tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
    return linear, tfinal_field

@tf.function()
def pm_data_test(dummy):
    print("PM graph")
    linear = flowpm.linear_field(nc, bs, ipklin, batch_size=world_size)
    if args.nbody:
        print('Nobdy sim')
        state = lpt_init(linear, a0=a0, order=args.lpt_order)
        final_state = nbody(state,  stages, nc)
    else:
        print('ZA/2LPT sim')
        final_state = lpt_init(linear, a0=af, order=args.lpt_order)
    tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
    return linear, tfinal_field


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



@tf.function
def recon_dm(linear, data):

    print('new graph')
    final_field = pm(linear)
    residual = final_field - data #.astype(np.float32)
    chisq = tf.multiply(residual, residual)
    chisq = tf.reduce_mean(chisq)                             
    lineark = r2c3d(linear, norm=nc**3)
    priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
    prior = tf.reduce_mean(tf.multiply(priormesh, 1/priorwt))
    loss = chisq + prior

    return loss, chisq, prior


@tf.function
def recon_dm_grad(x, y):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = recon_dm(x, y)[0]
    grad = tape.gradient(loss, x)
    return grad


@tf.function
def gal_sample(base):
    galmean = tfp.distributions.Poisson(rate = plambda * (1 + base))
    return galmean.sample()


@tf.function
def recon_poisson(linear, data):
    """                                                                                                                                                   
    """
    print('new graph')
    base = pm(linear)

    galmean = tfp.distributions.Poisson(rate = plambda * (1 + base))
    logprob = -tf.reduce_mean(galmean.log_prob(data))
    #logprob = tf.multiply(logprob, 1/nc**3, name='logprob')

    #Prior
    lineark = r2c3d(linear, norm=nc**3)
    priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
    prior = tf.reduce_mean(tf.multiply(priormesh, 1/priorwt))
#     prior = tf.multiply(prior, 1/nc**3, name='prior')
    #                                                                                                                                                     
    loss = logprob + prior

    return loss, logprob, prior



@tf.function
def recon_poisson_grad(x, y):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = recon_poisson(x, y)[0]
    grad = tape.gradient(loss, x)
    return grad



def check_im(xx, x_init, pred, fname=None):
    fig, ax = plt.subplots(1, 3, figsize = (12, 4))
    vmin, vmax = xx.sum(axis=0).min(), xx.sum(axis=0).max()
    ax[0].imshow(xx.sum(axis=0), vmin=vmin, vmax=vmax)
    ax[0].set_title('Truth')
    ax[1].imshow(x_init.sum(axis=0), vmin=vmin, vmax=vmax)
    ax[1].set_title('initial point')
    ax[2].imshow(pred.sum(axis=0), vmin=vmin, vmax=vmax)
    ax[2].set_title('RIM recon')
    if fname is not None: plt.savefig('./figs/' + fname)
    else: plt.savefig('./figs/rim-im.png')
    plt.close()



def check_2pt(xx, yy, rim, grad_fn, compares, nrim=10, fname=None):
    truemesh = [xx[0], yy[0]]
    rimpreds = []
    for it in range(nrim):
        x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
        #x_init = (yy - (yy.max() - yy.min())/2.)/yy.std() + np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
        pred = rim(tf.constant(x_init), tf.constant(yy), grad_fn)[-1]
        rimpreds.append([pred[0].numpy(), pm(pred)[0].numpy()])

    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
    for ip, preds in enumerate(rimpreds):
        k, pks = tools.get_ps(preds, truemesh, bs)
        for i in range(2):
            lbl = None
            if ip == 0 and i == 0: lbl = 'Linear'
            if ip == 0 and i == 1: lbl = 'Final'
            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d-'%i, alpha=0.4, label=lbl)
            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d-'%i, alpha=0.4)

    lss = ['-', '--', ':', '-.']
    lws = [ 1, 1, 2, 2]
    lbls = ['Adam', 'Adam 10x', 'Best recon']
    #for ip, preds in enumerate([pred_adam, pred_adam10]):
    for ip, preds in enumerate(compares):
        k, pks = tools.get_ps(preds, truemesh, bs)
        for i in range(2):
            lbl = None
            if i == 0: lbl = lbls[ip]
            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d'%i, ls=lss[ip+1], lw=lws[ip+1])
            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d'%i,  label=lbl, ls=lss[ip+1], lw=lws[ip+1])
        
    for axis in ax: 
        axis.semilogx()
        axis.grid(which='both')
        axis.legend(fontsize=12)
        axis.set_xlabel('k(h/Mpc)', fontsize=12)
    ax[0].set_ylim(-0.1, 1.2)
    ax[1].set_ylim(-0.5, 2.0)
    ax[0].set_ylabel('$r_c$', fontsize=12)
    ax[1].set_ylabel('$t_f$', fontsize=12)
    plt.tight_layout()
    if fname is not None: plt.savefig('./figs/' + fname)
    else: plt.savefig('./figs/rim-2pt.png')
    plt.close()






def main():
    """
    Model function for the CosmicRIM.
    """

    if args.parallel: rim = build_rim_parallel(params)
    else: rim = build_rim_split(params)
    grad_fn = recon_dm_grad
    #
#
    traindata, testdata = get_data()
    idx = np.random.randint(0, traindata.shape[0], 1)
    xx, yy = traindata[idx, 0].astype(np.float32), traindata[idx, 1].astype(np.float32), 
    x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
    x_pred = rim(x_init, yy, grad_fn)

    trainiter = args.trainiter
    rim.load_weights(ofolder + '%d'%trainiter)
    print('Loaded')
    idx = np.random.randint(0, testdata.shape[0], 1)
    x_test, y_test = testdata[idx, 0].astype(np.float32), testdata[idx, 1].astype(np.float32), 
    x_init = np.random.normal(size=x_test.size).reshape(x_test.shape).astype(np.float32)
    pred = rim(tf.constant(x_init), tf.constant(y_test), grad_fn)



    fig, ax = plt.subplots(1, 2, figsize = (9, 4))
    k, pkt = tools.power(x_test[0], boxsize= bs)
    lss = ["-"]*7 + ["--"]*7
    print(lss)
    for i in range(pred.shape[0]):
        print(i, pred[i].shape, x_test.shape)
        k, pk = tools.power(pred[i, 0].numpy(), boxsize= bs)
        k, px = tools.power(pred[i, 0].numpy(), f2=x_test[0], boxsize= bs)
        rcc =  px/(pkt*pk)**0.5
        print(rcc)
        ax[0].plot(k, rcc, 'C%d'%(i%7), alpha=0.7, ls=lss[(i%7)], label=i)
        ax[1].plot(k, (pk/pkt)**0.5, 'C%d'%(i%7), alpha=0.7, ls=lss[(i%7)])
    for axis in ax:
        axis.semilogx()
        axis.legend()
        axis.grid(which='both')
    ax[0].set_ylim(-0.1, 1.2)
    ax[1].set_ylim(-0.2, 2.5)
    plt.savefig('./figs/2pt-iters.png')
    plt.close()

    fig, ax = plt.subplots(2, 5, figsize = (14, 8))
    for i in range(10):
        ax.flatten()[i].imshow(pred[i+1, 0].numpy().sum(axis=0))
    plt.savefig('./figs/im-iters.png')
    plt.close()


    lss = ['-', '--', ':', '-.']
    
    pred_adam = adam(tf.constant(x_init), tf.constant(y_test), grad_fn)
    pred_adam = [pred_adam[0].numpy(), pm(pred_adam)[0].numpy()]
    pred_adam10 = adam10(tf.constant(x_init), tf.constant(y_test), grad_fn)
    pred_adam10 = [pred_adam10[0].numpy(), pm(pred_adam10)[0].numpy()]
    minic, minfin = fid_recon.reconstruct(tf.constant(y_test), RRs=[1.0, 0.0], niter=args.rim_iter*10, lr=0.1)
    compares =  [pred_adam, pred_adam10, [minic[0], minfin[0]]]
    print('Test set generated')
       

    check_im(x_test[0], x_init[0], pred.numpy()[0], fname= 'rim-im')
    check_2pt(x_test, y_test, rim, grad_fn, compares, fname= 'rim-2pt')

    x_init = pred.numpy().copy()    
    pred = rim(tf.constant(pred), tf.constant(y_test), grad_fn)[-1]
    check_im(x_test[0], x_init[0], pred.numpy()[0], fname= 'rim-im-pred')
    check_2pt(x_test, y_test, rim, grad_fn, compares, fname= 'rim-2pt-pred')

    x_init = y_test
    pred = rim(tf.constant(pred), tf.constant(y_test), grad_fn)[-1]
    check_im(x_test[0], x_init[0], pred.numpy()[0], fname= 'rim-im-data')
    check_2pt(x_test, y_test, rim, grad_fn, compares, fname= 'rim-2pt-data')

    x_init = x_test
    pred = rim(tf.constant(pred), tf.constant(y_test), grad_fn)[-1]
    check_im(x_test[0], x_init[0], pred.numpy()[0], fname= 'rim-im-truth')
    check_2pt(x_test, y_test, rim, grad_fn, compares, fname= 'rim-2pt-truth')

if __name__=="__main__":
    main()
