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

from rim_utils import build_rim, myAdam
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
parser.add_argument('--input_size', type=int, default=8, help='Input layer channel size')
parser.add_argument('--cell_size', type=int, default=8, help='Cell channel size')
parser.add_argument('--rim_iter', type=int, default=10, help='Optimization iteration')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')



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


#RIM Params
params = {}
params['input_size'] = args.input_size
params['cell_size'] = args.cell_size
params['cell_kernel_size'] = 5
params['input_kernel_size'] = 5
params['output_kernel_size'] = 5
params['rim_iter'] = args.rim_iter
params['nc'] = nc


adam = myAdam(params['rim_iter'])
adam10 = myAdam(10*params['rim_iter'])


def get_data(nsims=args.nsims):
    #if args.nbody: dpath = '/project/projectdirs/m3058/chmodi/rim-data/L%04d_N%03d_T%02d/'%(bs, nc, nsteps)
    #else: dpath = '/project/projectdirs/m3058/chmodi/rim-data/L%04d_N%03d_LPT%d/'%(bs, nc, args.lpt_order)
    if args.nbody: dpath = '../../data/rim-data/L%04d_N%03d_T%02d/'%(bs, nc, nsteps)
    else: dpath = '../../data/rim-data/L%04d_N%03d_LPT%d/'%(bs, nc, args.lpt_order)
    alldata = np.array([np.load(dpath + '%04d.npy'%i) for i in range(nsims)]).astype(np.float32)
    traindata, testdata = alldata[:int(0.9*nsims)], alldata[int(0.9*nsims):]
    return traindata, testdata



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
    """                                                                                                                                                   
    """
    print('new graph')
    final_field = pm(linear)

    residual = final_field - data #.astype(np.float32)

    chisq = tf.multiply(residual, residual)
    chisq = tf.reduce_mean(chisq)
#     chisq = tf.multiply(chisq, 1/nc**3, name='chisq')

    #Prior                                                                                                                                                
    lineark = r2c3d(linear, norm=nc**3)
    priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
    prior = tf.reduce_mean(tf.multiply(priormesh, 1/priorwt))
#     prior = tf.multiply(prior, 1/nc**3, name='prior')
    #                                                                                                                                                     
    loss = chisq + prior

    return loss, chisq, prior


@tf.function
def recon_dm_grad_true(x, y):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = recon_dm(x, y)[0]
    grad = tape.gradient(loss, x)
    return grad

@tf.function
def recon_dm_grad(x, y):
    return y
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = recon_dm(x, y)[0]
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
    if fname is not None: plt.savefig(fname)
    else: plt.savefig('rim-im.png')
    plt.close()



def check_2pt(xx, yy, rim, grad_rim, grad_adam, nrim=10, fname=None):
    truemesh = [xx[0], yy[0]]
    rimpreds = []
    for it in range(nrim):
        x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
        pred = rim(tf.constant(x_init), tf.constant(yy), grad_rim)[0][-1]
        if it == 0: check_im(xx[0], x_init[0], pred.numpy()[0])
        rimpreds.append([pred[0].numpy(), pm(pred)[0].numpy()])
    pred_adam = adam(tf.constant(x_init), yy, grad_adam)
    pred_adam = [pred_adam[0].numpy(), pm(pred_adam)[0].numpy()]
    pred_adam10 = adam10(tf.constant(x_init), yy, grad_adam)
    pred_adam10 = [pred_adam10[0].numpy(), pm(pred_adam10)[0].numpy()]

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
    lbls = ['Adam', 'Adam 10x']
    for ip, preds in enumerate([pred_adam, pred_adam10]):
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
    if fname is not None: plt.savefig(fname)
    else: plt.savefig('rim-2pt.png')
    plt.close()




def main():
    """
    Model function for the CosmicRIM.
    """

    rim = build_rim(params)
    grad_fn = recon_dm_grad
    #
    traindata, testdata = get_data()

    #
    # @tf.function
    def rim_train(x_true, x_init, y):

        with tf.GradientTape() as tape:
            x_pred, states = rim(x_init, y, grad_fn)
            res  = (x_true - x_pred)
            loss = tf.reduce_mean(tf.square(res))
        gradients = tape.gradient(loss, rim.trainable_variables)
        return loss, gradients


    ##Train and save
    piter, testiter  = 10, 25
    losses = []
    lrs = [0.001, 0.0005, 0.0001]
    liters = [201, 101, 1001]
    trainiter = 0 
    start = time.time()


    for il in range(1):
        print('Learning rate = %0.3e'%lrs[il])
        opt = tf.keras.optimizers.Adam(learning_rate=lrs[il])

        for i in range(liters[il]):
            idx = np.random.randint(0, traindata.shape[0], args.batch_size)
            xx, yy = traindata[idx, 0].astype(np.float32), traindata[idx, 1].astype(np.float32), 
            x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)

            loss, gradients = rim_train(x_true=tf.constant(xx), 
                                    x_init=tf.constant(x_init), 
                                    y=tf.constant(yy))

            losses.append(loss.numpy())    
            opt.apply_gradients(zip(gradients, rim.trainable_variables))

            if i%piter == 0: 
                print("Time taken for %d iterations : "%piter, time.time() - start)
                print("Loss at iteration %d : "%i, losses[-1])
                start = time.time()
            if i%testiter == 0: 
                plt.plot(losses)
                plt.savefig('losses.png')
                plt.close()

                #check 2pt and comapre to Adam
                idx = np.random.randint(0, testdata.shape[0], 1)
                xx, yy = testdata[idx, 0].astype(np.float32), testdata[idx, 1].astype(np.float32), 
                check_2pt(xx, yy, rim, grad_fn, grad_adam=recon_dm_grad_true, fname='norim-2pt.png'))
                

    #             savepath = './rim-models/L%04d_N%03d_T%02d-c8/iter-%04d'%(bs, nc, nsteps, trainiter)
    #             testrim.save_weights(savepath, overwrite=False)
            trainiter  += 1


if __name__=="__main__":
    main()
