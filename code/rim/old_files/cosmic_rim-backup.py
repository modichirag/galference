""" Implementation of Cosmic RIM estimator"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, sys, argparse, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from matplotlib import pyplot as plt

from rim_utils import build_rim
import flowpm
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d
sys.path.append('../../utils/')
import tools



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--nc', type=int, default=16, help='Grid size')
parser.add_argument('--bs', type=float, default=100, help='Box Size')
parser.add_argument('--nsteps', type=int, default=3, help='')
parser.add_argument('--niter', type=int, default=200, help='Number of iterations/Max iterations')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='adam', help='Which optimizer to use')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')

args = parser.parse_args()


nc, bs = args.nc, args.bs
niter = args.niter
optimizer = args.optimizer
lr = args.lr
a0, a, nsteps = 0.1, 1.0, 3
stages = np.linspace(a0, a, nsteps, endpoint=True)
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
params['input_size'] = 8
params['cell_size'] = 8
params['cell_kernel_size'] = 5
params['input_kernel_size'] = 5
params['output_kernel_size'] = 5
params['rim_iter'] = 10
params['nc'] = nc


def get_data(nsims=100):
    dpath = '../../data/rim-data/L%04d_N%03d_T%02d/'%(bs, nc, nsteps)
    alldata = np.array([np.load(dpath + '%04d.npy'%i) for i in range(nsims)]).astype(np.float32)
    traindata, testdata = alldata[:int(0.9*nsims)], alldata[int(0.9*nsims):]
    return traindata, testdata



@tf.function
def pm(linear):
    state = lpt_init(linear, a0=a0, order=1)
    final_state = nbody(state, stages, nc)
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
def recon_dm_grad(x, y):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = recon_dm(x, y)[0]
    grad = tape.gradient(loss, x)
    return grad




def get_ps(iterand, truth):
    ic, fin = truth
    ic1, fin1 = iterand

    pks = []
    if abs(ic1[0].mean()) < 1e-3: ic1[0] += 1
    #if abs(ic[0].mean()) < 1e-3: ic[0] += 1                                                                                                                  
    k, p1 = tools.power(ic1[0]+1, boxsize=bs)
    k, p2 = tools.power(ic[0]+1, boxsize=bs)
    k, p12 = tools.power(ic1[0]+1, f2=ic[0]+1, boxsize=bs)
    pks.append([p1, p2, p12])
    if fin1[0].mean() < 1e-3: fin1[0] += 1
    if fin[0].mean() < 1e-3: fin[0] += 1
    k, p1 = tools.power(fin1[0], boxsize=bs)
    k, p2 = tools.power(fin[0], boxsize=bs)
    k, p12 = tools.power(fin1[0], f2=fin[0], boxsize=bs)
    pks.append([p1, p2, p12])

    return k, pks



def test_callback(pred, x_init, xx, yy, suff='', pref=''):
            
    fig, ax = plt.subplots(1, 3, figsize = (12, 4))
    vmin, vmax = xx[0].sum(axis=0).min(), xx[0].sum(axis=0).max()
    ax[0].imshow(xx[0].sum(axis=0), vmin=vmin, vmax=vmax)
    ax[0].set_title('Truth')
    ax[1].imshow(x_init[0].sum(axis=0), vmin=vmin, vmax=vmax)
    ax[1].set_title('initial point')
    ax[2].imshow(pred[0].sum(axis=0), vmin=vmin, vmax=vmax)
    ax[2].set_title('RIM %d step'%(params['rim_iter']))
    plt.savefig(pref + 'rim-im' + suff + '.png')
    plt.close()
    
    ##
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    k, pks = get_ps([x_init, pm(x_init).numpy()], [xx, yy])
    for i in range(2):
        ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d--'%i)
        ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d--'%i)

    k, pks = get_ps([pred, pm(pred).numpy()], [xx, yy])
    for i in range(2):
        ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d'%i)
        ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d'%i)

    for axis in ax: 
        axis.semilogx()
        axis.grid(which='both')
    plt.savefig(pref + 'rim-2pt' + suff + '.png')
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
    piter, testiter  = 10, 50
    losses = []
    lrs = [0.001, 0.0005, 0.0001]
    liters = [201, 1001, 1001]
    trainiter = 0 
    start = time.time()
    counter = 0 
    for il in range(3):
        print('Learning rate = %0.3e'%lrs[il])
        opt = tf.keras.optimizers.Adam(learning_rate=lrs[il])

        for i in range(liters[il]):
            counter +=1
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
                #
                idx = np.random.randint(0, testdata.shape[0], 1)
                xx, yy = testdata[idx, 0].astype(np.float32), testdata[idx, 1].astype(np.float32), 
                x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
                pred = rim(tf.constant(x_init), tf.constant(yy), grad_fn)[0].numpy()[-1]
                test_callback(pred, x_init, xx, yy, suff='-%d'%counter)
                #
                savepath = './rim-models/L%04d_N%03d_T%02d-c8/iter-%04d'%(bs, nc, nsteps, trainiter)
                rim.save_weights(savepath, overwrite=False)
            trainiter  += 1


if __name__=="__main__":
    main()
