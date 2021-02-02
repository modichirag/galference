""" Implementation of Cosmic RIM estimator"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


import numpy as np
import os, sys, argparse, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from rim_utils import build_rim_parallel, myAdam
from recon_models import Recon_DM, Recon_Bias

import flowpm
from flowpm import linear_field, lpt_init, nbody, cic_paint, cic_readout
from flowpm.utils import r2c3d, c2r3d
sys.path.append('../../utils/')
import tools
from getbiasparams import getbias
import diagnostics as dg



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
parser.add_argument('--nc', type=int, default=32, help='Grid size')
parser.add_argument('--ncf', type=int, default=4, help='Grid size')
parser.add_argument('--bs', type=float, default=200, help='Box Size')
parser.add_argument('--numd', type=float, default=0.001, help='number density')
parser.add_argument('--nsteps', type=int, default=3, help='')
parser.add_argument('--niter', type=int, default=200, help='Number of iterations/Max iterations')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='adam', help='Which optimizer to use')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--nsims', type=int, default=100, help='Number of simulations')
parser.add_argument('--nbody', type=str2bool, default=True, help='Number of simulationss')
parser.add_argument('--lpt_order', type=int, default=2, help='Order of LPT Initial conditions')
parser.add_argument('--input_size', type=int, default=8, help='Input layer channel size')
parser.add_argument('--cell_size', type=int, default=8, help='Cell channel size')
parser.add_argument('--rim_iter', type=int, default=20, help='Optimization iteration')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--suffix', type=str, default='', help='Suffix for folder pathname')
parser.add_argument('--batch_in_epoch', type=int, default=20, help='Number of batches in epochs')
parser.add_argument('--posdata', type=str2bool, default=True, help='Position data')
parser.add_argument('--prior', type=str2bool, default=True, help='Use prior as sum')
parser.add_argument('--RRs', type=int, default=2, help='Position data')



args = parser.parse_args()


nc, bs = args.nc, args.bs
numd = args.numd
ncf = args.ncf*args.nc
niter = args.niter
optimizer = args.optimizer
lr = args.lr
a0, af, nsteps = 0.1, 1.0,  args.nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
#anneal = True
if args.RRs == 2: RRs = [1,  0]
elif args.RRs == 3: RRs = [2., 1.,  0]
print(RRs)

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
#fid_recon = Recon_DM(nc, bs, a0=a0, af=af, nsteps=nsteps, nbody=args.nbody, lpt_order=args.lpt_order, anneal=True)

ofolder = './figs/'

def get_data(nsims=args.nsims, posdata=True):

    path = '//mnt/ceph/users/cmodi/cosmo4d/z00/'
    path = path + '/L%04d_N%04d_D%04d//'%(bs, nc, numd*1e4)
    #if args.nbody: dpath = '/project/projectdirs/m3058/chmodi/rim-data/L%04d_N%03d_T%02d/'%(bs, nc, nsteps)
    #else: dpath = '/project/projectdirs/m3058/chmodi/rim-data/L%04d_N%03d_LPT%d/'%(bs, nc, args.lpt_order)
    alldata = np.array([np.load(path + 'S%04d.npy'%i) for i in range(100, 100+nsims)]).astype(np.float32)
    print(alldata.shape)
    if args.posdata:  traindata, testdata = alldata[:int(0.9*nsims),  [0,1]], alldata[int(0.9*nsims):,  [0,1]]
    else: traindata, testdata = alldata[:int(0.9*nsims),  [0,2]], alldata[int(0.9*nsims):,  [0,2]]
    return traindata, testdata


traindata, testdata = get_data()
print(traindata.shape, testdata.shape)
 


@tf.function
def pmpos(linear):
    if args.nbody:
        print('Nobdy sim')
        state = lpt_init(linear, a0=a0, order=args.lpt_order)
        final_state = nbody(state,  stages, nc)
    else:
        print('ZA/2LPT sim')
        final_state = lpt_init(linear, a0=af, order=args.lpt_order)
    tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
    return tfinal_field, final_state[0]



@tf.function
def biasfield(linear, bias):
    
    b1, b2 = bias[0], bias[1]
    final_field, fpos = pmpos(linear)
    w0 =  tf.reshape(linear, (linear.shape[0], -1))
    w0 = w0 - tf.expand_dims(tf.reduce_mean(w0, 1), -1)
    w2  = w0*w0
    w2 = w2 - tf.expand_dims(tf.reduce_mean(w2, 1), -1)
    weight = b1*w0 + b2*w2
    bmodel = cic_paint(tf.zeros_like(linear), fpos, weight = weight)
    #d0  = cic_paint(tf.zeros_like(linear), fpos, weight = w0)
    #d2 = cic_paint(tf.zeros_like(linear), fpos, weight = w2)
    #bmodel = b1*d0 + b2*d2
    return bmodel



@tf.function
def recon_model(linear, data, bias, errormesh):

    print('new graph')

    bmodel = biasfield(linear, bias)
    residual = bmodel - data
    resk = r2c3d(residual, norm=nc**3)
    reskmesh = tf.square(tf.cast(tf.abs(resk), tf.float32))
    chisq = tf.reduce_sum(tf.multiply(reskmesh, 1/errormesh))

    lineark = r2c3d(linear, norm=nc**3)
    priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
    if args.prior: prior = tf.reduce_sum(tf.multiply(priormesh, 1/priorwt))
    else: prior = tf.reduce_mean(tf.multiply(priormesh, 1/priorwt))

    loss = chisq + prior

    return loss, chisq, prior
    #return loss*nc**3, chisq*nc**3, prior*nc**3


@tf.function
def recon_grad(x, y, bias, errormesh):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = recon_model(x, y, bias, errormesh)[0]
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
    else: plt.savefig(ofolder + 'rim-im.png')
    plt.close()



def check_2pt(xx, yy, rim, grad_fn, grad_params, compares, nrim=10, fname=None):
    truemesh = [xx[0], yy[0]]
    rimpreds = []
    for it in range(nrim):
        x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
        #x_init = (yy - (yy.max() - yy.min())/2.)/yy.std() + np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
        pred = rim(tf.constant(x_init), tf.constant(yy), grad_fn, grad_params)[-1]
        rimpreds.append([pred[0].numpy(), biasfield(pred, grad_params[0])[0].numpy()])

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
    if fname is not None: plt.savefig(fname)
    else: plt.savefig('rim-2pt.png')
    plt.close()



def setupbias(nsims = 2, cutoff=1.5):

    b1, b2, perr = [], [], []
    for i in range(nsims):
        idx = np.random.randint(0, traindata.shape[0], 1)
        idx = idx*0 + 1
        xx = traindata[idx, 0].astype(np.float32)
        yy = traindata[idx, 1].astype(np.float32)
        _, fpos = pmpos(tf.constant(xx))
        fpos = fpos[0].numpy() *bs/nc
        bparams, bmodel = getbias(bs, nc, yy[0]+1, xx[0], fpos)
        errormesh = yy - np.expand_dims(bmodel, 0)
        kerror, perror = tools.power(errormesh[0]+1, boxsize=bs)
        kerror, perror = kerror[1:], perror[1:]
        perr += [perror]
        b1 += [bparams[0]]
        b2 += [bparams[1]]
    print("b1 : %0.3f $\pm$ %0.2f"%(np.array(b1).mean(), np.array(b1).std()))
    print("b2 : : %0.3f $\pm$ %0.2f"%(np.array(b2).mean(), np.array(b2).std()))
    b1, b2 = np.array(b1).mean(), np.array(b2).mean()

    perr = np.array(perr).mean(axis=0)
    kny = nc*np.pi/bs
    perr[np.where(kerror > cutoff*kny)] = perr[np.where(kerror > cutoff*kny)[0][0]]
    
    ipkerror = lambda x: 10**np.interp(np.log10(x), np.log10(kerror), np.log10(perr))
    errormesh = tf.constant(ipkerror(kmesh), dtype=tf.float32)
    return b1, b2, errormesh


def main():
    """
    Model function for the CosmicRIM.
    """

    if args.posdata: suff = 'pos'
    else: suff = 'mass'
    if args.nbody: suff = suff + '-T%02d'%nsteps
    else: suff = suff + '-LPT2'
    if args.prior : pass
    else: suff = suff + '-noprior'
    if len(RRs) !=2 : suff = suff + "-RR%d"%(len(RRs))
    print(suff)

    rim = build_rim_parallel(params)
    #grad_fn = recon_dm_grad
    #
    b1, b2, errormesh = setupbias()
    bias = tf.constant([b1, b2], dtype=tf.float32)
    grad_fn = recon_grad
    grad_params = [bias, errormesh]

    idx = np.random.randint(0, testdata.shape[0], 1)
    idx = idx*0  + 1
    xx, yy = testdata[idx, 0].astype(np.float32), testdata[idx, 1].astype(np.float32), 
    x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
    fid_recon = Recon_Bias(nc, bs, bias, errormesh, a0=0.1, af=1.0, nsteps=args.nsteps, nbody=args.nbody, lpt_order=2, anneal=True, prior=args.prior)
    #minic, minfin = fid_recon.reconstruct(tf.constant(yy), RRs=[1.0, 0.0], niter=args.rim_iter*10, lr=0.1)
    
    print("Loss at truth : ", recon_model(tf.constant(xx), tf.constant(yy),  *[bias, errormesh]))
    print("Loss at init : ", recon_model(tf.constant(x_init), tf.constant(yy),  *[bias, errormesh]))

    pred_adam = adam(tf.constant(x_init), tf.constant(yy), grad_fn, [bias, errormesh])
    print("Loss at adam : ", recon_model(tf.constant(pred_adam), tf.constant(yy),  *[bias, errormesh]))
    pred_adam = [pred_adam[0].numpy(), biasfield(pred_adam, bias)[0].numpy()]

    pred_adam10 = adam10(tf.constant(x_init), tf.constant(yy), grad_fn, [bias, errormesh])
    print("Loss at adam 10x : ", recon_model(tf.constant(pred_adam10), tf.constant(yy),  *[bias, errormesh]))
    pred_adam10 = [pred_adam10[0].numpy(), biasfield(pred_adam10, bias)[0].numpy()]
    minic, minfin = fid_recon.reconstruct(tf.constant(yy), RRs=RRs, niter=args.rim_iter*10, lr=0.1)
    compares =  [pred_adam, pred_adam10, [minic[0], minfin[0]]]

    check_im(xx[0], x_init[0], minic[0], fname= './figs/L%04d-N%03d-%s-im.png'%(bs, nc, suff))
    check_2pt(xx, yy, rim, grad_fn, grad_params, compares, fname= './figs/L%04d-N%03d-%s-2pt.png'%(bs, nc, suff))
    print('Test set generated')

    sys.exit()


    x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
    x_pred = rim(x_init, yy, grad_fn, grad_params)

    

    #
    # @tf.function
    def rim_train(x_true, x_init, y):

        with tf.GradientTape() as tape:
            x_pred = rim(x_init, y, grad_fn, grad_params)
            res  = (x_true - x_pred)
            loss = tf.reduce_mean(tf.square(res))
        gradients = tape.gradient(loss, rim.trainable_variables)
        return loss, gradients


    ##Train and save
    piter, testiter  = 10, 20
    losses = []
    lrs = [0.001, 0.0005, 0.0001]
    liters = [201, 1001, 1001]
    trainiter = 0 
    start = time.time()
    x_test, y_test = None, None

    for il in range(3):
        print('Learning rate = %0.3e'%lrs[il])
        opt = tf.keras.optimizers.Adam(learning_rate=lrs[il])

        for i in range(liters[il]):
            idx = np.random.randint(0, traindata.shape[0], args.batch_size)
            xx, yy = traindata[idx, 0].astype(np.float32), traindata[idx, 1].astype(np.float32), 
            x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
            #x_init = (yy - (yy.max() - yy.min())/2.)/yy.std() + np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
            

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
                plt.savefig(ofolder + 'losses.png')
                plt.close()

                if x_test is None:
                    idx = np.random.randint(0, testdata.shape[0], 1)
                    x_test, y_test = testdata[idx, 0].astype(np.float32), testdata[idx, 1].astype(np.float32), 
                    print("Loss at truth : ", recon_model(tf.constant(x_test), tf.constant(y_test),  *[bias, errormesh]))
                    print("Loss at init : ", recon_model(tf.constant(x_init), tf.constant(y_test),  *[bias, errormesh]))
                    
                    pred_adam = adam(tf.constant(x_init), tf.constant(y_test), grad_fn, [bias, errormesh])
                    print("Loss at adam : ", recon_model(tf.constant(pred_adam), tf.constant(y_test),  *[bias, errormesh]))
                    pred_adam = [pred_adam[0].numpy(), biasfield(pred_adam, bias)[0].numpy()]
                    
                    pred_adam10 = adam10(tf.constant(x_init), tf.constant(y_test), grad_fn, [bias, errormesh])
                    print("Loss at adam 10x : ", recon_model(tf.constant(pred_adam10), tf.constant(y_test),  *[bias, errormesh]))
                    pred_adam10 = [pred_adam10[0].numpy(), biasfield(pred_adam10, bias)[0].numpy()]
                    minic, minfin = fid_recon.reconstruct(tf.constant(y_test), RRs=[1.0, 0.0], niter=args.rim_iter*10, lr=0.1)
                    compares =  [pred_adam, pred_adam10, [minic[0], minfin[0]]]
                    check_2pt(x_test, y_test, rim, grad_fn, grad_params, compares, fname= 'halosrecon.png')
                    print('Test set generated')

                x_init = np.random.normal(size=x_test.size).reshape(x_test.shape).astype(np.float32)
                #x_init = (y_test - (y_test.max() - y_test.min())/2.)/y_test.std() + np.random.normal(size=x_test.size).reshape(x_test.shape).astype(np.float32)
                pred = rim(tf.constant(x_init), tf.constant(y_test), grad_fn, grad_params)[-1]
                check_im(x_test[0], x_init[0], pred.numpy()[0], fname=ofolder + 'rim-im-%04d.png'%trainiter)
                check_2pt(x_test, y_test, rim, grad_fn, grad_params, compares, fname=ofolder + 'rim-2pt-%04d.png'%trainiter)
                rim.save_weights(ofolder + '/%d'%trainiter)

            trainiter  += 1





#$#$
#$#$
#$#$    k, ps = tools.power(xx[0], boxsize=bs)
#$#$    k, pf = tools.power(zz[0], boxsize=bs)
#$#$    k, ph = tools.power(yy[0], boxsize=bs)
#$#$    k, pb = tools.power(bmodel, boxsize=bs)
#$#$    k, pxsh = tools.power(xx[0], f2=yy[0],  boxsize=bs)
#$#$    k, pxfh = tools.power(zz[0], f2=yy[0],  boxsize=bs)
#$#$    k, pxbh = tools.power(bmodel, f2=yy[0],  boxsize=bs)
#$#$
#$#$    fig, ax = plt.subplots(1, 2, figsize = (9, 4))
#$#$    ax[0].plot(k, (ps*ps)**0.5, 'C%d-'%0, alpha=0.7)
#$#$    ax[0].plot(k, (pf*pf)**0.5, 'C%d-'%1, alpha=0.7)
#$#$    ax[0].plot(k, (ph*ph)**0.5, 'C%d-'%2, alpha=0.7)
#$#$    ax[0].plot(k, pb, 'C%d--'%3, alpha=0.7)
#$#$    ax[0].plot(kerror, perror, 'C%d:'%4, alpha=0.7)
#$#$    ax[1].plot(k, pxsh/(ps*ph)**0.5, 'C%d-'%0, alpha=0.7)
#$#$    ax[1].plot(k, pxfh/(pf*ph)**0.5, 'C%d-'%1, alpha=0.7)
#$#$    ax[1].plot(k, pxbh/(pb*ph)**0.5, 'C%d-'%3, alpha=0.7)
#$#$    ax[0].loglog()
#$#$    ax[1].semilogx()
#$#$    ax[1].set_ylim(-0.1, 1.2)
#$#$    for axis in ax: axis.grid(which='both')
#$#$    plt.savefig('2pt-halos.png')
#$#$    plt.close()
#$#$
#$#$    print(xx.shape, yy.shape, zz.shape)
#$#$    fig, axar = plt.subplots(2, 3, figsize = (9, 8))
#$#$    ax = axar[0]
#$#$    ax[0].imshow(xx[0].sum(axis=0))
#$#$    ax[1].imshow(zz[0].sum(axis=0))
#$#$    ax[2].imshow(yy[0].sum(axis=0))
#$#$    ax = axar[1]
#$#$    ax[0].imshow(yy[0].sum(axis=0))
#$#$    ax[1].imshow(bmodel.sum(axis=0))
#$#$    ax[2].imshow(errormesh[0].sum(axis=0))
#$#$    plt.savefig('testhalos.png')
#$#$
#$#$
#$#$
#$#$    idx = np.random.randint(0, traindata.shape[0], 4)
#$#$    xx = traindata[idx, 0].astype(np.float32)
#$#$    bmodel = biasfield(tf.constant(xx), tf.constant([b1, b2], dtype=tf.float32)).numpy()
#$#$    yy = traindata[idx, 1].astype(np.float32)
#$#$
#$#$    print(xx.shape, bmodel.shape, yy.shape)
#$#$
#$#$    fig, ax = plt.subplots(1, 2, figsize = (9, 4))
#$#$    for i in range(len(idx)):
#$#$        print(i)
#$#$        k, ps = tools.power(xx[i], boxsize=bs)
#$#$        k, ph = tools.power(yy[i], boxsize=bs)
#$#$        k, pb = tools.power(bmodel[i], boxsize=bs)
#$#$        k, pxsh = tools.power(xx[i], f2=yy[i],  boxsize=bs)
#$#$        k, pxbh = tools.power(bmodel[i], f2=yy[i],  boxsize=bs)
#$#$        
#$#$        ax[0].plot(k, ps, 'C%d-'%0, alpha=0.7)
#$#$        ax[0].plot(k, ph, 'C%d-'%2, alpha=0.7)
#$#$        ax[0].plot(k, pb, 'C%d--'%3, alpha=0.7)
#$#$        ax[1].plot(k, pxsh/(ps*ph)**0.5, 'C%d-'%0, alpha=0.7)
#$#$        ax[1].plot(k, pxbh/(pb*ph)**0.5, 'C%d-'%3, alpha=0.7)
#$#$        ax[0].loglog()
#$#$        ax[1].semilogx()
#$#$        ax[1].set_ylim(-0.1, 1.2)
#$#$        for axis in ax: axis.grid(which='both')
#$#$    plt.savefig('2pt-halos-tf.png')
#$#$    plt.close()
#$#$
#$#$
#$#$
if __name__=="__main__":
    main()
