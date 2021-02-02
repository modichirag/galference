""" Implementation of Cosmic RIM estimator"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tensorflow_probability as tfp


import numpy as np
import os, sys, argparse, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


import flowpm
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d
sys.path.append('../../utils/')
import tools
import tools
from getbiasparams import getbias
import diagnostics as dg


class HaloData():

    def __init__(self, args):
        self.args = args

    @tf.function
    def pmpos(self, linear):
        args = self.args
        if args.nbody:
            print('Nobdy sim')
            state = lpt_init(linear, a0=args.a0, order=args.lpt_order)
            final_state = nbody(state, args.stages, args.nc)
        else:
            print('ZA/2LPT sim')
            final_state = lpt_init(linear, a0=args.af, order=args.lpt_order)
        tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
        return tfinal_field, final_state[0]




    @tf.function
    def biasfield(self, linear, bias):
        args = self.args
        b1, b2 = bias[0], bias[1]
        final_field, fpos = self.pmpos(linear)
        w0 =  tf.reshape(linear, (linear.shape[0], -1))
        w0 = w0 - tf.expand_dims(tf.reduce_mean(w0, 1), -1)
        w2  = w0*w0
        w2 = w2 - tf.expand_dims(tf.reduce_mean(w2, 1), -1)
        weight = b1*w0 + b2*w2
        bmodel = cic_paint(tf.zeros_like(linear), fpos, weight = weight)
        return bmodel


    @tf.function(
#        input_signature=[tf.TensorSpec(shape=[1, nc, nc, nc], dtype=tf.float32), 
#                         tf.TensorSpec(shape=[1, nc, nc, nc], dtype=tf.float32), 
#                         tf.TensorSpec(shape=[2], dtype=tf.float32), 
#                         tf.TensorSpec(shape=[nc, nc, nc], dtype=tf.float32) ]
    )
    def recon(self, linear, data, bias, errormesh):
        args = self.args
        bs, nc = args.bs, args.nc
        kmesh = args.kmesh
        priorwt = args.priorwt

        linear = tf.reshape(linear, data.shape)
        bmodel = self.biasfield(linear, bias)
        residual = bmodel - data
        base = residual
        
        resk = r2c3d(base, norm=nc**3)
        reskmesh = tf.square(tf.cast(tf.abs(resk), tf.float32))
        chisq = tf.reduce_sum(tf.multiply(reskmesh, 1/errormesh))
        #chisq = tf.reduce_mean(tf.multiply(reskmesh, 1/errormesh))
        #chisq = chisq * bs**3/nc**1.5

        if args.prior:
            lineark = r2c3d(linear, norm=nc**3)
            priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
            prior = tf.reduce_mean(tf.multiply(priormesh, 1/priorwt))
            prior = prior * bs**3/nc**1.5
        else: 
            print('\nRIM does not use prior\n')
            prior = 0.
        loss = chisq + prior
        return loss, chisq, prior
##
##        bmodel = self.biasfield(linear, bias)
##        residual = bmodel - data
##        resk = r2c3d(residual, norm=args.nc**3)
##        reskmesh = tf.square(tf.cast(tf.abs(resk), tf.float32))
##        chisq = tf.reduce_sum(tf.multiply(reskmesh, 1/errormesh))
##        chisq = chisq*bs**3 /nc**1.5
##
##        if args.prior:
##            lineark = r2c3d(linear, norm=args.nc**3)
##            priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
##            prior = tf.reduce_mean(tf.multiply(priormesh, 1/args.priorwt))
##            prior = prior*bs**3 / nc**1.5
##        else: prior = 0.
##        loss = chisq + prior
##
##        return loss, chisq, prior
##        #return loss*nc**3, chisq*nc**3, prior*nc**3
##
##
    @tf.function
    def recon_grad(self, x, y, bias, errormesh):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = self.recon(x, y, bias, errormesh)[0]
        grad = tape.gradient(loss, x)
        return grad


    
    @tf.function
    def reconstruct_loss(self, linear, data, bias, errormesh, Rsm=tf.constant(0.), useprior=True):
        """
        """
        args = self.args
        bs, nc = args.bs, args.nc
        kmesh = args.kmesh
        priorwt = args.priorwt

        linear = tf.reshape(linear, data.shape)
        bmodel = self.biasfield(linear, bias)
        residual = bmodel - data
        base = residual

        print("\nAdd annealing section to graph\n")
        Rsmsq = tf.multiply(Rsm*bs/nc, Rsm*bs/nc)
        smwts = tf.exp(tf.multiply(-kmesh**2, Rsmsq))
        basek = r2c3d(base, norm=nc**3)
        basek = tf.multiply(basek, tf.cast(smwts, tf.complex64))
        base = c2r3d(basek, norm=nc**3)   
        
        resk = r2c3d(base, norm=nc**3)
        reskmesh = tf.square(tf.cast(tf.abs(resk), tf.float32))
        chisq = tf.reduce_mean(tf.multiply(reskmesh, 1/errormesh))
        chisq = chisq * bs**3/nc**1.5

        if useprior:
            lineark = r2c3d(linear, norm=nc**3)
            priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
            prior = tf.reduce_mean(tf.multiply(priormesh, 1/priorwt))
            prior = prior * bs**3/nc**1.5
        else: prior = 0.
        loss = chisq + prior

        return loss #, chisq, prior

##        residual = bmodel - data
##        resk = r2c3d(residual, norm=args.nc**3)
##
##        print('\nAdd annealing graph\n')
##        Rsmsq = tf.multiply(Rsm*bs/nc, Rsm*bs/nc)
##        smwts = tf.exp(tf.multiply(-args.kmesh**2, Rsmsq))
##        resk = tf.multiply(resk, tf.cast(smwts, tf.complex64))
##
##        reskmesh = tf.square(tf.cast(tf.abs(resk), tf.float32))
##        logprob = tf.reduce_mean(tf.multiply(reskmesh, 1/errormesh))
##        logprob = logprob *bs**3/nc**1.5
##
##        #Prior
##        if useprior:
##            lineark = r2c3d(linear, norm=nc**3)
##            priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
##            prior = tf.reduce_mean(tf.multiply(priormesh, 1/args.priorwt))
##            prior = tf.multiply(prior, bs**3 / nc**1.5, name='prior')
##        else: prior = 0.
##        loss = logprob + prior
##        
##        return loss
##

    def reconstruct(self, data, bias, errormesh, RRs=[1.0, 0.0], niter=100, lr=0.1, x_init=None, useprior=True):

        print('reconstructing')
        args = self.args
        bs, nc = args.bs, args.nc

        @tf.function
        def grad(x, Rsm):
            with tf.GradientTape() as tape:
                tape.watch(x)
                loss = self.reconstruct_loss(x, data, bias, errormesh, Rsm, useprior=useprior)
            grad = tape.gradient(loss, x)
            return grad

            
        # Create an optimizer for Adam.
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        ##Reconstruction
        if x_init is None: 
            x_init = np.random.normal(0, 1, nc**3).reshape(data.shape).astype(np.float32) 
        linear = tf.Variable(name='linmesh', shape=data.shape, dtype=tf.float32,
                                 initial_value=x_init, trainable=True)


        for iR, RR in enumerate(RRs):
            print('For smoothing scale : ', RR)
            for i in range(niter):
                grads = grad([linear], tf.constant(RR, dtype=tf.float32))
                opt.apply_gradients(zip(grads, [linear]))
            minic = tf.reshape(linear, data.shape)
        #
        print('\nminimized\n')
        minsample = self.biasfield(minic, bias) #tf.constant(minic, dtype=tf.float32), bias).numpy()
        
        return minic, minsample


    def setupbias(self, traindata, nsims = 10, cutoff=1.5):

        args = self.args
        bs, nc = args.bs, args.nc
        b1, b2, perr = [], [], []
        for i in range(nsims):
            idx = np.random.randint(0, traindata.shape[0], 1)
            xx = traindata[idx, 0].astype(np.float32)
            yy = traindata[idx, 1].astype(np.float32)
            _, fpos = self.pmpos(tf.constant(xx))
            fpos = fpos[0].numpy() *bs/nc
            bparams, bmodel = getbias(bs, nc, yy[0]+1, xx[0], fpos)
            bmodeltf = self.biasfield(xx, tf.constant([bparams[0], bparams[1]], dtype=tf.float32)).numpy()
            errormesh = yy - bmodeltf# np.expand_dims(bmodel, 0)
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
        cutoff = 1.5
        perr[np.where(kerror > cutoff*kny)] = perr[np.where(kerror > cutoff*kny)[0][0]]
        ipkerror = interp1d(kerror, perr, bounds_error=False, fill_value=(perr[0], perr.max()))
        errormesh = tf.expand_dims(tf.constant(ipkerror(args.kmesh), dtype=tf.float32), 0)
        #ipkerror = lambda x: 10**np.interp(np.log10(x), np.log10(kerror), np.log10(perr))
        #errormesh = tf.constant(ipkerror(args.kmesh), dtype=tf.float32)
        bias = tf.constant([b1, b2], dtype=tf.float32)
        return bias, errormesh
    

#



def get_data(args, testfrac=0.9):
    
    bs, nc = args.bs, args.nc
    nsims = args.nsims
    numd = args.numd
    try: R=args.Rstd
    except: R=128
    path = '//mnt/ceph/users/cmodi/cosmo4d/z00/'
    #path = '/project/projectdirs/m3058/chmodi/rim-data/halos/z00/'
    path = path + '/L%04d_N%04d_D%04d//'%(bs, nc, numd*1e4)
    alldata = np.array([np.load(path + 'S%04d.npy'%i) for i in range(100, 100+nsims)]).astype(np.float32)
    if args.posdata:  traindata, testdata = alldata[:int(testfrac*nsims),  [0,1]], alldata[int(testfrac*nsims):,  [0,1]]
    else: traindata, testdata = alldata[:int(testfrac*nsims),  [0,2]], alldata[int(testfrac*nsims):,  [0,2]]
    if args.stdinit: 
        initdata = np.array([np.load(path + 'stdR%d_S%04d.npy'%(R,i)) for i in range(100, 100+nsims)]).astype(np.float32)
        traindatainit, testdatainit = initdata[:int(testfrac*nsims)], initdata[int(testfrac*nsims):]
        traindata = np.concatenate([traindata, np.expand_dims(traindatainit, 1)], axis=1)
        testdata = np.concatenate([testdata, np.expand_dims(testdatainit, 1)], axis=1)
    return traindata, testdata


def get_diff_spectra(args, ipklin,  nsims=10, nsteps=3):

    bs, nc = args.bs, args.nc
    nsims = args.nsims
    numd = args.numd
    try: R=args.Rstd
    except: R=128
    ncf=args.ncf
    
    path = '//mnt/ceph/users/cmodi/cosmo4d/z00/'
    dpath = path + '/L%04d_N%04d_D%04d//'%(bs, nc, numd*1e4)
    alldata = np.array([np.load(dpath + 'S%04d.npy'%i) for i in range(100, 100+nsims)]).astype(np.float32)
    initdata = np.array([np.load(dpath + 'stdR%d_S%04d.npy'%(R,i)) for i in range(100, 100+nsims)]).astype(np.float32)

    try:
        dyn = "%02dstep"%nsteps
        path = '//mnt/ceph/users/cmodi/cosmo4d/z00/'
        path = path + '/L%04d_N%04d_%s//'%(bs, nc, dyn)
        final = np.array([tools.readbigfile(path + '/L%04d_N%04d_S%04d_%02dstep/mesh/d/'%(bs, nc, seed, nsteps)) for seed in range(100, 100+nsims)]).astype(np.float32)
    except:
        dyn = "%02dstep_B1"%nsteps
        path = '//mnt/ceph/users/cmodi/cosmo4d/z00/'
        path = path + '/L%04d_N%04d_%s//'%(bs, nc, dyn)
        final = np.array([tools.readbigfile(path + '/L%04d_N%04d_S%04d_%02dstep/mesh/d/'%(bs, nc, seed, nsteps)) for seed in range(100, 100+nsims)]).astype(np.float32)
        
    print('alldata shape :', alldata.shape)
    pdiffs, bb = [], []
    for j in range(nsims):
        k, pfin = tools.power(final[j], boxsize=bs)
        ph = tools.power(1+alldata[j, 1], boxsize=bs)[1]
        bias = ((ph[1:5]/pfin[1:5])**0.5).mean()
        bb.append(bias)
        recon = initdata[j] / bias
        precon =tools.power(1+recon, boxsize=bs)[1]
        pdiff = ipklin(k) - precon
        pdiffs.append(pdiff)

    pdiff = np.array(pdiffs).mean(axis=0)
    bias = np.array(bb).mean(axis=0)
    xx, yy = k[pdiff > 0], pdiff[pdiff > 0]
    ipkdiff = lambda x: 10**np.interp(np.log10(x), np.log10(xx), np.log10(yy))

    return ipkdiff, bias




    
def get_ps(bs, iterand, truth):
    ic, fin = truth
    ic1, fin1 = iterand

    pks = []
    #if abs(ic1[0].mean()) < 1e-3: ic1[0] += 1
    #if abs(ic[0].mean()) < 1e-3: ic[0] += 1                                                                                                                  
    k, p1 = tools.power(ic1[0], boxsize=bs)
    k, p2 = tools.power(ic[0], boxsize=bs)
    k, p12 = tools.power(ic1[0], f2=ic[0], boxsize=bs)
    pks.append([p1, p2, p12])
    #if fin1[0].mean() < 1e-3: fin1[0] += 1
    #if fin[0].mean() < 1e-3: fin[0] += 1
    k, p1 = tools.power(fin1[0], boxsize=bs)
    k, p2 = tools.power(fin[0], boxsize=bs)
    k, p12 = tools.power(fin1[0], f2=fin[0], boxsize=bs)
    pks.append([p1, p2, p12])

    return k, pks
    


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





def check_2pt(datamodel, rim_test, fid_test, grad_params, fname):
    
    nc, bs = datamodel.args.nc, datamodel.args.bs
    
    fig, axar = plt.subplots(2, 2, figsize=(9, 9))
    a, b = rim_test
    xx, yy = a
    x_init, pred = b
    k, pks = get_ps(bs, [x_init.numpy(), datamodel.biasfield(x_init, grad_params[0]).numpy()], [xx.numpy(), yy.numpy()])
    for i in range(2):
        ax = axar[i]
        ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C0:', lw=1, label='Init')
        ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C0:', lw=1)

    k, pks = get_ps(bs, [pred.numpy(), datamodel.biasfield(pred, grad_params[0]).numpy()], [xx.numpy(), yy.numpy()])
    for i in range(2):
        ax= axar[i]
        ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d'%1, label='RIM')
        ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d'%1)

    #
    a, b = fid_test
    xx, yy = a
    pred_adam, pred_adam10, minic = b
    k, pks = get_ps(bs, [pred_adam.numpy(), datamodel.biasfield(pred_adam, grad_params[0]).numpy()], [xx.numpy(), yy.numpy()])
    for i in range(2):
        ax = axar[i]
        ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d:'%2, lw=1, label='Adam')
        ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d:'%2, lw=0.5)

    k, pks = get_ps(bs, [pred_adam10.numpy(), datamodel.biasfield(pred_adam10, grad_params[0]).numpy()], [xx.numpy(), yy.numpy()])
    for i in range(2):
        ax = axar[i]
        ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d'%2, label='Adam x10')
        ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d'%2)

    k, pks = get_ps(bs, [minic.numpy(), datamodel.biasfield(minic, grad_params[0]).numpy()], [xx.numpy(), yy.numpy()])
    for i in range(2):
        ax = axar[i]
        ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d-'%3, label='Annealed')
        ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d-'%3)


    for axis in axar.flatten(): 
        axis.legend()
        axis.semilogx()
        axis.grid(which='both')
    for ax in axar:
        ax[0].set_ylim(-0.1, 1.2)
        ax[1].set_ylim(-0.5, 2.5)
    plt.savefig(fname)
    plt.close()


##def check_2pt(xx, yy, rim, grad_fn, grad_params, compares, nrim=10, fname=None):
##    truemesh = [xx[0], yy[0]]
##    rimpreds = []
##    for it in range(nrim):
##        x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
##        #x_init = (yy - (yy.max() - yy.min())/2.)/yy.std() + np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
##        pred = rim(tf.constant(x_init), tf.constant(yy), grad_fn, grad_params)[-1]
##        rimpreds.append([pred[0].numpy(), biasfield(pred, grad_params[0])[0].numpy()])
##
##    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
##    for ip, preds in enumerate(rimpreds):
##        k, pks = tools.get_ps(preds, truemesh, bs)
##        for i in range(2):
##            lbl = None
##            if ip == 0 and i == 0: lbl = 'Linear'
##            if ip == 0 and i == 1: lbl = 'Final'
##            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d-'%i, alpha=0.4, label=lbl)
##            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d-'%i, alpha=0.4)
##
##    lss = ['-', '--', ':', '-.']
##    lws = [ 1, 1, 2, 2]
##    lbls = ['Adam', 'Adam 10x', 'Best recon']
##    #for ip, preds in enumerate([pred_adam, pred_adam10]):
##    for ip, preds in enumerate(compares):
##        k, pks = tools.get_ps(preds, truemesh, bs)
##        for i in range(2):
##            lbl = None
##            if i == 0: lbl = lbls[ip]
##            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d'%i, ls=lss[ip+1], lw=lws[ip+1])
##            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d'%i,  label=lbl, ls=lss[ip+1], lw=lws[ip+1])
##        
##    for axis in ax: 
##        axis.semilogx()
##        axis.grid(which='both')
##        axis.legend(fontsize=12)
##        axis.set_xlabel('k(h/Mpc)', fontsize=12)
##    ax[0].set_ylim(-0.1, 1.2)
##    ax[1].set_ylim(-0.5, 2.0)
##    ax[0].set_ylabel('$r_c$', fontsize=12)
##    ax[1].set_ylabel('$t_f$', fontsize=12)
##    plt.tight_layout()
##    if fname is not None: plt.savefig(fname)
##    else: plt.savefig('rim-2pt.png')
##    plt.close()
##
##
