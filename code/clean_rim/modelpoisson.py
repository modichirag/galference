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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


import flowpm
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d
sys.path.append('../../utils/')
import tools


class PoissonData():

    def __init__(self, args):
        self.args = args

        

    @tf.function
    def pm(self, linear):
        args = self.args
        nc, bs = args.nc, args.bs

        if args.nbody:
            print('Nobdy sim')
            state = lpt_init(linear, a0=args.a0, order=args.lpt_order)
            final_state = nbody(state, args.stages, nc)
        else:
            print('ZA/2LPT sim')
            final_state = lpt_init(linear, a0=args.af, order=args.lpt_order)
        tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
        return tfinal_field


    @tf.function
    def gal_sample(self, base):
        plambda = self.args.plambda        
        galmean = tfp.distributions.Poisson(rate = plambda * (1 + base))
        return galmean.sample()

    
    @tf.function()
    def pm_data(self, dummy):
        args = self.args
        nc, bs = args.nc, args.bs

        print("PM graph")
        linear = flowpm.linear_field(nc, bs, args.ipklin, batch_size=args.batch_size)
        base = self.pm(linear)
        sample = self.gal_sample(base)
        return linear, sample


    @tf.function()
    def pm_data_test(self, dummy):
        args = self.args
        nc, bs = args.nc, args.bs
        
        print("PM graph")
        linear = flowpm.linear_field(nc, bs, args.ipklin, batch_size=args.world_size)
        base = self.pm(linear)
        sample = self.gal_sample(base)
        return linear, sample


    

    @tf.function
    def recon(self, linear, data):
        """                                                                                                                
        """
        args = self.args
        print('new recon graph')
        base = self.pm(linear)

        galmean = tfp.distributions.Poisson(rate = args.plambda * (1 + base))
        logprob = -tf.reduce_mean(galmean.log_prob(data))
        #logprob = tf.multiply(logprob, 1/nc**3, name='logprob')
        #Prior
        lineark = r2c3d(linear, norm=args.nc**3)
        priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
        prior = tf.reduce_mean(tf.multiply(priormesh, 1/args.priorwt))
        #prior = tf.multiply(prior, 1/nc**3, name='prior')

        loss = logprob + prior
        return loss, logprob, prior


    @tf.function
    def recon_grad(self, x, y):
        with tf.GradientTape() as tape:
            tape.watch(x)
            loss = self.recon(x, y)[0]
        grad = tape.gradient(loss, x)
        return grad



def get_ps(bs, iterand, truth):
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



def check_2pt(datamodel, xx, yy, x_init, pred, pred_adam, pred_adam10, fname):
    
    nc, bs = datamodel.args.nc, datamodel.args.bs
    
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    k, pks = get_ps(bs, [x_init.numpy(), datamodel.gal_sample(datamodel.pm(x_init)).numpy()], [xx.numpy(), yy.numpy()])
    for i in range(2):
        ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d--'%i, lw=0.5)
        ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d--'%i, lw=0.5)

    k, pks = get_ps(bs, [pred.numpy(), datamodel.gal_sample(datamodel.pm(pred)).numpy()], [xx.numpy(), yy.numpy()])
    for i in range(2):
        ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d'%i)
        ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d'%i)

    k, pks = get_ps(bs, [pred_adam.numpy(), datamodel.gal_sample(datamodel.pm(pred_adam)).numpy()], [xx.numpy(), yy.numpy()])
    for i in range(2):
        ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d-.'%i, lw=0.5)
        ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d-.'%i, lw=0.5)

    k, pks = get_ps(bs, [pred_adam10.numpy(), datamodel.gal_sample(datamodel.pm(pred_adam10)).numpy()], [xx.numpy(), yy.numpy()])
    for i in range(2):
        ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d:'%i)
        ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d:'%i)

    for axis in ax: 
        axis.semilogx()
        axis.grid(which='both')
    ax[0].set_ylim(-0.1, 1.2)
    ax[1].set_ylim(-0.5, 2.5)
    plt.savefig(fname)
    plt.close()

    
###
###def check_2pt(xx, yy, rim, grad_fn, compares, nrim=10, fname=None):
###    truemesh = [xx[0], yy[0]]
###    rimpreds = []
###    for it in range(nrim):
###        x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
###        pred = rim(tf.constant(x_init), tf.constant(yy), grad_fn)[-1]
###        rimpreds.append([pred[0].numpy(), gal_sample(pm(pred))[0].numpy()])
###
###    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
###    for ip, preds in enumerate(rimpreds):
###        k, pks = tools.get_ps(preds, truemesh, bs)
###        for i in range(2):
###            lbl = None
###            if ip == 0 and i == 0: lbl = 'Linear'
###            if ip == 0 and i == 1: lbl = 'Final'
###            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d-'%i, alpha=0.4, label=lbl)
###            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d-'%i, alpha=0.4)
###
###    lss = ['-', '--', ':', '-.']
###    lws = [ 1, 1, 2, 2]
###    lbls = ['Adam', 'Adam 10x', 'Best recon']
###    #for ip, preds in enumerate([pred_adam, pred_adam10]):
###    for ip, preds in enumerate(compares):
###        k, pks = tools.get_ps(preds, truemesh, bs)
###        for i in range(2):
###            lbl = None
###            if i == 0: lbl = lbls[ip]
###            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d'%i, ls=lss[ip+1], lw=lws[ip+1])
###            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d'%i,  label=lbl, ls=lss[ip+1], lw=lws[ip+1])
###        
###    for axis in ax: 
###        axis.semilogx()
###        axis.grid(which='both')
###        axis.legend(fontsize=12)
###        axis.set_xlabel('k(h/Mpc)', fontsize=12)
###    ax[0].set_ylim(-0.1, 1.2)
###    ax[1].set_ylim(-0.5, 2.0)
###    ax[0].set_ylabel('$r_c$', fontsize=12)
###    ax[1].set_ylabel('$t_f$', fontsize=12)
###    plt.tight_layout()
###    if fname is not None: plt.savefig(fname)
###    else: plt.savefig('rim-2pt.png')
###    plt.close()
###
