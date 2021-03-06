""" Implementation of Cosmic RIM estimator"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
world_size = len(physical_devices)

import numpy as np
import os, sys, argparse, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import json

from rim_utils import  build_rim_parallel_single, myAdam
from recon_models import Recon_Bias
from modelhalo import HaloData, check_2pt, check_im, get_data, get_diff_spectra

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
parser.add_argument('--decay', type=float, default=0.9, help='Decay rate')
parser.add_argument('--decayiter', type=int, default=100, help='Decay rate')
parser.add_argument('--optimizer', type=str, default='adam', help='Which optimizer to use')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--nsims', type=int, default=100, help='Number of simulations')
parser.add_argument('--nbody', type=str2bool, default=False, help='Number of simulationss')
parser.add_argument('--lpt_order', type=int, default=2, help='Order of LPT Initial conditions')
parser.add_argument('--input_size', type=int, default=8, help='Input layer channel size')
parser.add_argument('--cell_size', type=int, default=8, help='Cell channel size')
parser.add_argument('--rim_iter', type=int, default=10, help='Optimization iteration')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--suffix', type=str, default='', help='Suffix for folder pathname')
parser.add_argument('--batch_in_epoch', type=int, default=20, help='Number of batches in epochs')
parser.add_argument('--posdata', type=str2bool, default=True, help='Position data')
parser.add_argument('--parallel', type=str2bool, default=True, help='Parallel')
parser.add_argument('--stdinit', type=str2bool, default=False, help='Parallel')
parser.add_argument('--Rstd', type=int, default=128, help='Parallel')
parser.add_argument('--priorinit', type=str2bool, default=False, help='Start with priorinit')
parser.add_argument('--nsimsbias', type=int, default=10, help='Number of simulations to get bias')
parser.add_argument('--diffps', type=str2bool, default=False, help='Parallel')
parser.add_argument('--prior', type=str2bool, default=False, help='Use prior for RIM')



args = parser.parse_args()


nc, bs = args.nc, args.bs
numd = args.numd
ncf = args.ncf*args.nc
niter = args.niter
lr = args.lr
a0, af, nsteps = 0.1, 1.0,  args.nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
args.stages = stages
args.a0, args.af = a0, af
args.world_size = world_size
RRs = [2.0, 1.0, 0.5, 0.0]

#
klin = np.loadtxt('../../data/Planck15_a1p00.txt').T[0]
plin = np.loadtxt('../../data//Planck15_a1p00.txt').T[1]
ipklin = iuspline(klin, plin)
# Compute necessary Fourier kernels                                                                                          
kvec = tools.fftk((nc, nc, nc), boxsize=bs, symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
priorwt = ipklin(kmesh)

args.kmesh = kmesh
args.ipklin = ipklin
args.priorwt = priorwt

datamodel = HaloData(args)

########################################


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


rim = build_rim_parallel_single(params)
adam = myAdam(params['rim_iter'])
adam10 = myAdam(10*params['rim_iter'])
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    args.lr,
    decay_steps=args.decayiter,
    decay_rate=args.decay,
    staircase=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)



#################################



traindata, testdata = get_data(args)
print(traindata.shape, testdata.shape)
if args.stdinit:
    ipkdiff, b1eul = get_diff_spectra(args, ipklin, nsims=args.nsimsbias, nsteps=3)
    print("B1 eulerian : ", b1eul)

BUFFER_SIZE = len(traindata)
GLOBAL_BATCH_SIZE = args.batch_size
train_dataset = tf.data.Dataset.from_tensor_slices((traindata[:, 0], traindata[:, 1:])).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
test_dataset = tf.data.Dataset.from_tensor_slices((testdata[:, 0], testdata[:, 1:])).shuffle(len(testdata)).batch(1) 

grad_fn = datamodel.recon_grad
bias, errormesh = datamodel.setupbias(traindata, nsims=args.nsimsbias)
errormesh = tf.constant(np.expand_dims(errormesh, 0), dtype=tf.float32)
print(bias)
print(errormesh.shape)
grad_params = [bias, errormesh]

#


if args.parallel: suffpath = '_halo' + args.suffix
else: suffpath = '_halo_split' + args.suffix
if args.nbody: ofolder = './models/L%04d_N%03d/T%02d%s/'%(bs, nc, nsteps, suffpath)
else: ofolder = './models/L%04d_N%03d/LPT%d%s/'%(bs, nc, args.lpt_order, suffpath)
try: os.makedirs(ofolder)
except Exception as e: print(e)

with open(ofolder + 'params.json', 'w') as fp:
    json.dump(params, fp)
#######################################


x_test, y_test = testdata[0:1, 0], testdata[0:1, 1:]
x_test = tf.constant(x_test, dtype=tf.float32)
fpos = datamodel.pmpos(x_test)[1].numpy()[0]*bs/nc
bparams, bmodel = getbias(bs, nc, y_test[0, 0], x_test.numpy()[0], fpos)
bias_test = tf.constant([bparams[0], bparams[1]], dtype=tf.float32)
print('Bias test : ', bias_test)
bmodeltf = datamodel.biasfield(x_test, bias_test).numpy()
errormesh = y_test[:, 0] - bmodeltf
kerror, perror = tools.power(errormesh[0], boxsize=bs)
kerror, perror = kerror[1:], perror[1:]
ipkerror = interp1d(kerror, perror, bounds_error=False, fill_value=(perror[0], perror.max()))
errormesh_test = tf.expand_dims(tf.constant(ipkerror(kmesh), dtype=tf.float32), 0)
#
if args.stdinit:
    x_init = tf.constant(y_test[:, 1] / b1eul , dtype=tf.float32)
    if args.diffps : x_init = x_init + linear_field(nc, bs, ipkdiff, batch_size=y_test.shape[0])
elif args.priorinit:
    x_init =  linear_field(nc, bs, ipklin, batch_size=y_test.shape[0])
else: 
    x_init = tf.random.normal(x_test.shape)
y_test = tf.constant(y_test[:, 0])
pred_adam = adam(x_init, y_test, grad_fn, grad_params)
pred_adam10 = adam10(x_init, y_test, grad_fn, grad_params)
#fid_recon = Recon_Bias(nc, bs, bias, errormesh, a0=0.1, af=1.0, nsteps=args.nsteps, nbody=args.nbody, lpt_order=2, anneal=True, prior=True)    
#minic, minfin = fid_recon.reconstruct(tf.constant(y_test), RRs=RRs, niter=args.rim_iter*10, lr=0.1)
minic, minfin = datamodel.reconstruct(tf.constant(y_test), bias_test, errormesh_test,
                                      RRs=RRs, niter=args.rim_iter*20, lr=0.5, x_init=x_init, useprior=True)

check_2pt(datamodel,
          #[[x_test, y_test], [x_init, minic]], 
          #[[x_test, y_test], [pred_adam, pred_adam10, minic]], grad_params, ofolder + 'fid_recon')
          [[x_test+1., y_test], [x_init+1., minic+1.]], 
          [[x_test+1., y_test], [pred_adam+1., pred_adam10+1., minic+1.]], grad_params, ofolder + 'fid_recon')

#######################################



def train_step(inputs):
    x_true, y = inputs
    if args.stdinit:
        x_init = y[:, 1] / b1eul 
        if args.diffps : x_init = x_init + linear_field(nc, bs, ipkdiff, batch_size=y.shape[0])
    elif args.priorinit:
        x_init =  linear_field(nc, bs, ipklin, batch_size=y.shape[0])
    else: 
        x_init = tf.random.normal(x_true.shape)
    y = y[:, 0]
    if len(rim.trainable_variables) == 0:
        #Hack since sometimes this si the first time RIM is called and so hasn't been inisitalized
        i = 0
        a, b, c = x_init[i:i+1], y[i:i+1],  x_true[i:i+1]
        _ =  rim(tf.constant(a), tf.constant(b), grad_fn, tf.constant(c), grad_params)[1] / args.batch_size
            
    #
    gradients = [0.]*len(rim.trainable_variables)
    #n = args.sims_in_loop
    for i in range(args.batch_size // world_size):
        with tf.GradientTape() as tape:
            a, b, c = x_init[i:i+1], y[i:i+1],  x_true[i:i+1]
            loss =  rim(tf.constant(a), tf.constant(b), grad_fn, tf.constant(c), grad_params)[1] / args.batch_size
        grads = tape.gradient(loss, rim.trainable_variables)
        for j in range(len(grads)):
            gradients[j] = gradients[j] + grads[j] 
    optimizer.apply_gradients(zip(gradients, rim.trainable_variables))
    return loss


def test_step(inputs):
    x_true, y = inputs
    #x_init = tf.random.normal(x_true.shape)
    if args.stdinit:
        x_init = y[:, 1] / b1eul 
        if args.diffps: x_init = x_init + linear_field(nc, bs, ipkdiff, batch_size=y.shape[0])
    elif args.priorinit:
        x_init =  linear_field(nc, bs, ipklin, batch_size=y.shape[0])
    else: x_init = tf.random.normal(x_true.shape)
    y = y[:, 0]
    x_pred = rim(x_init, y, grad_fn, x_true, grad_params)[0]
    return x_pred, x_init, x_true, y



###########################################
####Train
###
#Training


losses = []    


for epoch in range(args.epochs):
    print("\nFor epoch %d\n"%epoch)
    #TRAIN LOOP
    total_loss = 0.0
    num_batches = 0
    starte = time.time()
    for x in train_dataset:
        #print(len(x), x[0].values[0].shape)
        startb = time.time()
        loss = train_step(x)
        losses.append(loss.numpy())
        total_loss += loss
        print("epoch %d, num batch %d, loss : "%(epoch, num_batches), loss)
        print("Time taken : ", time.time() - startb)
        num_batches += 1
    train_loss = total_loss / num_batches
    #print("Train loss for epoch %d "%epoch, train_loss)
    #print("Time taken for epoch %d: "%epoch, time.time() - starte)
    plt.plot(losses)
    plt.savefig(ofolder + 'losses.png')

    ##Test Epoch Training
    for x in test_dataset:
        print('Testing')
        a, b, c, d = test_step(x)
        #print(a.values[0].shape, b.values[0].shape, c.values[0].shape, d.values[0].shape)
        try: pred, x_init, xx, yy = a.values[0], b.values[0], c.values[0], d.values[0]



        except: pred, x_init, xx, yy = a, b, c, d
        #pred_adam = adam(x_init, yy, grad_fn, grad_params)
        #pred_adam10 = adam10(x_init, yy, grad_fn, grad_params)
        
        check_im(xx[0].numpy(), x_init[0].numpy(), pred[0].numpy(), ofolder + 'rim-im-%d.png'%epoch)
        check_2pt(datamodel,
                  #[[xx, yy], [x_init, pred]], 
                  #[[x_test, y_test], [pred_adam, pred_adam10, minic]], grad_params, ofolder + 'rim-2pt-%d.png'%epoch)
                  [[xx+1., yy], [x_init+1., pred+1.]], 
                  [[x_test+1., y_test], [pred_adam+1., pred_adam10+1., minic+1.]], grad_params, ofolder + 'rim-2pt-%d.png'%epoch)

        break
    
    rim.save_weights(ofolder + '/%d'%epoch)
