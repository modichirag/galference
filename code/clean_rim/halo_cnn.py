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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from rim_utils import  build_rim_parallel, myAdam
from cnn_utils import  SimpleUNet
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
parser.add_argument('--ncf', type=int, default=128, help='Grid size')
parser.add_argument('--bs', type=float, default=200, help='Box Size')
parser.add_argument('--numd', type=float, default=0.001, help='number density')
parser.add_argument('--nsteps', type=int, default=3, help='')
parser.add_argument('--niter', type=int, default=200, help='Number of iterations/Max iterations')
parser.add_argument('--rim_iter', type=int, default=10, help='Number of iterations/Max iterations')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='adam', help='Which optimizer to use')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--nsims', type=int, default=100, help='Number of simulations')
parser.add_argument('--nbody', type=str2bool, default=False, help='Number of simulationss')
parser.add_argument('--lpt_order', type=int, default=2, help='Order of LPT Initial conditions')
parser.add_argument('--cell_size', type=int, default=32, help='Cell channel size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--suffix', type=str, default='', help='Suffix for folder pathname')
parser.add_argument('--batch_in_epoch', type=int, default=20, help='Number of batches in epochs')
parser.add_argument('--posdata', type=str2bool, default=True, help='Position data')
parser.add_argument('--parallel', type=str2bool, default=True, help='Parallel')
parser.add_argument('--stdinit', type=str2bool, default=True, help='Standard IC')
parser.add_argument('--priorinit', type=str2bool, default=False, help='Start with priorinit')
parser.add_argument('--prior', type=str2bool, default=True, help='Start with priorinit')
parser.add_argument('--diffps', type=str2bool, default=True, help='Start with priorinit')
parser.add_argument('--Rstd', type=int, default=128, help='Parallel')
parser.add_argument('--nsimsbias', type=int, default=10, help='Parallel')



args = parser.parse_args()

if args.stdinit and args.priorinit:
    print('Both std and prior init are true')
    sys.exit()

nc, bs = args.nc, args.bs
numd = args.numd
ncf = args.ncf
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
kvec = tools.fftk((nc, nc, nc), boxsize=nc, symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
priorwt = ipklin(kmesh)

args.kmesh = kmesh
args.ipklin = ipklin
args.priorwt = priorwt

datamodel = HaloData(args)

########################################


###RIM params
##params = {}
##params['input_size'] = args.input_size
##params['cell_size'] = args.cell_size
##params['strides'] = 2
##params['middle_size'] = args.input_size // params['strides']  #lets divide by strides
##params['cell_kernel_size'] = 5
##params['input_kernel_size'] = 5
##params['middle_kernel_size'] = 5
##params['output_kernel_size'] = 5
##params['rim_iter'] = args.rim_iter
##params['input_activation'] = 'tanh'
##params['output_activation'] = 'linear'
##params['nc'] = nc
##

adam = myAdam(10)
adam10 = myAdam(100)
#
cnn = SimpleUNet(args.cell_size, kernel_size=5)
#optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

step = tf.Variable(0, trainable=False)
boundaries = [100, 1000]
values = [args.lr, args.lr/2., args.lr/5.]
learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)
learning_rate = learning_rate_fn(step)
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    args.lr,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)



#################################



traindata, testdata = get_data(args)
print(traindata.shape, testdata.shape)
ipkdiff, b1eul = get_diff_spectra(args, ipklin, nsims=args.nsimsbias, nsteps=args.nsteps)
print("B1 eulerian : ", b1eul)

BUFFER_SIZE = len(traindata)
GLOBAL_BATCH_SIZE = args.batch_size
train_dataset = tf.data.Dataset.from_tensor_slices((traindata[:, 0], traindata[:, 1:])).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
test_dataset = tf.data.Dataset.from_tensor_slices((testdata[:, 0], testdata[:, 1:])).shuffle(len(testdata)).batch(1) 

grad_fn = datamodel.recon_grad
bias, errormesh = datamodel.setupbias(traindata, nsims=10)
print(bias)
grad_params = [bias, errormesh]


if args.parallel: suffpath = '_halo_unet' + args.suffix
else: suffpath = '_halo_split' + args.suffix
if args.nbody: ofolder = './models/L%04d_N%03d/T%02d%s/'%(bs, nc, nsteps, suffpath)
else: ofolder = './models/L%04d_N%03d/LPT%d%s/'%(bs, nc, args.lpt_order, suffpath)
try: os.makedirs(ofolder)
except Exception as e: print(e)


#######################################


x_test, y_test = testdata[0:1, 0], testdata[0:1, 1:]
x_test = tf.constant(x_test, dtype=tf.float32)
if args.stdinit:
    x_init = tf.constant(y_test[:, 1] / b1eul , dtype=tf.float32)
    if args.diffps : x_init = x_init + linear_field(nc, bs, ipkdiff, batch_size=y_test.shape[0])
elif args.priorinit:
    x_init =  linear_field(nc, bs, ipklin, batch_size=y_test.shape[0])
else: 
    x_init = tf.random.normal(x_test.shape)
y_test = tf.constant(y_test[:, 0])
minic, minfin = datamodel.reconstruct(tf.constant(y_test), bias, errormesh,
                                      RRs=RRs, niter=args.rim_iter*20, lr=0.5, x_init=x_init, useprior=True)
pred_adam, _ = datamodel.reconstruct(tf.constant(y_test), bias, errormesh,
                                      RRs=[0.0], niter=args.rim_iter, lr=0.5, x_init=x_init, useprior=True)
pred_adam10, _ = datamodel.reconstruct(tf.constant(y_test), bias, errormesh,
                                      RRs=[0.0], niter=args.rim_iter*10, lr=0.5, x_init=x_init, useprior=True)

check_2pt(datamodel,
          #[[x_test, y_test], [x_init, minic]], 
          #[[x_test, y_test], [pred_adam, pred_adam10, minic]], grad_params, ofolder + 'fid_recon')
          [[x_test+1., y_test], [x_init+1., minic+1.]], 
          [[x_test+1., y_test], [pred_adam+1., pred_adam10+1., minic+1.]], grad_params, ofolder + 'fid_recon')

#######################################


def train_step(inputs):
    x_true, y = inputs
    #x_init = tf.random.normal(x_true.shape)
    if args.stdinit:
        print('std init')
        x_init = y[:, 1] / b1eul + linear_field(nc, bs, ipkdiff, batch_size=y.shape[0])
        #x_init = y[:, 1]
        y = y[:, 0]
    else:
        print('data init')
        x_init = y
    with tf.GradientTape() as tape:
        x_pred = cnn(tf.expand_dims(x_init, -1))[..., 0]
        print(x_pred.shape)
        res  = (x_true - x_pred)
        #print(res.shape)
        loss = tf.reduce_mean(tf.square(res))
    gradients = tape.gradient(loss, cnn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, cnn.trainable_variables))
    print("learnign rate : ", optimizer._decayed_lr(tf.float32))
    print('Len trainable : ', len(cnn.trainable_variables))
    return loss


def test_step(inputs):
    x_true, y = inputs
    #x_init = tf.random.normal(x_true.shape)
    if args.stdinit:
        x_init = y[:, 1] / b1eul + linear_field(nc, bs, ipkdiff, batch_size=y.shape[0])
        #x_init = y[:, 1]
        y = y[:, 0]
    else:
        print('data init')
        x_init = y
    x_pred = cnn(tf.expand_dims(x_init, -1))[..., 0]
    #x_pred = cnn(x_init)
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
    print("Train loss for epoch %d "%epoch, train_loss)
    print("Time taken for epoch %d: "%epoch, time.time() - starte)
    plt.plot(losses)
    plt.savefig(ofolder + 'losses.png')

    ##Test Epoch Training
    for x in test_dataset:
        print('Testing')
        #print(len(x), x[0].values[0].shape)
        a, b, c, d = test_step(x)
        #print(a.values[0].shape, b.values[0].shape, c.values[0].shape, d.values[0].shape)
        try: pred, x_init, xx, yy = a.values[0], b.values[0], c.values[0], d.values[0]
        except: pred, x_init, xx, yy = a, b, c, d
        #pred_adam = adam(x_init, yy, grad_fn, grad_params)
        #pred_adam10 = adam10(x_init, yy, grad_fn, grad_params)
        
        check_im(xx[0].numpy(), x_init[0].numpy(), pred[0].numpy(), ofolder + 'cnn-im-%d.png'%epoch)
        check_2pt(datamodel,
                  #[[xx, yy], [x_init, pred]], 
                  #[[x_test, y_test], [pred_adam, pred_adam10, minic]], grad_params, ofolder + 'rim-2pt-%d.png'%epoch)
                  [[xx+1., yy], [x_init+1., pred+1.]], 
                  [[x_test+1., y_test], [pred_adam+1., pred_adam10+1., minic+1.]], grad_params, ofolder + 'rim-2pt-%d.png'%epoch)

        break
    
    cnn.save_weights(ofolder + '/%d'%epoch)
