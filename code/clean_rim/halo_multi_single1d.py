""" Implementation of Cosmic RIM estimator"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
world_size = len(physical_devices)
print("\nphysical_devices\n", physical_devices)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
for device in physical_devices:
    config = tf.config.experimental.set_memory_growth(device, True)

import tensorflow_probability as tfp
import numpy as np
import os, sys, argparse, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from rim_utils1d import  build_rim1d
from modelhalo import HaloData, check_2pt, check_im, get_data, get_diff_spectra

import flowpm
from flowpm import linear_field, lpt_init, nbody, cic_paint
from flowpm.utils import r2c3d, c2r3d
sys.path.append('../../utils/')
import tools
from getbiasparams import getbias
import diagnostics as dg


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

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
parser.add_argument('--diffps', type=str2bool, default=False, help='Diff PS')
parser.add_argument('--prior', type=str2bool, default=False, help='Use prior for RIM')



args = parser.parse_args()


nc, bs = args.nc, args.bs
numd = args.numd
ncf = args.nc
niter = args.niter
lr = args.lr
a0, af, nsteps = 0.1, 1.0,  args.nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
args.stages = stages
args.a0, args.af = a0, af
args.world_size = world_size
RRs = [1.0, 0.5, 0.0]
#
klin = np.loadtxt('../../data/Planck15_a1p00.txt').T[0]
plin = np.loadtxt('../../data//Planck15_a1p00.txt').T[1]
ipklin = iuspline(klin, plin)
# Compute necessary Fourier kernels                                                                                          
kvec = tools.fftk((nc, nc, nc), boxsize=bs, symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
priorwt = ipklin(kmesh)
kedges = np.histogram(kmesh.flatten(), bins=nc)[1]
kbinmap = np.digitize(np.expand_dims(kmesh, 0), kedges, right=False).astype(np.int32)
kbinmap[kbinmap == kbinmap.max()] = kbinmap.max()-1
kbinmap -= 1
#kbinmap = tf.constant(kbinmap)

args.kmesh = kmesh
args.ipklin = ipklin
args.priorwt = priorwt

datamodel = HaloData(args)

########################################
#RIM Params
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
params['bs'] = bs
params['batch_size'] = args.batch_size
params['epoch'] = args.epochs
params['cell_size1d'] = 128
params['input_activation1d'] = 'tanh'
params['output_activation1d'] = 'tanh'
params['kbinmap'] = kbinmap



#strategy = tf.distribute.MirroredStrategy(devices=["/device:GPU:0", "/device:GPU:1"])
strategy = tf.distribute.MirroredStrategy()
print ('\nNumber of devices: {}\n'.format(strategy.num_replicas_in_sync))
BATCH_SIZE_PER_REPLICA = params['batch_size'] // strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = params['batch_size']


#################################


traindata, testdata = get_data(args)
print(traindata.shape, testdata.shape)
if args.stdinit:
    ipkdiff, b1eul = get_diff_spectra(args, ipklin, nsims=args.nsimsbias, nsteps=3)
    print("B1 eulerian : ", b1eul)

train_dataset = tf.data.Dataset.from_tensor_slices((traindata[:, 0], traindata[:, 1:])).shuffle(len(traindata)).batch(GLOBAL_BATCH_SIZE) 
test_dataset = tf.data.Dataset.from_tensor_slices((testdata[:, 0], testdata[:, 1:])).batch(strategy.num_replicas_in_sync) 

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


if args.parallel: suffpath = '_halo_cutoff_w%d'%world_size + args.suffix
else: suffpath = '_halo_split' + args.suffix
if args.nbody: ofolder = './models/L%04d_N%03d/T%02d%s/'%(bs, nc, nsteps, suffpath)
else: ofolder = './models/L%04d_N%03d_1d/LPT%d%s/'%(bs, nc, args.lpt_order, suffpath)
print('Output in \n%s\n'%ofolder)
try: os.makedirs(ofolder)
except Exception as e: print(e)



############################################

with strategy.scope():
    rim = build_rim1d(params)
    grad_fn = datamodel.recon_grad
    bias, errormesh = datamodel.setupbias(traindata, nsims=args.nsimsbias)
    errormesh = tf.constant(np.expand_dims(errormesh, 0), dtype=tf.float32)
    print(bias)
    print(errormesh.shape)
    grad_params = [bias, errormesh]

    step = tf.Variable(0, trainable=False)
    boundaries = [100, 1000]
    values = [args.lr, args.lr/2., args.lr/5.]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values)
    #learning_rate = learning_rate_fn(step)
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr,
        decay_steps=args.decayiter,
        decay_rate=args.decay,
        staircase=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    checkpoint = tf.train.Checkpoint(model=rim)
    #

############################################

x_test, y_test = testdata[0:1, 0], testdata[0:1, 1:]
x_test = tf.constant(x_test, dtype=tf.float32)
#
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

bmodeltf = datamodel.biasfield(x_test, bias)
error = y_test - bmodeltf
k, ph = tools.power(y_test.numpy()[0], boxsize=bs)
k, pb = tools.power(bmodeltf.numpy()[0], boxsize=bs)
kerror, perr = tools.power(error.numpy()[0]+1, boxsize=bs)
kny = nc*np.pi/bs
cutoff = 1.5
perr[np.where(kerror > cutoff*kny)] = perr[np.where(kerror > cutoff*kny)[0][0]]
k, px = tools.power(y_test.numpy()[0], f2=bmodeltf.numpy()[0], boxsize=bs)

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].plot(k, ph, label='halo')
ax[0].plot(k, pb, '--', label='bias')
ax[0].plot(k, perr, label='error')
ax[0].loglog()
ax[1].plot(k, px/(ph*pb)**0.5)
ax[1].plot(k, (pb/ph)**0.5, '--', label='tf')
ax[1].semilogx()
ax[1].set_ylim(-0.2, 1.5)
ax[0].set_ylim(10, 1e5)
for axis in ax:
    axis.legend()
    axis.grid(which='both')
plt.savefig(ofolder + 'evalbfit.png')
plt.close()


def train_step(inputs):
    x_true, y = inputs
    if len(rim.trainable_variables) == 0:
        #Hack since sometimes this si the first time RIM is called and so hasn't been inisitalized
        i = 0
        b, c = y[i:i+1],  x_true[i:i+1]
        if args.stdinit:
            a = b[:, 1] / b1eul 
            if args.diffps : a = a + linear_field(nc, bs, ipkdiff, batch_size=b.shape[0])
        else: a = tf.random.normal(c.shape)
        b = b[:, 0]
        try: a, b, c = tf.constant(a), tf.constant(b),  tf.constant(c)
        except: pass
        _ =  rim(a, b, grad_fn, c,  grad_params)
    #
    gradients = [0.]*len(rim.trainable_variables)
    print("#number of trainable variables : ", len(gradients))
    for i in range(args.batch_size // world_size):
        with tf.GradientTape() as tape:
            b, c = y[i:i+1],  x_true[i:i+1]
            if args.stdinit:
                a = b[:, 1] / b1eul #+ linear_field(nc, bs, ipkdiff, batch_size=b.shape[0])
                if args.diffps : a = a + linear_field(nc, bs, ipkdiff, batch_size=b.shape[0])
            else: a = tf.random.normal(c.shape)
            b = b[:, 0]
            try: a, b, c = tf.constant(a), tf.constant(b),  tf.constant(c)
            except: pass
            #pred =  rim(a, b, grad_fn,  grad_params)
            #loss = tf.reduce_mean(tf.square((pred - c))) / args.batch_size
            loss =  rim(a, b, grad_fn, c, grad_params)[1] / args.batch_size
        grads = tape.gradient(loss, rim.trainable_variables)
        for j in range(len(grads)):
            gradients[j] = gradients[j] + grads[j] 
    optimizer.apply_gradients(zip(gradients, rim.trainable_variables))
    return loss



def test_step(inputs):
    x_true, y = inputs
    if args.stdinit:
        x_init = y[:, 1] / b1eul 
        if args.diffps : x_init = x_init + linear_field(nc, bs, ipkdiff, batch_size=y.shape[0])
    else: x_init = tf.random.normal(x_true.shape)
    y = y[:, 0]
    x_pred = rim(x_init, y, grad_fn, x_true, grad_params)[0]
    #x_pred =  rim(x_init, y, grad_fn,  grad_params)[-1]
    return x_pred, x_init, x_true, y



# `run` replicates the provided computation and runs it
# with the distributed input.
@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

@tf.function
def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs,))


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
    for x in train_dist_dataset:
        startb = time.time()
        loss = distributed_train_step(x)
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
    for x in test_dist_dataset:
        print('Testing')
        a, b, c, d = distributed_test_step(x)
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
        #check_2pt(datamodel, xx, yy, x_init, pred, pred_adam, pred_adam10, grad_params, ofolder + 'rim-2pt-%d.png'%epoch)

        break

    rim.save_weights(ofolder + '/%d'%epoch)
    
    

#    
