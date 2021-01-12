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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from modelpoisson import PoissonData, check_2pt, check_im, get_ps
from rim_utils import  build_rim_parallel_single, myAdam
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
parser.add_argument('--nc', type=int, default=32, help='Grid size')
parser.add_argument('--bs', type=float, default=200, help='Box Size')
parser.add_argument('--nsteps', type=int, default=3, help='')
parser.add_argument('--niter', type=int, default=200, help='Number of iterations/Max iterations')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='adam', help='Which optimizer to use')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--nsims', type=int, default=100, help='Number of simulations')
parser.add_argument('--nbody', type=str2bool, default=False, help='Number of simulationss')
parser.add_argument('--lpt_order', type=int, default=2, help='Order of LPT Initial conditions')
parser.add_argument('--input_size', type=int, default=16, help='Input layer channel size')
parser.add_argument('--cell_size', type=int, default=16, help='Cell channel size')
parser.add_argument('--rim_iter', type=int, default=10, help='Optimization iteration')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--suffix', type=str, default='', help='Suffix for folder pathname')
parser.add_argument('--batch_in_epoch', type=int, default=20, help='Number of batches in epochs')
parser.add_argument('--plambda', type=float, default=0.10, help='Poisson probability')
parser.add_argument('--parallel', type=str2bool, default=True, help='Parallel or Split')


args = parser.parse_args()


nc, bs = args.nc, args.bs
niter = args.niter
lr = args.lr
a0, af, nsteps = 0.1, 1.0,  args.nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
plambda = args.plambda
args.stages = stages
args.a0, args.af = a0, af
args.world_size = world_size

#
klin = np.loadtxt('../../data/Planck15_a1p00.txt').T[0]
plin = np.loadtxt('../../data//Planck15_a1p00.txt').T[1]
ipklin = iuspline(klin, plin)
# Compute necessary Fourier kernels                                                                                          
kvec = tools.fftk((nc, nc, nc), boxsize=nc, symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
priorwt = ipklin(kmesh)

args.ipklin = ipklin
args.priorwt = priorwt


datamodel = PoissonData(args)

############################

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
params['batch_size'] = args.batch_size
params['epoch'] = args.epochs


adam = myAdam(params['rim_iter'])
adam10 = myAdam(10*params['rim_iter'])
#fid_recon = Recon_Poisson(nc, bs, plambda=plambda, a0=a0, af=af, nsteps=nsteps, nbody=args.nbody, lpt_order=args.lpt_order, anneal=True)


#strategy = tf.distribute.MirroredStrategy(devices=["/device:GPU:0", "/device:GPU:1"])
strategy = tf.distribute.MirroredStrategy()
print ('\nNumber of devices: {}\n'.format(strategy.num_replicas_in_sync))
BATCH_SIZE_PER_REPLICA = params['batch_size'] // strategy.num_replicas_in_sync
GLOBAL_BATCH_SIZE = params['batch_size']


#################################################

if args.parallel: suffpath = '_p%03d_w%d_single'%(plambda*100, world_size) + args.suffix
else: suffpath = '_p%03d_split'%(plambda*100) + args.suffix
if args.nbody: ofolder = './models/poisson_L%04d_N%03d_T%02d%s/'%(bs, nc, nsteps, suffpath)
else: ofolder = './models/poisson_L%04d_N%03d_LPT%d%s/'%(bs, nc, args.lpt_order, suffpath)
try: os.makedirs(ofolder)
except Exception as e: print(e)


train_dataset = tf.data.Dataset.range(args.batch_in_epoch)
train_dataset = train_dataset.map(datamodel.pm_data)
train_dataset = train_dataset.prefetch(-1)
test_dataset = tf.data.Dataset.range(strategy.num_replicas_in_sync).map(datamodel.pm_data_test).prefetch(-1)

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


###############################################


with strategy.scope():
    rim = build_rim_parallel_single(params)
    grad_fn = datamodel.recon_grad
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    checkpoint = tf.train.Checkpoint(model=rim)
    #


def train_step(inputs):
    x_true, y = inputs
    x_init = tf.random.normal(x_true.shape)    
    if len(rim.trainable_variables) == 0:
        #Hack since sometimes this si the first time RIM is called and so hasn't been inisitalized
        i = 0
        a, b, c = x_init[i:i+1], y[i:i+1],  x_true[i:i+1]
        try: a, b, c = tf.constant(a), tf.constant(b),  tf.constant(c)
        except: pass
        _  =  rim(a, b, grad_fn, c)[1] / args.batch_size

        
    gradients = [0.]*len(rim.trainable_variables)
    #n = args.sims_in_loop
    for i in range(args.batch_size // world_size):
        with tf.GradientTape() as tape:
            #a, b, c = x_init[n*i:n*i+n], y[n*i:n*i+n],  x_true[n*i:n*i+n]
            a, b, c = x_init[i:i+1], y[i:i+1],  x_true[i:i+1]
            try: a, b, c = tf.constant(a), tf.constant(b),  tf.constant(c)
            except: pass
            loss =  rim(a, b, grad_fn, c)[1] / args.batch_size
        grads = tape.gradient(loss, rim.trainable_variables)
        for j in range(len(grads)):
            gradients[j] = gradients[j] + grads[j] 
    optimizer.apply_gradients(zip(gradients, rim.trainable_variables))
    return loss



def test_step(inputs):
    x_true, y = inputs
    x_init = tf.random.normal(x_true.shape)
    x_pred = rim(x_init, y, grad_fn, x_true)[0]
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
        pred_adam = adam(x_init, yy, grad_fn)
        pred_adam10 = adam10(x_init, yy, grad_fn)
        print(x_init.shape, xx.shape, yy.shape, pred.shape, pred_adam.shape, pred_adam10.shape)

        check_im(xx[0].numpy(), x_init[0].numpy(), pred[0].numpy(), ofolder + 'rim-im-%d.png'%epoch)
        check_2pt(datamodel, xx, yy, x_init, pred, pred_adam, pred_adam10, ofolder + 'rim-2pt-%d.png'%epoch)

        break
    
    rim.save_weights(ofolder + '/%d'%epoch)
    
    

#$@
#$@    print(len(x), x[0].values[0].shape)
#$@        a, b, c, d = distributed_test_step(x)
#$@        print(a.values[0].shape, b.values[0].shape, c.values[0].shape, d.values[0].shape)
#$@        pred, x_init, xx, yy = a.values[0], b.values[0], c.values[0], d.values[0]
#$@        pred = pred[-1]
#$@        pred_adam = adam(x_init, yy, grad_fn)
#$@        pred_adam10 = adam10(x_init, yy, grad_fn)
#$@        
#$@        fig, ax = plt.subplots(1, 3, figsize = (12, 4))
#$@        vmin, vmax = xx[0].numpy().sum(axis=0).min(), xx[0].numpy().sum(axis=0).max()
#$@        ax[0].imshow(xx[0].numpy().sum(axis=0), vmin=vmin, vmax=vmax)
#$@        ax[0].set_title('Truth')
#$@        ax[1].imshow(x_init[0].numpy().sum(axis=0), vmin=vmin, vmax=vmax)
#$@        ax[1].set_title('initial point')
#$@        ax[2].imshow(pred[0].numpy().sum(axis=0), vmin=vmin, vmax=vmax)
#$@        ax[2].set_title('RIM %d step'%(params['rim_iter']))
#$@        plt.savefig(ofolder + 'rim-im-%d.png'%epoch)
#$@        plt.close()
#$@
#$@        ##
#$@        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
#$@        print(x_init.shape, xx.shape, yy.shape, pred.shape, pred_adam.shape, pred_adam10.shape)
#$@        k, pks = get_ps([x_init.numpy(), gal_sample(pm(x_init)).numpy()], [xx.numpy(), yy.numpy()])
#$@        for i in range(2):
#$@            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d--'%i, lw=0.5)
#$@            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d--'%i, lw=0.5)
#$@
#$@        k, pks = get_ps([pred.numpy(), gal_sample(pm(pred)).numpy()], [xx.numpy(), yy.numpy()])
#$@        for i in range(2):
#$@            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d'%i)
#$@            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d'%i)
#$@
#$@        k, pks = get_ps([pred_adam.numpy(), gal_sample(pm(pred_adam)).numpy()], [xx.numpy(), yy.numpy()])
#$@        for i in range(2):
#$@            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d-.'%i, lw=0.5)
#$@            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d-.'%i, lw=0.5)
#$@
#$@        k, pks = get_ps([pred_adam10.numpy(), gal_sample(pm(pred_adam10)).numpy()], [xx.numpy(), yy.numpy()])
#$@        for i in range(2):
#$@            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d:'%i)
#$@            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d:'%i)
#$@
#$@        for axis in ax: 
#$@            axis.semilogx()
#$@            axis.grid(which='both')
#$@        ax[0].set_ylim(-0.1, 1.2)
#$@        ax[1].set_ylim(-0.5, 2.5)
#$@        plt.savefig(ofolder + 'rim-2pt-%d.png'%epoch)
#$@        plt.close()
#$@
#$@        break
#$@

#    
