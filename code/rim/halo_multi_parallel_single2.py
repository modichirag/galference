""" Implementation of Cosmic RIM estimator"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
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

from rim_utils import build_rim_parallel_single, myAdam
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
parser.add_argument('--rim_iter', type=int, default=10, help='Optimization iteration')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--suffix', type=str, default='', help='Suffix for folder pathname')
parser.add_argument('--batch_in_epoch', type=int, default=20, help='Number of batches in epochs')
parser.add_argument('--posdata', type=str2bool, default=True, help='Position data')
parser.add_argument('--parallel', type=str2bool, default=True, help='Parallel')
parser.add_argument('--sims_in_loop', type=int, default=1, help='Number of sims ion loop')


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
#RRs = [2, 1, 0.5, 0]

#
klin = np.loadtxt('../../data/Planck15_a1p00.txt').T[0]
plin = np.loadtxt('../../data//Planck15_a1p00.txt').T[1]
ipklin = iuspline(klin, plin)
# Compute necessary Fourier kernels                                                                                                                          
kvec = tools.fftk((nc, nc, nc), boxsize=bs, symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)
priorwt = ipklin(kmesh)


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




    
def get_data(nsims=args.nsims, posdata=True):

    path = '//mnt/ceph/users/cmodi/cosmo4d/z00/'
    path = '/project/projectdirs/m3058/chmodi/rim-data/halos/z00/'
    path = path + '/L%04d_N%04d_D%04d//'%(bs, nc, numd*1e4)
    #if args.nbody: dpath = '/project/projectdirs/m3058/chmodi/rim-data/L%04d_N%03d_T%02d/'%(bs, nc, nsteps)
    #else: dpath = '/project/projectdirs/m3058/chmodi/rim-data/L%04d_N%03d_LPT%d/'%(bs, nc, args.lpt_order)
    alldata = np.array([np.load(path + 'S%04d.npy'%i) for i in range(100, 100+nsims)]).astype(np.float32)
    print(alldata.shape)
    if args.posdata:  traindata, testdata = alldata[:int(0.9*nsims),  [0,1]], alldata[int(0.9*nsims):,  [0,1]]
    else: traindata, testdata = alldata[:int(0.9*nsims),  [0,2]], alldata[int(0.9*nsims):,  [0,2]]
    return traindata, testdata


def setupbias(nsims = 10, cutoff=1.5):

    b1, b2, perr = [], [], []
    for i in range(nsims):
        idx = np.random.randint(0, traindata.shape[0], 1)
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
    prior = tf.reduce_mean(tf.multiply(priormesh, 1/priorwt))

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



#strategy = tf.distribute.MirroredStrategy(devices=["/device:GPU:0", "/device:GPU:1"])
strategy = tf.distribute.MirroredStrategy()
print ('\nNumber of devices: {}\n'.format(strategy.num_replicas_in_sync))

traindata, testdata = get_data()
print(traindata.shape, testdata.shape)

BUFFER_SIZE = len(traindata)
BATCH_SIZE_PER_REPLICA = params['batch_size']
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = params['epoch']
#train_dataset = tf.data.Dataset.from_tensor_slices((traindata[:, 0], traindata[:, 1])).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
train_dataset = tf.data.Dataset.from_tensor_slices((np.arange(traindata.shape[0]), np.arange(traindata.shape[0]))).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
test_dataset = tf.data.Dataset.from_tensor_slices((testdata[:, 0], testdata[:, 1])).batch(strategy.num_replicas_in_sync) 

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

# Create a checkpoint directory to store the checkpoints.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


with strategy.scope():
    if args.parallel: rim = build_rim_parallel_single(params)
    else: rim = build_rim_split(params)
    grad_fn = recon_grad
    b1, b2, errormesh = setupbias()
    bias = tf.constant([b1, b2], dtype=tf.float32)
    print(bias)
    grad_params = [bias, errormesh]

    def get_opt(lr):
        return  tf.keras.optimizers.Adam(learning_rate=lr)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    checkpoint = tf.train.Checkpoint(model=rim)
    #




def train_step():
    n = args.sims_in_loop
    if len(rim.trainable_variables) == 0:
        idx = np.random.randint(0, traindata.shape[0], 1)
        xx, yy = traindata[idx, 0].astype(np.float32), traindata[idx, 1].astype(np.float32), 
        x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
        a, b, c = x_init, yy,  xx
        _s =  rim(a, b, grad_fn, c, grad_params)[1]
        print("First rim call in trainstep : "i, a.shape, b.shape, c.shape)
        
    gradients = [0.]*len(rim.trainable_variables)
    getgrads = True
    #print(len(gradients))
    for i in range(args.batch_size//n):
        idx = np.random.randint(0, traindata.shape[0], n)
        print("%d traindata samples : "%i,, idx)
        xx, yy = traindata[idx, 0].astype(np.float32), traindata[idx, 1].astype(np.float32), 
        x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
        a, b, c = x_init, yy,  xx
        if a.shape[0] > 0:
            with tf.GradientTape() as tape:
                loss =  rim(a, b, grad_fn, c, grad_params)[1]
        else: 
            getgrads = False
            continue
        if getgrads : 
            grads = tape.gradient(loss, rim.trainable_variables)
            for j in range(len(grads)): gradients[j] = gradients[j] + grads[j] / (args.batch_size//n)
    #print('looped')
    if getgrads : optimizer.apply_gradients(zip(gradients, rim.trainable_variables))
    #print('optimized')
    return loss


def test_step(inputs):
    x_true, y = inputs
    x_init = tf.random.normal(x_true.shape)
    x_pred, _ = rim(x_init, y, grad_fn, x_true, grad_params)
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


###
#Training


losses = []    


adam = myAdam(params['rim_iter'])
adam10 = myAdam(10*params['rim_iter'])

if args.parallel: suffpath = '_halo_parallel_single' + args.suffix
else: suffpath = '_halo_split' + args.suffix
if args.nbody: ofolder = './models/L%04d_N%03d_T%02d%s/'%(bs, nc, nsteps, suffpath)
else: ofolder = './models/L%04d_N%03d_LPT%d%s/'%(bs, nc, args.lpt_order, suffpath)
try: os.makedirs(ofolder)
except Exception as e: print(e)


##for x in test_dist_dataset:
##    print('Testing')
##    print(len(x), x[0].values[0].shape)
##    a, b, c, d = distributed_test_step(x)
##    print(a.values[0].shape, b.values[0].shape, c.values[0].shape, d.values[0].shape)
##    pred, x_init, xx, yy = a.values[0], b.values[0], c.values[0], d.values[0]
##    #pred = pred[-1]
##    pred_adam = adam(x_init, yy, grad_fn, grad_params)
##    pred_adam10 = adam10(x_init, yy, grad_fn, grad_params)
##    print(pred.shape, pred_adam.shape, pred_adam10.shape)
##    break
##

for epoch in range(EPOCHS):
    print("\nFor epoch %d\n"%epoch)
    #TRAIN LOOP
    total_loss = 0.0
    num_batches = 0
    starte = time.time()
    for x in train_dist_dataset:
        print(len(x), x[0].values[0].shape)
        startb = time.time()
        loss = distributed_train_step()
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
    for x in test_dist_dataset:
        print('Testing')
        print(len(x), x[0].values[0].shape)
        a, b, c, d = distributed_test_step(x)
        print(a.values[0].shape, b.values[0].shape, c.values[0].shape, d.values[0].shape)
        pred, x_init, xx, yy = a.values[0], b.values[0], c.values[0], d.values[0]
        #pred = pred[-1]
        pred_adam = adam(x_init, yy, grad_fn, grad_params)
        pred_adam10 = adam10(x_init, yy, grad_fn, grad_params)
        
        fig, ax = plt.subplots(1, 3, figsize = (12, 4))
        vmin, vmax = xx[0].numpy().sum(axis=0).min(), xx[0].numpy().sum(axis=0).max()
        ax[0].imshow(xx[0].numpy().sum(axis=0), vmin=vmin, vmax=vmax)
        ax[0].set_title('Truth')
        ax[1].imshow(x_init[0].numpy().sum(axis=0), vmin=vmin, vmax=vmax)
        ax[1].set_title('initial point')
        ax[2].imshow(pred[0].numpy().sum(axis=0), vmin=vmin, vmax=vmax)
        ax[2].set_title('RIM %d step'%(params['rim_iter']))
        plt.savefig(ofolder + 'rim-im-%d.png'%epoch)
        plt.close()

        ##
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        print(x_init.shape, xx.shape, yy.shape, pred.shape, pred_adam.shape, pred_adam10.shape)
        k, pks = get_ps([x_init.numpy(), biasfield(pmpos(x_init)[0], bias).numpy()], [xx.numpy(), yy.numpy()])
        for i in range(2):
            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d--'%i, lw=0.5)
            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d--'%i, lw=0.5)

        k, pks = get_ps([pred.numpy(), biasfield(pmpos(pred)[0], bias).numpy()], [xx.numpy(), yy.numpy()])
        for i in range(2):
            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d'%i)
            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d'%i)

        k, pks = get_ps([pred_adam.numpy(), biasfield(pmpos(pred_adam)[0], bias).numpy()], [xx.numpy(), yy.numpy()])
        for i in range(2):
            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d-.'%i, lw=0.5)
            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d-.'%i, lw=0.5)

        k, pks = get_ps([pred_adam10.numpy(), biasfield(pmpos(pred_adam10)[0], bias).numpy()], [xx.numpy(), yy.numpy()])
        for i in range(2):
            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d:'%i)
            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d:'%i)

        for axis in ax: 
            axis.semilogx()
            axis.grid(which='both')
        ax[0].set_ylim(-0.1, 1.2)
        ax[1].set_ylim(-0.5, 2.5)
        plt.savefig(ofolder + 'rim-2pt-%d.png'%epoch)
        plt.close()

        break

    rim.save_weights(ofolder + '/%d.png'%epoch)
    
    

#    
