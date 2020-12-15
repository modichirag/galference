""" Implementation of Cosmic RIM estimator"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
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
parser.add_argument('--nc', type=int, default=16, help='Grid size')
parser.add_argument('--bs', type=float, default=100, help='Box Size')
parser.add_argument('--nsteps', type=int, default=3, help='')
parser.add_argument('--niter', type=int, default=200, help='Number of iterations/Max iterations')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='adam', help='Which optimizer to use')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--nsims', type=int, default=100, help='Batch size')
parser.add_argument('--nbody', type=str2bool, default=True, help='Number of simulationss')
parser.add_argument('--lpt_order', type=int, default=1, help='Order of LPT Initial conditions')
parser.add_argument('--input_size', type=int, default=8, help='Input layer channel size')
parser.add_argument('--cell_size', type=int, default=8, help='Cell channel size')
parser.add_argument('--rim_iter', type=int, default=10, help='Optimization iteration')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--plambda', type=float, default=0.10, help='Poisson probability')


args = parser.parse_args()


nc, bs = args.nc, args.bs
niter = args.niter
optimizer = args.optimizer
lr = args.lr
a0, af, nsteps = 0.1, 1.0,  args.nsteps
stages = np.linspace(a0, af, nsteps, endpoint=True)
plambda = args.plambda
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
params['cell_kernel_size'] = 5
params['input_kernel_size'] = 5
params['output_kernel_size'] = 5
params['rim_iter'] = args.rim_iter
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




    
def get_data(nsims=args.nsims):
    if args.nbody: dpath = '/project/projectdirs/m3058/chmodi/rim-data/poisson_L%04d_N%03d_T%02d_p%03d/'%(bs, nc, nsteps, plambda*100)
    else: dpath = '/project/projectdirs/m3058/chmodi/rim-data/poisson_L%04d_N%03d_LPT%d_p%03d/'%(bs, nc, args.lpt_order, plambda*100)
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
def gal_sample(base):
    galmean = tfp.distributions.Poisson(rate = plambda * (1 + base))
    return galmean.sample()


@tf.function
def recon(linear, data):
    """                                                                                                                                                   
    """
    print('new graph')
    base = pm(linear)

    galmean = tfp.distributions.Poisson(rate = plambda * (1 + base))
    logprob = -tf.reduce_mean(galmean.log_prob(data))
    #logprob = tf.multiply(logprob, 1/nc**3, name='logprob')


    #Prior
    lineark = r2c3d(linear, norm=nc**3)
    priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
    prior = tf.reduce_mean(tf.multiply(priormesh, 1/priorwt))
#     prior = tf.multiply(prior, 1/nc**3, name='prior')
    #                                                                                                                                                     
    loss = logprob + prior

    return loss, logprob, prior


@tf.function
def recon_grad(x, y):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = recon(x, y)[0]
    grad = tape.gradient(loss, x)
    return grad


#strategy = tf.distribute.MirroredStrategy(devices=["/device:GPU:0", "/device:GPU:1"])
strategy = tf.distribute.MirroredStrategy()
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

traindata, testdata = get_data()
print(traindata.shape, testdata.shape)

BUFFER_SIZE = len(traindata)
BATCH_SIZE_PER_REPLICA = params['batch_size']
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = params['epoch']
train_dataset = tf.data.Dataset.from_tensor_slices((traindata[:, 0], traindata[:, 2])).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
test_dataset = tf.data.Dataset.from_tensor_slices((testdata[:, 0], testdata[:, 2])).batch(strategy.num_replicas_in_sync) 

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

# Create a checkpoint directory to store the checkpoints.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


with strategy.scope():
    rim = build_rim(params)
    grad_fn = recon_grad

    def get_opt(lr):
        return  tf.keras.optimizers.Adam(learning_rate=lr)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    checkpoint = tf.train.Checkpoint(model=rim)
    #



def train_step(inputs):
    x_true, y = inputs
    x_init = tf.random.normal(x_true.shape)
    with tf.GradientTape() as tape:
        x_pred, states = rim(x_init, y, grad_fn)
        res  = (x_true - x_pred)
        loss = tf.reduce_mean(tf.square(res)) ##This is not advised, come back to this
    gradients = tape.gradient(loss, rim.trainable_variables)
    #optimizer = get_opt(lr)
    optimizer.apply_gradients(zip(gradients, rim.trainable_variables))
    return loss


def test_step(inputs):
    x_true, y = inputs
    x_init = tf.random.normal(x_true.shape)
    x_pred, _ = rim(x_init, y, grad_fn)
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

suffpath = '_p%03d'%(plambda*100)
if args.nbody: ofolder = './models/poisson_L%04d_N%03d_T%02d%s/'%(bs, nc, nsteps, suffpath)
else: ofolder = './models/poisson_L%04d_N%03d_LPT%d%s/'%(bs, nc, args.lpt_order, suffpath)
try: os.makedirs(ofolder)
except Exception as e: print(e)


for epoch in range(EPOCHS):
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
        a, b, c, d = distributed_test_step(x)
        pred, x_init, xx, yy = a.values[0], b.values[0], c.values[0], d.values[0]
        pred = pred[-1]
        pred_adam = adam(x_init, yy, grad_fn)
        pred_adam10 = adam10(x_init, yy, grad_fn)
        
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
        k, pks = get_ps([x_init.numpy(), gal_sample(pm(x_init)).numpy()], [xx.numpy(), yy.numpy()])
        for i in range(2):
            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d--'%i, lw=0.5)
            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d--'%i, lw=0.5)

        k, pks = get_ps([pred.numpy(), gal_sample(pm(pred)).numpy()], [xx.numpy(), yy.numpy()])
        for i in range(2):
            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d'%i)
            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d'%i)

        k, pks = get_ps([pred_adam.numpy(), gal_sample(pm(pred_adam)).numpy()], [xx.numpy(), yy.numpy()])
        for i in range(2):
            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d-.'%i, lw=0.5)
            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d-.'%i, lw=0.5)

        k, pks = get_ps([pred_adam10.numpy(), gal_sample(pm(pred_adam10)).numpy()], [xx.numpy(), yy.numpy()])
        for i in range(2):
            ax[0].plot(k, pks[i][2]/(pks[i][0]*pks[i][1])**0.5, 'C%d:'%i)
            ax[1].plot(k, (pks[i][0]/pks[i][1])**0.5, 'C%d:'%i)

        for axis in ax: 
            axis.semilogx()
            axis.grid(which='both')
        plt.savefig(ofolder + 'rim-2pt-%d.png'%epoch)
        plt.close()

        break

    rim.save_weights(ofolder + '/%d.png'%epoch)
    
    

#    
