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

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)


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
    
#    BATCH_SIZE_PER_REPLICA = 64
#    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
#    train_dataset = tf.data.Dataset.from_tensor_slices((traindata[:, 0], traindata[:, 1])).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
#    test_dataset = tf.data.Dataset.from_tensor_slices((testdata[:, 0], testdata[:, 1])).batch(GLOBAL_BATCH_SIZE) 
#    return train_dataset, test_dataset
#

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
    residual = final_field - data

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





#strategy = tf.distribute.MirroredStrategy(devices=["/device:GPU:0", "/device:GPU:1"])
strategy = tf.distribute.MirroredStrategy()
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

traindata, testdata = get_data()
print(traindata.shape, testdata.shape)

BUFFER_SIZE = len(traindata)
BATCH_SIZE_PER_REPLICA = 16
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = 10
train_dataset = tf.data.Dataset.from_tensor_slices((traindata[:, 0], traindata[:, 1])).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
test_dataset = tf.data.Dataset.from_tensor_slices((testdata[:, 0], testdata[:, 1])).batch(strategy.num_replicas_in_sync) 

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

# Create a checkpoint directory to store the checkpoints.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


with strategy.scope():
    rim = build_rim(params)
    grad_fn = recon_dm_grad
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=rim)
    #

def train_step(inputs):
    x_true, y = inputs
    x_init = tf.random.normal(x_true.shape)
    with tf.GradientTape() as tape:
        x_pred, states = rim(x_init, y, grad_fn)
        res  = (x_true - x_pred)
        loss = tf.reduce_mean(tf.square(res), axis=(1, 2, 3, 4))
    gradients = tape.gradient(loss, rim.trainable_variables)
    optimizer.apply_gradients(zip(gradients, rim.trainable_variables))
    return loss

def test_step(inputs):
    x_true, y = inputs
    x_init = tf.random.normal(x_true.shape)
    x_pred, _ = rim(x_init, y, grad_fn)
    return x_pred, x_init, x_true, y




##with strategy.scope():
##  # Set reduction to `none` so we can do the reduction afterwards and divide by
##  # global batch size.
##  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
##      from_logits=True,
##      reduction=tf.keras.losses.Reduction.NONE)
##  def compute_loss(labels, predictions):
##    per_example_loss = loss_object(labels, predictions)
##    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
##
##
##with strategy.scope():
##  test_loss = tf.keras.metrics.Mean(name='test_loss')
##
##  model = create_model()
##
##  optimizer = tf.keras.optimizers.Adam()
##
##
##
##
##
##def train_step(inputs):
##  images, labels = inputs
##
##  with tf.GradientTape() as tape:
##    predictions = model(images, training=True)
##    loss = compute_loss(labels, predictions)
##
##  gradients = tape.gradient(loss, model.trainable_variables)
##  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
##
##  train_accuracy.update_state(labels, predictions)
##  return loss 
##
##
##def test_step(inputs):
##  images, labels = inputs
##
##  predictions = model(images, training=False)
##  t_loss = loss_object(labels, predictions)
##
##  test_loss.update_state(t_loss)
##  test_accuracy.update_state(labels, predictions)
##



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


#
#counter = 0
#for x in test_dist_dataset:
#    counter += 1
#    print('counter : ', counter)
#    a, b, c, d = distributed_test_step(x)
#    print(len(a.values))
#    print(a.values[0].shape)
#

losses = []    
for epoch in range(EPOCHS):

    #TRAIN LOOP
    total_loss = 0.0
    num_batches = 0
    for x in train_dist_dataset:
        total_loss += distributed_train_step(x)
        print(epoch, num_batches, total_loss)
        num_batches += 1
    train_loss = total_loss / num_batches
    print("Train loss for epoch %d "%epoch, train_loss)
    losses = losses + list(train_loss.numpy())
    plt.plot(losses)
    plt.savefig('losses.png')


#    
##
##  # TEST LOOP
##  for x in test_dist_dataset:
##    distributed_test_step(x)
##
##  if epoch % 2 == 0:
##    checkpoint.save(checkpoint_prefix)
##
##  template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
##              "Test Accuracy: {}")
##  print (template.format(epoch+1, train_loss,
##                         train_accuracy.result()*100, test_loss.result(),
##                         test_accuracy.result()*100))
##
##  test_loss.reset_states()
##  train_accuracy.reset_states()
##  test_accuracy.reset_states()  
##
##
##
##    ##Train and save
##    piter, testiter  = 10, 25
##    losses = []
##    lrs = [0.001, 0.0005, 0.0001]
##    liters = [201, 101, 1001]
##    trainiter = 0 
##    start = time.time()
##    for il in range(1):
##        print('Learning rate = %0.3e'%lrs[il])
##        opt = tf.keras.optimizers.Adam(learning_rate=lrs[il])
##
##        for i in range(liters[il]):
##            idx = np.random.randint(0, traindata.shape[0], args.batch_size)
##            xx, yy = traindata[idx, 0].astype(np.float32), traindata[idx, 1].astype(np.float32), 
##            x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
##
##            loss, gradients = rim_train(x_true=tf.constant(xx), 
##                                    x_init=tf.constant(x_init), 
##                                    y=tf.constant(yy))
##
##            losses.append(loss.numpy())    
##            opt.apply_gradients(zip(gradients, rim.trainable_variables))
##
##            if i%piter == 0: 
##                print("Time taken for %d iterations : "%piter, time.time() - start)
##                print("Loss at iteration %d : "%i, losses[-1])
##                start = time.time()
##            if i%testiter == 0: 
##                plt.plot(losses)
##                plt.savefig('losses.png')
##                plt.close()
##                idx = np.random.randint(0, testdata.shape[0], 1)
##                xx, yy = testdata[idx, 0].astype(np.float32), testdata[idx, 1].astype(np.float32), 
##                x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
##                pred = rim(tf.constant(x_init), tf.constant(yy), grad_fn)[0].numpy()[-1]
##                print(pred.shape, x_init.shape, xx.shape, yy.shape)
##                test_callback(pred, x_init, xx, yy)
##
##            trainiter  += 1
##
##
