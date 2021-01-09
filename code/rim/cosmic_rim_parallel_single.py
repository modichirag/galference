""" Implementation of Cosmic RIM estimator"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

from convolutional_recurrent import ConvLSTM3DCell
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose, MaxPool3D, AveragePooling3D


import numpy as np
import os, sys, argparse, time
from scipy.interpolate import InterpolatedUnivariateSpline as iuspline
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from rim_utils import myAdam, build_rim_parallel_single
from recon_models import Recon_DM

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
parser.add_argument('--nc', type=int, default=16, help='Grid size')
parser.add_argument('--bs', type=float, default=100, help='Box Size')
parser.add_argument('--nsteps', type=int, default=3, help='')
parser.add_argument('--niter', type=int, default=200, help='Number of iterations/Max iterations')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--optimizer', type=str, default='adam', help='Which optimizer to use')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--nsims', type=int, default=100, help='Number of simulations')
parser.add_argument('--nbody', type=str2bool, default=True, help='Number of simulationss')
parser.add_argument('--lpt_order', type=int, default=1, help='Order of LPT Initial conditions')
parser.add_argument('--input_size', type=int, default=8, help='Input layer channel size')
parser.add_argument('--cell_size', type=int, default=8, help='Cell channel size')
parser.add_argument('--rim_iter', type=int, default=10, help='Optimization iteration')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--suffix', type=str, default='', help='Suffix for folder pathname')
parser.add_argument('--batch_in_epoch', type=int, default=20, help='Number of batches in epochs')
parser.add_argument('--sims_in_loop', type=int, default=2, help='Number of sims in gradient loop')



args = parser.parse_args()


nc, bs = args.nc, args.bs
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
fid_recon = Recon_DM(nc, bs, a0=a0, af=af, nsteps=nsteps, nbody=args.nbody, lpt_order=args.lpt_order, anneal=True)

suffpath = '_parallel' + args.suffix
if args.nbody: ofolder = './models/L%04d_N%03d_T%02d%s/'%(bs, nc, nsteps, suffpath)
else: ofolder = './models/L%04d_N%03d_LPT%d%s/'%(bs, nc, args.lpt_order, suffpath)
try: os.makedirs(ofolder)
except Exception as e: print(e)




def get_data(nsims=args.nsims):
    #if args.nbody: dpath = '/project/projectdirs/m3058/chmodi/rim-data/L%04d_N%03d_T%02d/'%(bs, nc, nsteps)
    #else: dpath = '/project/projectdirs/m3058/chmodi/rim-data/L%04d_N%03d_LPT%d/'%(bs, nc, args.lpt_order)
    if args.nbody: dpath = '../../data/rim-data/L%04d_N%03d_T%02d/'%(bs, nc, nsteps)
    else: dpath = '../../data/rim-data/L%04d_N%03d_LPT%d/'%(bs, nc, args.lpt_order)
    alldata = np.array([np.load(dpath + '%04d.npy'%i) for i in range(nsims)]).astype(np.float32)
    traindata, testdata = alldata[:int(0.9*nsims)], alldata[int(0.9*nsims):]
    return traindata, testdata



@tf.function()
def pm_data(dummy):
    print("PM graph")
    linear = flowpm.linear_field(nc, bs, ipklin, batch_size=args.batch_size)
    if args.nbody:
        print('Nobdy sim')
        state = lpt_init(linear, a0=a0, order=args.lpt_order)
        final_state = nbody(state,  stages, nc)
    else:
        print('ZA/2LPT sim')
        final_state = lpt_init(linear, a0=af, order=args.lpt_order)
    tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
    return linear, tfinal_field

@tf.function()
def pm_data_test(dummy):
    print("PM graph")
    linear = flowpm.linear_field(nc, bs, ipklin, batch_size=world_size)
    if args.nbody:
        print('Nobdy sim')
        state = lpt_init(linear, a0=a0, order=args.lpt_order)
        final_state = nbody(state,  stages, nc)
    else:
        print('ZA/2LPT sim')
        final_state = lpt_init(linear, a0=af, order=args.lpt_order)
    tfinal_field = cic_paint(tf.zeros_like(linear), final_state[0])
    return linear, tfinal_field


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
def recon_dm(linear, data):

    print('new graph')
    final_field = pm(linear)
    residual = final_field - data #.astype(np.float32)
    chisq = tf.multiply(residual, residual)
    chisq = tf.reduce_mean(chisq)                             
    lineark = r2c3d(linear, norm=nc**3)
    priormesh = tf.square(tf.cast(tf.abs(lineark), tf.float32))
    prior = tf.reduce_mean(tf.multiply(priormesh, 1/priorwt))
    loss = chisq + prior

    return loss, chisq, prior


@tf.function
def recon_dm_grad(x, y):
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = recon_dm(x, y)[0]
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



def check_2pt(xx, yy, rim, grad_fn, compares, nrim=10, fname=None):
    truemesh = [xx[0], yy[0]]
    rimpreds = []
    for it in range(nrim):
        x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
        #x_init = (yy - (yy.max() - yy.min())/2.)/yy.std() + np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
        #pred = rim(tf.constant(x_init), tf.constant(yy), grad_fn)[-1]
        pred, _ = rim(tf.constant(x_init), tf.constant(yy), grad_fn, tf.constant(xx))
        rimpreds.append([pred[0].numpy(), pm(pred)[0].numpy()])

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





###
###class RIM3D_parallel(tf.keras.Model):
###
###    def __init__(self, cell1, cell2, input_layer, input_layer_sub, output_layer_up, output_layer, strides, niter):
###        super(RIM3D_parallel, self).__init__()
###        self.cell1 = cell1
###        self.cell2 = cell2
###        self.output_layer = output_layer
###        self.output_layer_up = output_layer_up
###        self.input_layer = input_layer
###        self.input_layer_sub = input_layer_sub
###        self.strides = strides
###        self.niter = niter
###        self.beta_1, self.beta_2 = 0.9, 0.999
###        self.lr, self.eps = 0.1, 1e-7
###            
###            
###    def call(self, x_init, y, grad_fn, x_true, grad_args=[], initstates = None, return_steps=False):
###
###
###        if initstates is None: 
###            #stateshape = tuple(i//self.strides for i in x_init.shape) + tuple([self.cell1.filters])
###            #stateshape = x_init.shape + tuple([self.cell.filters])
###            #initstates = [tf.zeros(stateshape), tf.zeros(stateshape)]
###            nc2 = int(x_init.shape[1]/self.strides)
###            stateshape = (x_init.shape[0], nc2, nc2, nc2, self.cell1.filters)
###            initstates1 = [tf.zeros(stateshape), tf.zeros(stateshape)]
###            stateshape = x_init.shape + tuple([self.cell2.filters])
###            initstates2 = [tf.zeros(stateshape), tf.zeros(stateshape)]
###            initstates = [initstates1, initstates2]
###
###        i = tf.constant(0, dtype=tf.int32)
###        curr_state = initstates
###        curr_pos = x_init        
###        m = tf.zeros_like(x_init)
###        v = tf.zeros_like(x_init)
###        
###        def body(i, pos, states, m, v):  
###            gradient = grad_fn(pos, y, *grad_args)           
###            t = tf.cast(i+1, tf.float32)
###            m = self.beta_1*m + (1-self.beta_1)*gradient
###            v = self.beta_2*v + (1-self.beta_2)*gradient**2
###            mc = m/(1-self.beta_1**t)
###            vc = v/(1-self.beta_2**t)
###            delta = -1.*self.lr*mc/(tf.sqrt(vc) + self.eps)
###            #
###            states1, states2 = states
###            concat_input = tf.stack([pos, delta], axis=-1)
###            #
###            cell_input_sub = self.input_layer_sub(concat_input)
###            delta_pos1, new_states1 = self.cell1(cell_input_sub, states1)
###            delta_pos1 = self.output_layer_up(delta_pos1) 
###            #
###            cell_input = self.input_layer(concat_input)
###            delta_pos2, new_states2 = self.cell2(cell_input, states2)
###            #delta_pos2 = self.output_layer(delta_pos2)
###            #
###            #delta_pos = delta_pos1 + delta_pos2
###            delta_pos = tf.concat([delta_pos1, delta_pos2], axis=-1)
###            delta_pos = self.output_layer(delta_pos)
###            new_pos = pos + delta_pos[..., 0]
###            new_states = [new_states1, new_states2]
###            return i + 1 , new_pos, new_states, m, v
###
###        loss  = 0.
###        while tf.less(i, tf.constant(self.niter)):
###            i, curr_pos, curr_state, m, v =  body(i, curr_pos, curr_state, m, v)
###            loss = loss + tf.reduce_mean(tf.square(x_true - curr_pos))
###        return curr_pos, loss
###
###
###
###
###def build_rim_parallel(params):
###
###    nc = params['nc']
###    input_layer = Conv3D(params['input_size'], kernel_size=params['input_kernel_size'], 
###                         trainable=True, padding='SAME', 
###                         input_shape=(None, nc, nc, nc, 2), activation=params['input_activation'])
###
###    input_layer_sub = Conv3D(params['input_size'], kernel_size=params['input_kernel_size'], 
###                             trainable=True, padding='SAME', strides= [params['strides']]*3,
###                             input_shape=(None, nc, nc, nc, 2), activation=params['input_activation'])
###
###    cell1 = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')
###
###    output_layer_up = Conv3DTranspose(params['cell_size'], kernel_size=params['middle_kernel_size'], 
###                         trainable=True, padding='SAME', strides=[params['strides']]*3, 
###                         activation=params['output_activation'])
###
###    cell2 = ConvLSTM3DCell(params['cell_size'], kernel_size=params['cell_kernel_size'], padding='SAME')
###
###    output_layer = Conv3D(1, kernel_size=params['output_kernel_size'], trainable=True, padding='SAME', 
###                          input_shape=(None, nc, nc, nc, params['cell_size']*2), activation=params['output_activation'])
###   
###    rim = RIM3D_parallel(cell1, cell2, input_layer, input_layer_sub, output_layer_up, output_layer, strides=params['strides'],
###                       niter=params['rim_iter'])
###
###    return rim
###


def main():
    """
    Model function for the CosmicRIM.
    """

    rim = build_rim_parallel_single(params)
    grad_fn = recon_dm_grad
    #

#
#    train_dataset = tf.data.Dataset.range(args.batch_in_epoch)
#    train_dataset = train_dataset.map(pm_data)
#    # dset = dset.apply(tf.data.experimental.unbatch())
#    train_dataset = train_dataset.prefetch(-1)
#    test_dataset = tf.data.Dataset.range(1).map(pm_data_test).prefetch(-1)
#
    traindata, testdata = get_data()
    idx = np.random.randint(0, traindata.shape[0], 1)
    xx, yy = traindata[idx, 0].astype(np.float32), traindata[idx, 1].astype(np.float32), 
    x_init = np.random.normal(size=xx.size).reshape(xx.shape).astype(np.float32)
    x_pred, _ = rim(tf.constant(x_init), tf.constant(yy), grad_fn, tf.constant(xx))
    

    #
    # @tf.function
    def rim_train(x_true, x_init, y):

        gradients = [0.]*len(rim.trainable_variables)
        n = args.sims_in_loop
        for i in range(args.batch_size//n):
            print(i, n*i, n*i+n)
            with tf.GradientTape() as tape:
                a, b, c = x_init[n*i:n*i+n], y[n*i:n*i+n],  x_true[n*i:n*i+n]
                print(a.shape, b.shape, c.shape)
                #loss = loss + rim(x_init[i:i+1], y[i:i+1], grad_fn, x_true[i:i+1])[1]
                loss =  rim(tf.constant(a), tf.constant(b), grad_fn, tf.constant(c))[1]
            grads = tape.gradient(loss, rim.trainable_variables)
            #print(len(grads), type(grads))
            for j in range(len(grads)):
                gradients[j] = gradients[j] + grads[j]
        return loss, gradients


    ##Train and save
    piter, testiter  = 10, 20
    losses = []
    lrs = [0.001, 0.0005, 0.0001]
    liters = [51, 201, 201]
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

                ##check 2pt and comapre to Adam
                #idx = np.random.randint(0, testdata.shape[0], 1)
                #xx, yy = testdata[idx, 0].astype(np.float32), testdata[idx, 1].astype(np.float32), 
                if x_test is None:
                    idx = np.random.randint(0, testdata.shape[0], 1)
                    x_test, y_test = testdata[idx, 0].astype(np.float32), testdata[idx, 1].astype(np.float32), 
                    pred_adam = adam(tf.constant(x_init), tf.constant(y_test), grad_fn)
                    pred_adam = [pred_adam[0].numpy(), pm(pred_adam)[0].numpy()]
                    pred_adam10 = adam10(tf.constant(x_init), tf.constant(y_test), grad_fn)
                    pred_adam10 = [pred_adam10[0].numpy(), pm(pred_adam10)[0].numpy()]
                    minic, minfin = fid_recon.reconstruct(tf.constant(y_test), RRs=[1.0, 0.0], niter=args.rim_iter*10, lr=0.1)
                    compares =  [pred_adam, pred_adam10, [minic[0], minfin[0]]]
                    print('Test set generated')

                x_init = np.random.normal(size=x_test.size).reshape(x_test.shape).astype(np.float32)
                #pred = rim(tf.constant(x_init), tf.constant(y_test), grad_fn)[-1]
                pred, _ = rim(tf.constant(x_init), tf.constant(y_test), grad_fn, tf.constant(x_test))
                check_im(x_test[0], x_init[0], pred.numpy()[0], fname=ofolder + 'rim-im-%04d.png'%trainiter)
                check_2pt(x_test, y_test, rim, grad_fn, compares, fname=ofolder + 'rim-2pt-%04d.png'%trainiter)

                rim.save_weights(ofolder + '/%d'%trainiter)

            trainiter  += 1


if __name__=="__main__":
    main()
