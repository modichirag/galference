import numpy as np
import sys

import flowpm.utils as utils
import flowpm.tfpm as tfpm
import flowpm.kernels as kernels

sys.path.append('../../utils/')
import tools
from getbiasparams import getbias
import tensorflow as tf


posdata = True
bs = 1000
nc = 64
ncf = 512
nsteps = 3
nstepsf = 20
numd = 3e-4
num = int(numd*bs**3)
R = 256 

kvsym = tools.fftk([nc, nc, nc], bs, symmetric=True, dtype=np.float32)
#kv = kernels.fftk([nc, nc, nc], symmetric=False, dtype=base.dtype)
# Compute necessary Fourier kernels
kvec = tools.fftk((nc, nc, nc), boxsize=bs, symmetric=False)
kmesh = (sum(k**2 for k in kvec)**0.5).astype(np.float32)

@tf.function
def standardrecon(base, pos, bias, R):

    #base = base.astype(np.float32)
    #pos = pos.astype(base.dtype)
    smwts = tf.exp(tf.multiply(-kmesh**2, R**2))
    basek = utils.r2c3d(base, norm=nc**3)
    basek = tf.multiply(basek, tf.cast(smwts, tf.complex64))
    basesm = utils.c2r3d(basek, norm=nc**3)

    grid = bs/nc*np.indices((nc, nc, nc)).reshape(3, -1).T.astype(np.float32)
    grid = tf.constant(np.expand_dims(grid, 0))
    grid = grid *nc/bs
    pos = pos *nc/bs
        
    mesh = basesm #tf.constant(basesm.astype(np.float32))
    meshk = utils.r2c3d(mesh, norm=nc**3)
    
    DX = tfpm.lpt1(meshk, pos, kvec=kvec)
    DX = tf.multiply(DX, -1/bias)
    pos = tf.add(pos, DX)
    displaced = tf.zeros_like(mesh)
    displaced = utils.cic_paint(displaced, pos, name='displaced')
    
    DXrandom = tfpm.lpt1(meshk, grid, kvec)
    DXrandom = tf.multiply(DXrandom, -1/bias)
    posrandom = tf.add(grid, DXrandom)
    random = tf.zeros_like(mesh)
    random = utils.cic_paint(random, posrandom, name='random')
    return displaced, random




##@tf.function
##def standardinit( base, pos, final, R=8):
##
##    ##
##    print('Initial condition from standard reconstruction')
##    
##    if abs(base.mean()) > 1e-6: 
##        base = (base - base.mean())/base.mean()
##    pfin = tools.power(final, boxsize=bs)[1]
##    ph = tools.power(1+base, boxsize=bs)[1]
##    bias = ((ph[1:5]/pfin[1:5])**0.5).mean()
##    print('Bias = ', bias)
##
##    tfdisplaced, tfrandom = standardrecon(bs, nc, np.expand_dims(base, 0), np.expand_dims(pos, 0), bias, R=R)
##
##    with tf.Session() as sess:
##        sess.run(tf.global_variables_initializer())
##        displaced, random = sess.run([tfdisplaced, tfrandom])
##
##    displaced /= displaced.mean()
##    displaced -= 1
##    random /= random.mean()
##    random -= 1
##    recon = displaced - random
##    return recon
##

def get_data(nsims=1, posdata=True):
    path = '//mnt/ceph/users/cmodi/cosmo4d/z00/'
    path = path + '/L%04d_N%04d_D%04d//'%(bs, nc, numd*1e4)
    alldata = np.array([np.load(path + 'S%04d.npy'%i) for i in range(100, 100+nsims)]).astype(np.float32)
    print(alldata.shape)
    if posdata:  traindata, testdata = alldata[:int(0.9*nsims),  [0,1]], alldata[int(0.9*nsims):,  [0,1]]
    else: traindata, testdata = alldata[:int(0.9*nsims),  [0,2]], alldata[int(0.9*nsims):,  [0,2]]
    return traindata, testdata



def all_sim():

    path = '//mnt/ceph/users/cmodi/cosmo4d/z00/'
    dyn = "%02dstep_B1"%nsteps
    dynf = "%02dstep_B1"%nstepsf
    hpath = path + '/L%04d_N%04d_%s//'%(bs, ncf, dynf)
    path = path + '/L%04d_N%04d_%s//'%(bs, nc, dyn)


    for seed in range(100, 601):
        print(seed)
        ic = tools.readbigfile(path + '/L%04d_N%04d_S%04d_%02dstep/mesh/s/'%(bs, nc, seed, nsteps))
        final = tools.readbigfile(path + '/L%04d_N%04d_S%04d_%02dstep/mesh/d/'%(bs, nc, seed, nsteps))
        hpos = tools.readbigfile(hpath + '/L%04d_N%04d_S%04d_%02dstep/FOF/PeakPosition/'%(bs, ncf, seed, nstepsf))[:num]
        hmassall = tools.readbigfile(hpath + '/L%04d_N%04d_S%04d_%02dstep/FOF/Mass/'%(bs, ncf, seed, nstepsf)).flatten()
        hmass = hmassall[:num]
        hmeshpos = tools.paintcic(hpos, bs, nc)
        hmeshmass = tools.paintcic(hpos, bs, nc, hmass.flatten()*1e10)
        hmeshmass /= hmeshmass.mean()
        hmeshmass -= 1
        hmeshpos /= hmeshpos.mean()
        hmeshpos -= 1
        
        if posdata: 
            data = tf.constant(hmeshpos.reshape(1, nc, nc, nc), dtype=tf.float32)
        else: data = tf.constant(hmeshmass.reshape(1, nc, nc, nc), dtype=tf.float32)
    
        base = hmeshpos
        pfin = tools.power(final, boxsize=bs)[1]
        ph = tools.power(1+base, boxsize=bs)[1]
        bias = ((ph[1:5]/pfin[1:5])**0.5).mean()
        
        tfdisplaced, tfrandom = standardrecon(data, tf.expand_dims(tf.constant(hpos, dtype=tf.float32), 0),
                                              tf.constant(bias, dtype=tf.float32), R=tf.constant(R, dtype=tf.float32))
        
        displaced, random = tfdisplaced.numpy()[0], tfrandom.numpy()[0]
        
        displaced /= displaced.mean()
        displaced -= 1
        random /= random.mean()
        random -= 1
        recon = np.squeeze(displaced - random)
        savepath =  '//mnt/ceph/users/cmodi/cosmo4d/z00/L%04d_N%04d_D%04d//'%(bs, nc, numd*1e4)
        np.save(savepath + 'stdR%d_S%04d'%(R, seed), recon)


        if seed == 100:
            import matplotlib.pyplot as plt
            plt.figure(figsize = (9, 4))
            plt.subplot(131)
            plt.imshow(ic.sum(axis=0))
            plt.subplot(132)
            plt.imshow(data.numpy()[0].sum(axis=0))
            plt.subplot(133)
            plt.imshow(recon.sum(axis=0))
            plt.savefig('tmp.png')
            plt.close()

            print(ic.mean(),  recon.mean())
            k, p1 = tools.power(ic+1, boxsize=bs)
            p2 = tools.power(recon+1, boxsize=bs)[1]
            px = tools.power(ic+1, f2=recon+1, boxsize=bs)[1]
            plt.plot(k, p2/p1)
            plt.plot(k, px/(p1*p2)**0.5, '--')
            plt.semilogx()
            plt.savefig('tmp2.png')
            plt.close()


def main():
    #bs, nc = 400, 64
    #ncf, stepf = nc*4, 40
    numd = 1e-3
    num = int(numd*bs**3)
    seed = 100     


    path = '//mnt/ceph/users/cmodi/cosmo4d/z00/'
    dyn = "%02dstep_B1"%nsteps
    dynf = "%02dstep_B1"%nstepsf
    hpath = path + '/L%04d_N%04d_%s//'%(bs, ncf, dynf)
    path = path + '/L%04d_N%04d_%s//'%(bs, nc, dyn)

    ic = tools.readbigfile(path + '/L%04d_N%04d_S%04d_%02dstep/mesh/s/'%(bs, nc, seed, nsteps))
    final = tools.readbigfile(path + '/L%04d_N%04d_S%04d_%02dstep/mesh/d/'%(bs, nc, seed, nsteps))
    
    hpos = tools.readbigfile(hpath + '/L%04d_N%04d_S%04d_%02dstep/FOF/PeakPosition/'%(bs, ncf, seed, nstepsf))[:num]
    hmassall = tools.readbigfile(hpath + '/L%04d_N%04d_S%04d_%02dstep/FOF/Mass/'%(bs, ncf, seed, nstepsf)).flatten()
    print(hmassall.shape, hmassall.shape[0]/bs**3, hmassall.shape[0]/bs**3 /numd)
    hmass = hmassall[:num]
    print(hmass.shape, hmass.shape[0]/bs**3, hmass.shape[0]/bs**3 /numd)
    hmeshpos = tools.paintcic(hpos, bs, nc)
    hmeshmass = tools.paintcic(hpos, bs, nc, hmass.flatten()*1e10)
    hmeshmass /= hmeshmass.mean()
    hmeshmass -= 1
    hmeshpos /= hmeshpos.mean()
    hmeshpos -= 1

    if posdata: data = tf.constant(hmeshpos.reshape(1, nc, nc, nc), dtype=tf.float32)
    else: data = tf.constant(hmeshmass.reshape(1, nc, nc, nc), dtype=tf.float32)
    
    base = hmeshpos
    #base = (base - base.mean())/base.mean()
    pfin = tools.power(final, boxsize=bs)[1]
    ph = tools.power(1+base, boxsize=bs)[1]
    bias = ((ph[1:5]/pfin[1:5])**0.5).mean()

    tfdisplaced, tfrandom = standardrecon(data, tf.expand_dims(tf.constant(hpos, dtype=tf.float32), 0),
                                          tf.constant(bias, dtype=tf.float32), R=tf.constant(8, dtype=tf.float32))
    
    displaced, random = tfdisplaced.numpy()[0], tfrandom.numpy()[0]

    displaced /= displaced.mean()
    displaced -= 1
    random /= random.mean()
    random -= 1
    recon = np.squeeze(displaced - random)
    print(recon.mean())
    print(displaced.shape, random.shape)

    import matplotlib.pyplot as plt
    plt.figure(figsize = (9, 4))
    plt.subplot(131)
    plt.imshow(ic.sum(axis=0))
    plt.subplot(132)
    plt.imshow(data.numpy()[0].sum(axis=0))
    plt.subplot(133)
    plt.imshow(recon.sum(axis=0))
    plt.savefig('tmp.png')
    plt.close()

    print(ic.mean(),  recon.mean())
    k, p1 = tools.power(ic+1, boxsize=bs)
    p2 = tools.power(recon+1, boxsize=bs)[1]
    px = tools.power(ic+1, f2=recon+1, boxsize=bs)[1]
    plt.plot(k, p2/p1)
    plt.plot(k, px/(p1*p2)**0.5, '--')
    plt.semilogx()
    plt.savefig('tmp2.png')
    plt.close()


    for R in [4, 8, 16, 24, 32, 64, 128, 200, 256]:
        tfdisplaced, tfrandom = standardrecon(data, tf.expand_dims(tf.constant(hpos, dtype=tf.float32), 0),
                                          tf.constant(bias, dtype=tf.float32), R=tf.constant(R, dtype=tf.float32))
    
        displaced, random = tfdisplaced.numpy()[0], tfrandom.numpy()[0]

        displaced /= displaced.mean()
        displaced -= 1
        random /= random.mean()
        random -= 1
        recon = np.squeeze(displaced - random)

        print(ic.mean(),  recon.mean())
        k, p1 = tools.power(ic+1, boxsize=bs)
        p2 = tools.power(recon+1, boxsize=bs)[1]
        px = tools.power(ic+1, f2=recon+1, boxsize=bs)[1]
        #plt.plot(k, p2/p1)
        plt.plot(k, px/(p1*p2)**0.5, '-', label=R)
    plt.semilogx()
    plt.legend()
    plt.semilogx()
    plt.grid(which='both')
    plt.ylim(-0.2, 1.2)
    plt.savefig('stdRcompare.png')
        


if __name__=="__main__":

    #main()
    all_sim()
