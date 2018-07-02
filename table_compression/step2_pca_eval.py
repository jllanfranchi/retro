'''
Script to apply PCA to tables, save reduced information tables
'''

import numpy as np
import os.path
import sys
from sklearn.decomposition import IncrementalPCA as PCA
from numba import jit
import cPickle as pickle

# load the latest, greates PCA
with open('pca_after_cl79.pkl','rb') as f:
    pca = pickle.load(f)

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser()
parser.add_argument('--cluster-idx', type=int,
                    help='table cluster index (0,79)')
parser.add_argument('--nfs', action='store_true',
                    help='also acces over NFS')
parser.add_argument('--overwrite', action='store_true',
                    help='overwrite existing file')
args = parser.parse_args()

print 'table cluster %s'%(args.cluster_idx)

if os.path.isfile('/data/icecube/retro_tables/tilt_on_anisotropy_on_noazimuth_80/cl%s/pca_reduced_table.npy'%(args.cluster_idx)):
    if args.overwrite:
        print 'overwritting existing file'
    else:
        print 'file exists, abort'
        sys.exit()

# try fles on fastio first
fname = '/fastio/icecube/retro/tables/cl%s/ckv_table.npy'%(args.cluster_idx)
if not os.path.isfile(fname):
    fname = '/fastio2/icecube/retro/tables/cl%s/ckv_table.npy'%(args.cluster_idx)
if not os.path.isfile(fname):
    if args.nfs:
        fname = '/data/icecube/retro_tables/tilt_on_anisotropy_on_noazimuth_80/cl%s/ckv_table.npy'%(args.cluster_idx)
    else:
        sys.exit()
table_5d = np.load(fname)

print 'table loaded'
# create 3d table
table_3d = np.sum(table_5d, axis=(3,4))
print '3d table created'
# reject low stats samples (these contian a lot of specific shapes just due to too low statistics. we don't want to add these shapes to our template library)
mask = table_3d < 1000.
print 'mask created'
# reduced data
print 'creating data...'
# prepare data structure
data = np.empty((np.product(table_5d.shape[:3])-np.sum(mask),np.product(table_5d.shape[3:])), dtype=np.float32)
counter = 0
for i,m in np.ndenumerate(mask):
    if not m:
        data[counter] = table_5d[i].ravel() / table_3d[i]
        counter += 1
del table_3d
del table_5d
print 'data created'
reduced_data = pca.transform(data)
print 'PCA transformed'
np.save('/data/icecube/retro_tables/tilt_on_anisotropy_on_noazimuth_80/cl%s/pca_reduced_table.npy'%(args.cluster_idx), reduced_data)