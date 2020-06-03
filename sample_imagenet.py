# -*- coding: utf-8 -*-
import numpy as np
import os
import sys

root_dir = '/cache/datasets'

trn_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')

trn_sub = os.path.join(root_dir, 'train_sub100')
val_sub = os.path.join(root_dir, 'val_sub100')

os.makedirs(trn_sub, exist_ok=True)
os.makedirs(val_sub, exist_ok=True)

classes = os.listdir(trn_dir)
classes = [cname for cname in classes if os.path.isdir(os.path.join(trn_dir, cname))]
assert len(classes) == 1000
ids = np.random.choice(len(classes), 100, replace=False)
print('sampled 100 sub-classes')

for i, ii in enumerate(ids):
    cname = classes[ii]
    ## trn
    cur_dir = os.path.join(trn_dir, cname)
    files = os.listdir(cur_dir)
    files = [os.path.join(cur_dir, f) for f in files]
    files = [f for f in files if os.path.isfile(f)]
    cur_ids = np.random.choice(len(files), 800, replace=False) if len(files) > 800 else list(range(len(files)))
    new_dir = os.path.join(trn_sub, cname)
    os.makedirs(new_dir, exist_ok=True)
    for ci in cur_ids:
        os.system('cp {} {}'.format(files[ci], os.path.join(new_dir, files[ci].split('/')[-1])))
    ## val
    os.system('cp -r {} {}'.format(os.path.join(val_dir, cname), os.path.join(val_sub, cname)))
    print('finish class {}, {}, {}'.format(i, ii, cname))
