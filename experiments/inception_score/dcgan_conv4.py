import os
from subprocess import call
from os import path
import sys

# Executables
executable = 'python'

# Paths
rootdir = '../..'
scriptname = 'run.py'
cwd = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(cwd, 'out/dcgan_conv4')

args = [
# Architecture
'--image-size', '375',
'--output-size', '32',
'--c-dim', '3',
'--z-dim', '256',
'--gf-dim', '64',
'--df-dim', '64',
'--g-architecture', 'dcgan4',
'--d-architecture', 'dcgan4',
'--gan-type','standard',
# Training
'--optimizer', 'altgd',
'--altgd-gsteps', '1',
'--altgd-dsteps', '1',
'--nsteps', '150000',
'--ntest', '1000',
'--learning-rate', '1e-4',
'--batch-size', '64',
'--log-dir', os.path.join(outdir, 'tf_logs'),
'--sample-dir', os.path.join(outdir, 'samples'),
'--is-inception-scores',
'--inception-dir', './inception',
# Data set
'--dataset', 'cifar-10',
'--data-dir', './data',
'--split', 'train'
]

# Run
my_env = os.environ.copy()
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
