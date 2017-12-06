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
outdir = os.path.join(cwd, 'out/conopt2')

args = [
# Architecture
'--image-size', '160',
'--output-size', '64',
'--c-dim', '3',
'--z-dim', '256',
'--gf-dim', '128',
'--df-dim', '128',
'--g-architecture', 'resnet_cf',
'--d-architecture', 'resnet_cf',
'--gan-type','standard',
# Training
'--optimizer', 'conopt',
'--nsteps', '1500000',
'--ntest', '1000',
'--learning-rate', '1e-4',
'--reg-param', '10.',
'--batch-size', '128',
'--log-dir', os.path.join(outdir, 'tf_logs'),
'--sample-dir', os.path.join(outdir, 'samples'),
# Data set
'--dataset', 'celebA',
'--data-dir', './data',
'--split', 'train'
]

# Run
my_env = os.environ.copy()
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
