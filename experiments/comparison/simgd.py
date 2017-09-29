import os
from subprocess import call
from os import path
import sys

# Job id
if len(sys.argv) > 1:
    job_id = int(sys.argv[1])
else:
    print('Missing argument: job_id.')
    quit()

# Executables
executable = 'python'

# Paths
rootdir = '../..'
scriptname = 'run.py'
cwd = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(cwd, 'out/simgd_%d' % job_id)

# Arguments
architecture = ['dcgan4', 'dcgan4_nobn', 'dcgan4_nobn_cf', 'dcgan4_nobn_cf']
gantype = ['standard', 'standard', 'standard', 'JS']

args = [
# Architecture
'--image-size', '375',
'--output-size', '32',
'--c-dim', '3',
'--z-dim', '256',
'--gf-dim', '64',
'--df-dim', '64',
'--reg-param', '10',
'--g-architecture', architecture[job_id],
'--d-architecture', architecture[job_id],
'--gan-type', gantype[job_id],
# Training
'--optimizer', 'simgd',
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
