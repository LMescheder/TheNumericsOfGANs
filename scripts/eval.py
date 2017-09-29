from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
from pandas.stats.moments import rolling_mean
import glob
from os import path
import logging
from matplotlib import pyplot as plt
import argparse
import yaml
import os

# Arguments
parser = argparse.ArgumentParser(description='Plot results.')
parser.add_argument('config', type=str, help='Location of config file.')
args = parser.parse_args()

# Parse config
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

log_configs = config['log configs']
mean_window = config['mean window']
xaxis = config['xaxis']

rootdir = os.path.dirname(args.config)

# Evaluate logs
fix, ax = plt.subplots(figsize=(15, 10))

for i, config in enumerate(log_configs):
    print('Parsing %d/%d (%s)' % (i+1, len(log_configs), config['label']))

    folder = os.path.join(rootdir, config['folder'])
    eventfiles = glob.glob(path.join(folder,'events.out.tfevents.*'))
    if len(eventfiles) == 0:
        logging.warning('No event file found in %s. Skipping config.' % config['folder'])
        continue
    elif len(eventfiles) > 1:
        logging.warning('Multiple event files found in %s.' % config['folder'])

    event_acc = EventAccumulator(folder)
    event_acc.Reload()

    time_out = pd.DataFrame(event_acc.Scalars('inception_score/Wall_clock_time'))
    inception_out = pd.DataFrame(event_acc.Scalars('inception_score/mean'))

    df = pd.merge(time_out[['step', 'value']], inception_out[['step', 'value']], on='step')
    df.columns = ['step', 'time', 'inception score']
    df['inception score'] = df['inception score'].rolling(window=mean_window, center=False).mean()

    ax.plot(df[xaxis], df['inception score'], label=config['label'])

ax.legend()
plt.savefig(os.path.join(rootdir, 'out.png'), bbox_inches='tight')
plt.show()
