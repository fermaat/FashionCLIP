import logging as log
import os
import sys

import datetime


def init_logging(dev_log=True, env_name='DEV', service_name='theSeeker', log_dir='logs', log_file_fmt='%Y_%m_%d.log', log_to_file=False):
    """
    Initialize logging tool
    """
    log.basicConfig()

    root_log = log.getLogger()
    root_log.setLevel(log.INFO)
    root_log.handlers = []

    if dev_log:
        fmt = log.Formatter(
            f'%(asctime)s.%(msecs)03d %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        fmt = log.Formatter(
            f'%(asctime)s.%(msecs)03d {env_name} %(levelname)6s {service_name} %(filename)20s:%(lineno)-5s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S'
        )

    # log to stdout
    stdout = log.StreamHandler(stream=sys.stdout)
    stdout.setFormatter(fmt)
    root_log.addHandler(stdout)

    if log_to_file:
        # log to file
        os.makedirs(log_dir, exist_ok=True)
        filename = datetime.datetime.utcnow().strftime(log_file_fmt)
        fout = log.FileHandler(os.path.join(log_dir, filename), mode='a')
        fout.setFormatter(fmt)
        root_log.addHandler(fout)
