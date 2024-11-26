from .mgpt_lm import MLM

def lm_entry(config):
    return globals()[config['type']](**config['kwargs'])