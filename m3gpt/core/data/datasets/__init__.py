from .dataset_text2motion_cb_lm import Text2MotionDatasetCB
from .dataset_text2dance_cb_lm import Text2DanceDatasetCB
from .dataset_text2motion_cb_lm_with_constructed_text2dance import Text2MotionDatasetCB_with_Text2Dance

from .dataset_music2dance_cb_lm import Music2DanceDatasetCB

from .dataset_music2dance_cb_lm_finedance import Music2DanceDatasetCB_Finedance
from .dataset_music2dance_cb_lm_aist import Music2DanceDatasetCB_Aistpp

from .dataset_music2dance_cb_lm_da_to_d import Music2DanceDatasetCB_DA2D
from .dataset_music2dance_cb_lm_ad_to_a import Music2DanceDatasetCB_AD2A

from .dataset_motion_dance_cb_lm import MotionDanceDatasetCB

from .dataset_music2text_cb_lm import Music2TextDatasetCB
from .dataset_music2text_cb_lm_v0 import Music2TextDatasetCB_v0

from core.utils import printlog

def dataset_entry(config):
    printlog('config[kwargs]',config['kwargs'])
    return globals()[config['type']](**config['kwargs'])
