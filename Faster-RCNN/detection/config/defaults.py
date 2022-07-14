from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.MASK_ON = False

# ---------------------------------------------------------------------------- #
# Dataset options
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.TRAINS = ()
_C.DATASETS.TESTS = ()
_C.DATASETS.TARGETS = ()

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "vgg16"

# -----------------------------------------------------------------------------
# INPUT options
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.TRANSFORMS_TRAIN = []
_C.INPUT.TRANSFORMS_TEST = []

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.ANCHOR_SIZES = (128, 256, 512)
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1, 2)
_C.MODEL.RPN.ANCHOR_STRIDE = 16
_C.MODEL.RPN.NUM_CHANNELS = 512
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 300
_C.MODEL.RPN.NMS_THRESH = 0.7
_C.MODEL.RPN.MIN_SIZE = 1

# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
_C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
_C.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100

# ---------------------------------------------------------------------------- #
# Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 9
_C.MODEL.ROI_BOX_HEAD.POOL_SPATIAL_SCALE = 1.0 / 16
_C.MODEL.ROI_BOX_HEAD.POOL_RESOLUTION = 7
_C.MODEL.ROI_BOX_HEAD.POOL_TYPE = "pooling"
_C.MODEL.ROI_BOX_HEAD.BOX_PREDICTOR = 'vgg16_predictor'

# ---------------------------------------------------------------------------- #
# Adversarial options
# ---------------------------------------------------------------------------- #
_C.ADV = CN()
_C.ADV.LAMBDA_FROM = 1.0
_C.ADV.LAMBDA_TO = 1.0

_C.ADV.LAYERS = [False, False, True]
_C.ADV.DIS_MODEL = [{'in_channels': 512}]
_C.ADV.LOSS_FUNC = 'focal_loss'  # ['cross_entropy', 'focal_loss']
_C.ADV.FOCAL_LOSS_GAMMA = 5
_C.ADV.LOSS_WEIGHT = 1.0

# ---------------------------------------------------------------------------- #
# ROI_ONE_DIS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_ONE_DIS = CN()
_C.MODEL.ROI_ONE_DIS.DOM = CN()
_C.MODEL.ROI_ONE_DIS.DOM.LOSS_WEIGHT = 1.0
_C.MODEL.ROI_ONE_DIS.DOM.MON = False
_C.MODEL.ROI_ONE_DIS.DOM.DISCRIMINATOR = 'OneFCOSDiscriminator'
_C.MODEL.ROI_ONE_DIS.DOM.NUM_CONVS = 4
_C.MODEL.ROI_ONE_DIS.DOM.GRL_LAMBDA = 0.01

_C.MODEL.ROI_ONE_DIS.CLS = CN()
_C.MODEL.ROI_ONE_DIS.CLS.LOSS_WEIGHT = 1.0
_C.MODEL.ROI_ONE_DIS.CLS.MON = False
_C.MODEL.ROI_ONE_DIS.CLS.DISCRIMINATOR = 'OneFCOSDiscriminator_cc'
_C.MODEL.ROI_ONE_DIS.CLS.NUM_CONVS = 4
_C.MODEL.ROI_ONE_DIS.CLS.GRL_LAMBDA = 0.01
_C.MODEL.ROI_ONE_DIS.CLS.LOSS_DIRECT_W = 1.0
_C.MODEL.ROI_ONE_DIS.CLS.LOSS_GRL_W = 1.0
_C.MODEL.ROI_ONE_DIS.CLS.SAMPLES_THRESH = 0.5
_C.MODEL.ROI_ONE_DIS.CLS.NUM_CLASSES = 1

# ---------------------------------------------------------------------------- #
# ROI_TWO_DIS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_TWO_DIS = CN()
_C.MODEL.ROI_TWO_DIS.DOM = CN()
_C.MODEL.ROI_TWO_DIS.DOM.LOSS_WEIGHT = 1.0
_C.MODEL.ROI_TWO_DIS.DOM.MON = False
_C.MODEL.ROI_TWO_DIS.DOM.DISCRIMINATOR = 'VGG16TwoDiscriminator'
_C.MODEL.ROI_TWO_DIS.DOM.POOL_RESOLUTION = 7
_C.MODEL.ROI_TWO_DIS.DOM.GRL_LAMBDA = 0.01
_C.MODEL.ROI_TWO_DIS.DOM.SAMPLE_WEIGHTS = 1.0

_C.MODEL.ROI_TWO_DIS.CLS = CN()
_C.MODEL.ROI_TWO_DIS.CLS.LOSS_WEIGHT = 1.0
_C.MODEL.ROI_TWO_DIS.CLS.MON = False
_C.MODEL.ROI_TWO_DIS.CLS.DISCRIMINATOR = 'VGG16TwoDiscriminator_cc'
_C.MODEL.ROI_TWO_DIS.CLS.POOL_RESOLUTION = 7
_C.MODEL.ROI_TWO_DIS.CLS.GRL_LAMBDA = 0.01
_C.MODEL.ROI_TWO_DIS.CLS.LOSS_DIRECT_W = 1.0
_C.MODEL.ROI_TWO_DIS.CLS.LOSS_GRL_W = 1.0
_C.MODEL.ROI_TWO_DIS.CLS.SAMPLES_THRESH = 0.5
_C.MODEL.ROI_TWO_DIS.CLS.NUM_CLASSES = 1
_C.MODEL.ROI_TWO_DIS.CLS.SAMPLE_WEIGHTS = 1.0

# ---------------------------------------------------------------------------- #
# GENFEATURE Options
# ---------------------------------------------------------------------------- #
_C.MODEL.GENFEATURE = CN()
_C.MODEL.GENFEATURE.TWOMULTSCALE = True
_C.MODEL.GENFEATURE.LOCAL_GLOBAL_MERGE = True
_C.MODEL.GENFEATURE.FPN_STRIDES = [1.0/16]

# ---------------------------------------------------------------------------- #
# Solver options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 25
_C.SOLVER.STEPS = (16, 22)
_C.SOLVER.LR = 1e-5
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 1

# ---------------------------------------------------------------------------- #
# Test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EVAL_TYPES = ('voc',)

_C.WORK_DIR = "./work_dir"
