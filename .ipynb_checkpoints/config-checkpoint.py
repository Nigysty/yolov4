from easydict import EasyDict
Cfg = EasyDict()


Cfg.batch = 64  # 根据自身情况做更改
Cfg.subdivisions = 16 # 根据自身情况做更改 好的机器可降低
Cfg.width = 608
Cfg.height = 608
Cfg.momentum = 0.949
Cfg.decay = 0.0005
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1
Cfg.jitter = 0.3
Cfg.mosaic = True  

Cfg.learning_rate = 0.001
Cfg.burn_in = 500  # 根据自身情况做更改 ，起保护权重作用
Cfg.max_batches = 8000
Cfg.steps = [4000, 6000]
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1

Cfg.classes = 2  # 根据自身情况做更改
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height

Cfg.cosine_lr = False
Cfg.smoooth_label = False  # 如果数据集不好，存在错标漏标，建议变为True
Cfg.TRAIN_OPTIMIZER = 'adam'