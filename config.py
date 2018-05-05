class Config(object):
    data_path = 'data/' # 文本文件存放路径
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = True
    epoch = 10
    batch_size = 100
    plot_every = 200  # 每200个batch 可视化一次
    # use_env = True  # 是否使用visodm
    env = 'sentiment'  # visdom env
    debug_file = '/tmp/debugp'
    model_path = None  # 预训练模型路径
    model_prefix = 'checkpoints/'  # 模型保存路径

opt = Config()