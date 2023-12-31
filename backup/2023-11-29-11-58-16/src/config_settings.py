class settings:
    # 随机性
    seed = 7240

    # hardware
    gpus = "0"

    # dataset
    nyu_dataset_root_path = "../data/nyudepth_hdf5"
    nyu_train_csv = "nyudepth_hdf5_train.csv"
    nyu_test_csv = "nyudepth_hdf5_val.csv"
    sparse_density = 0.005

    # model
    res_block = "BasicBlock"  # BasicBlock, Bottleneck
    res_channels = [32, 64, 128, 128, 128]
    # res_channels = [64, 128, 256, 256, 256]

    # loss
    max_depth = 10.0
    decay = 0.8
    alpha = 1.0
    beta = 1.0

    # Optimizer
    optimizer = "ADAM"  # SGD, ADAM, RMSprop
    lr = 1e-3
    weight_decay = 1e-5
    # SGD
    momentum = 0.9
    # ADAM
    betas = (0.9, 0.99)
    epsilon = 1e-8

    # Scheduler
    scheduler = "CosineAnnealingLR"  # LambdaLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
    LambdaLR_decay = [3, 5, 7]
    LambdaLR_gamma = [1.0, 0.2, 0.04]
    ReduceLROnPlateau_factor = 0.1
    ReduceLROnPlateau_patience = 5

    # train
    epochs = 10
    batch_size = 4
