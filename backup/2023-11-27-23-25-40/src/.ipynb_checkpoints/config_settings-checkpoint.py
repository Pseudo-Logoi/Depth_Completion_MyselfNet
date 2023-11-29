class settings:
    # 随机性
    seed = 7240

    # hardware
    gpus = "0"

    # dataset
    nyu_dataset_root_path = "../data/nyudepth_hdf5"
    nyu_train_csv = "nyudepth_hdf5_train.csv"
    nyu_test_csv = "nyudepth_hdf5_val.csv"
    sparse_density = 0.05

    # model
    model = "base"

    # loss
    max_depth = 80.0
    decay = 0.8
    alpha = 1.0
    beta = 1.0

    # Optimizer
    optimizer = "ADAM"  # SGD, ADAM, RMSprop
    lr = 1e-3
    weight_decay = 1e-4
    # SGD
    momentum = 0.9
    # ADAM
    betas = (0.9, 0.999)
    epsilon = 1e-8

    # Scheduler
    scheduler = "LambdaLR"  # LambdaLR, MultiStepLR, ReduceLROnPlateau
    LambdaLR_decay = [2, 3, 4]
    LambdaLR_gamma = [1.0, 0.2, 0.04]

    # train
    epochs = 100
    batch_size = 6
