class config_settings:
    # 随机性
    seed = 7240

    # hardware
    gpus = "0"

    # dataset
    dataset_choose = "kitti"  # nyu, kitti
    sparse_density = 0.005
    # tranform
    jitter = 0.2
    random_crop = True
    random_crop_height = 384
    random_crop_width = 512
    # for nyu
    nyu_dataset_root_path = "../data/nyudepth_hdf5"
    nyu_train_csv = "nyudepth_hdf5_train.csv"
    nyu_test_csv = "nyudepth_hdf5_val.csv"
    # for kitti
    kitti_data_folder = "/root/autodl-tmp/KITTI_Depth_Completion"
    kitti_data_folder_rgb = "/root/autodl-tmp/KITTI_Depth_Completion/raw"
    kitti_val_mode = "full" # full, select
    kitti_image_height = 352
    kitti_image_width = 1216

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


settings = config_settings()

if __name__ == "__main__":
    print(settings)
