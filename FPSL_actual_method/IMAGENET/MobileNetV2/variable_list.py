import API_multi as api

class V:
    #number of classes
    n_c = 1000 #200
    #batch_size
    b_size = 256 #256
    #dataset
    dataset_string ='ImageNet' #'TinyImageNet'
    #image_dim
    ##(3, 32, 32) CIFAR10, CIFAR100
    image_dim = (3, 224, 224) #(3,64,64)
    # image_dim = (3, 224, 224) ##for ImageNet
    #model
    model_str = 'MobileNetV2'
    ##ignore_last_few_linear layers (for resnet ig_l = 0, for vgg16 ig_l =3)
    ig_l = 0

    #restore checkpoint path for pretrained weights
    restore_checkpoint_path = '/home/milton/DATA1/project_results/FRANK/base_model/normal/ImageNet/mobilenetv2/original_Scratch_mobilenetv2/Training_Results_original_last_epoch/last_epoch.ckpt'


    # path to store results
    base_path_results = '/home/milton/DATA1/project_results/FRANK/FRANK_actual_method/normal/ImageNet_MobileNetV2/Iterative_pruning_filter_depth_normalization'

    dataset = api.Datasets(dataset_string, b_size)

