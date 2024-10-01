import API_multi as api

class V:
    #number of classes
    n_c = 10 #200
    #batch_size
    b_size = 256 #64
    #dataset
    dataset_string ='CIFAR10' #'TinyImageNet'
    #image_dim
    ##CIFAR10, CIFAR100
    image_dim = (3, 32, 32) #(3,64,64)
    # image_dim = (3, 224, 224) ##for ImageNet
    #model
    model_str = 'ResNet'
    #number of layers
    n_l = 110 #18
    ##ignore_last_few_linear layers (for resnet ig_l = 0, for vgg16 ig_l =3)
    ig_l = 0

    ### ckpt_trial number
    ckpt_trial = 1
    ### results_trial number (for convenience)(10*ckpt_trial + 1/2/3..)
    results_trial = 14

    #restore checkpoint path for pretrained weights
    restore_checkpoint_path = '/home/milton/DATA1/project_results/FRANK/base_model/normal/CIFAR10/ResNet110_trial'+str(ckpt_trial)+'/original_Scratch_resnet110/Training_Results_original_last_epoch/last_epoch.ckpt'
    # restore_checkpoint_path = '/home/milton/DATA1/project_results/GFI_AP_input_ckpt/TinyImageNet/ResNet18/original_Scratch_resnet18/Training_Results_original_last_epoch/last_epoch.ckpt'
    # restore_checkpoint_path = '/home/milton/DATA1/project_results/GFI_AP_input_ckpt/CIFAR10/ResNet110/original_Scratch_resnet110/Training_Results_original_last_epoch/last_epoch.ckpt'


    # path to store results
    base_path_results = '/home/milton/DATA1/project_results/FRANK/FRANK_actual_method/normal/CIFAR10_RESNET110/Iterative_pruning_filter_depth_normalization_trial_'+str(results_trial)


    dataset = api.Datasets(dataset_string, b_size)

