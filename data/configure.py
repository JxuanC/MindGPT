kamitani_Tr_Aug = '' # setting candidate images path

DIR_dataset_dir = '' # setting DIR dataset path

DIR_train_subs = {# setting train sub path
    'sub-1':f'{DIR_dataset_dir}/sub-01_perceptionNaturalImageTraining_VC_v2.h5',
    'sub-2':'{DIR_dataset_dir}/sub-02_perceptionNaturalImageTraining_VC_v2.h5',
    'sub-3':f'{DIR_dataset_dir}/sub-03_perceptionNaturalImageTraining_VC_v2.h5'
}
kamitani_sti_trainID = "" # setting DIR sti_trainID

kamitani_sti_testID = "" # setting DIR sti_testID

smallCap_Kamitani_train = "caption/example.json" #setting SMALLCAP Cpation path

DIR_test_subs = {# setting test sub path
    'sub-1':f'{DIR_dataset_dir}/sub-01_perceptionNaturalImageTest_VC_v2.h5',
    'sub-2':'{DIR_dataset_dir}/sub-02_perceptionNaturalImageTest_VC_v2.h5',
    'sub-3':f'{DIR_dataset_dir}/sub-03_perceptionNaturalImageTest_VC_v2.h5'
}

CLIPGPTFEATURE = 'features/imagenet.hdf5' # CLIP-B-32 feature for brain2text

