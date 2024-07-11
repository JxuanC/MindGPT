import glob
import pandas as pd
import faiss
import h5py
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import data.configure as config
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPFeatureExtractor, CLIPVisionModel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def extract_visual_feature(images_dir, batch_size, index_type, encoder_name, save_name, faiss_index = False):
    last_hidden_state = []
    class_embedding = []
    h5py_file = h5py.File('./features/{}.hdf5'.format(save_name), 'w')
    feature_extractor = CLIPFeatureExtractor.from_pretrained(encoder_name) 
    clip_encoder = CLIPVisionModel.from_pretrained(encoder_name).to(DEVICE)
    for idx in tqdm(range(0, len(images_dir), batch_size)):
        imgids = images_dir[idx:idx + batch_size]
        images = [Image.open(file_name).convert("RGB") for file_name in imgids]
        with torch.no_grad():
            pixel_values = feature_extractor(images, return_tensors='pt').pixel_values.to(DEVICE)
            encodings = clip_encoder(pixel_values=pixel_values).last_hidden_state.cpu().numpy()
            #last_hidden_state.append(encodings)
            class_embedding.append(encodings[:, 0, :])
        for imgid, encoding in zip(imgids, encodings):
            h5py_file.create_dataset('n' + str(imgid).split('n')[-1], (50, 768), data = encoding)

    if(faiss_index):
        class_embedding = np.vstack(class_embedding)
        embedding_dimension = class_embedding.shape[1]
        embedding_nums = class_embedding.shape[0]

        index_type = 'dot'
        if index_type == "L2":
            cpu_index = faiss.IndexFlatL2(embedding_dimension) 
            #gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        if index_type == "dot":
            cpu_index = faiss.IndexFlatIP(embedding_dimension) 
            #gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        if index_type == "cosine":
            # cosine = normalize & dot
            faiss.normalize_L2(class_embedding)
            cpu_index = faiss.IndexFlatIP(embedding_dimension)
            #gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

        print(cpu_index.is_trained)  
        cpu_index.add(class_embedding)
        faiss.write_index(cpu_index, f"data/features/CLIP_{save_name}_index")
        #faiss.write_index(gpu_index, f"database/GPU_{save_name}")
    
encoder_name = 'openai/clip-vit-base-patch32'

imgs_dir = np.concatenate([np.array(glob.glob(f"{config.kamitani_Tr_Aug}/*/*.JPEG"))])

extract_visual_feature(imgs_dir, 128, "dot", encoder_name, 'imagenet')
