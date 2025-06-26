import os
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder
import cv2

def resize_np_array(np_array, target_size):    
    resized_list = []
    for i in range(np_array.shape[0]):
        # Resize each channel using OpenCV
        resized = cv2.resize(np_array[i], (target_size[2], target_size[1]), interpolation=cv2.INTER_LINEAR)
        
        # If the resized size is different from the target size, add padding.
        if resized.shape != target_size[1:]:
            pad_height = max(0, target_size[1] - resized.shape[0])
            pad_width = max(0, target_size[2] - resized.shape[1])
            resized = cv2.copyMakeBorder(resized, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=0)        
        resized_list.append(resized)    
    # Resized array on stack
    return np.stack(resized_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 512],
                    )
    parser.add_argument('--my_height', type=int, default=0)
    parser.add_argument('--my_width', type=int, default=0)
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    my_height = args.my_height
    my_width = args.my_width
    # ckpt_path = f"ckpt/{dataset_name}/best_ckpt.pth"
    ckpt_path = f"ckpts/{dataset_name}/best_ckpt.pth"

    data_dir = f"{dataset_path}/language_features"
    output_dir = f"{dataset_path}/language_features_dim3"
    os.makedirs(output_dir, exist_ok=True)
    
    # copy the segmentation map
    for fk, filename in enumerate(os.listdir(data_dir)):
        print("\rloop:{}".format(fk), end="")
        if filename.endswith("_s.npy"):
            source_path = os.path.join(data_dir, filename)
            target_path = os.path.join(output_dir, filename)

            if my_height != 0 or my_width !=0:
                seg_map_np = np.load(source_path)
                seg_dims, seg_h, seg_w = seg_map_np.shape
                seg_map_np = np.stack([cv2.resize(seg_map_np[i], (my_width, my_height), interpolation=cv2.INTER_NEAREST) for i in range(seg_dims)])
                np.save(target_path, seg_map_np)
            else:
                shutil.copy(source_path, target_path)


    checkpoint = torch.load(ckpt_path)
    train_dataset = Autoencoder_dataset(data_dir)

    test_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=256,
        shuffle=False, 
        num_workers=16, 
        drop_last=False   
    )


    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    for idx, feature in tqdm(enumerate(test_loader)):
        data = feature.to("cuda:0")
        with torch.no_grad():
            outputs = model.encode(data).to("cpu").numpy()  
        if idx == 0:
            features = outputs
        else:
            features = np.concatenate([features, outputs], axis=0)

    os.makedirs(output_dir, exist_ok=True)
    start = 0
    
    for k,v in train_dataset.data_dic.items():
        path = os.path.join(output_dir, k)
        np.save(path, features[start:start+v])
        start += v
