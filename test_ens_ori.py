import os
import torch
import argparse
import numpy as np

from utils import *
from data.dataset import data_loader, data_loader_attacks
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

import mlp

def majority_voting(data_loader, model, mlps_list):
    acc_ = 0.0
    for images, labels in data_loader:
        final_prediction = []
        images = images.cuda()

        # ----------------------------------------
        # 1. VIT PREDICTION (GPU)
        # ----------------------------------------
        vit_output = model(images)
        vit_predictions = torch.argmax(vit_output.detach().cpu(), dim=-1)
        final_prediction.append(vit_predictions)

        # ----------------------------------------
        # 2. PATCHES FOR MLP BLOCKS
        # ----------------------------------------
        x = model.patch_embed(images)
        x_0 = model.pos_drop(x)

        # ----------------------------------------
        # 3. RUN MLPs ONLY ON CPU
        # ----------------------------------------
        i = 0
        for mlp_cpu in mlps_list:
            # Run ViT block (GPU)
            x_0 = model.blocks[i](x_0)

            # Move ViT output to CPU BEFORE MLP
            mlp_input_cpu = x_0.detach().cpu()

            # Run MLP on CPU
            mlp_output = mlp_cpu(mlp_input_cpu)
            mlp_predictions = torch.argmax(mlp_output, dim=-1)

            final_prediction.append(mlp_predictions)

            i += 1

        # ----------------------------------------
        # 4. MAJORITY VOTE
        # ----------------------------------------
        stacked = torch.stack(final_prediction, dim=1)
        preds_major = torch.argmax(
            torch.nn.functional.one_hot(stacked).sum(dim=1),
            dim=-1
        )

        acc = (preds_major == labels).sum().item() / len(labels)
        acc_ += acc

    final_acc = acc_ / len(data_loader)
    print(f"Final Accuracy From Majority Voting = {final_acc*100:.3f}%")
    return final_acc

# ----------------------------
# MAIN (Windows-safe)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Majority Voting')

    parser.add_argument('--images_type', type=str, choices=['clean', 'adversarial'],
                        help='Type of images')

    parser.add_argument('--image_folder_path', type=str,
                        help='Path to root directory of images')

    parser.add_argument('--vit_path', type=str,
                        help='Path to ViT model')

    parser.add_argument('--mlp_path', type=str,
                        help='Path to MLPs folder')

    parser.add_argument('--attack_name', type=str,
                        help='Attack name')

    args = parser.parse_args()

    # Load ViT
    model = torch.load(args.vit_path, map_location="cuda").cuda()
    model.eval()
    print("ViT is loaded!")

    # Load MLPs
    MLPs_list = get_classifiers_list(MLP_path=args.mlp_path)
    print("All MLPs are loaded!")

    # CLEAN images
    if args.images_type == "clean":
        loader_, dataset_ = data_loader(
            root_dir=args.image_folder_path,
            batch_size=15  # 👉 Windows fix
        )
        majority_voting(data_loader=loader_['test'], model=model, mlps_list=MLPs_list)

    # ADVERSARIAL images
    else:
        loader_, dataset_ = data_loader_attacks(
            root_dir=args.image_folder_path,
            attack_name=args.attack_name,
            batch_size=15 # 👉 Windows fix
        )
        majority_voting(data_loader=loader_, model=model, mlps_list=MLPs_list)
