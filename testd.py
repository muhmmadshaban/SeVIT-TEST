# BYESION TESTING 
import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

import mlp
from utils import *
from data.dataset import data_loader, data_loader_attacks

# BAYESIAN CLASSIFIER WITH DROPOUT
class BayesianClassifier(nn.Module):
    """
    MLP classifier with Bayesian Dropout.
    Dropout is active during both training and inference for uncertainty estimation.
    """
    def __init__(self, num_classes=3, in_features=768*196, dropout_p=0.3):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_features, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=2048)
        self.linear3 = nn.Linear(in_features=2048, out_features=128)
        self.linear4 = nn.Linear(in_features=128, out_features=num_classes)
        self.dropout_p = dropout_p

    def forward(self, x):
        x = x.reshape(-1, 196*768)
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.dropout(x, p=self.dropout_p, training=True)  # Bayesian dropout
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.dropout(x, p=self.dropout_p, training=True)
        x = torch.nn.functional.relu(self.linear3(x))
        x = torch.nn.functional.dropout(x, p=self.dropout_p, training=True)
        x = self.linear4(x)
        return x

def convert_to_bayesian_dropout(model, dropout_p=0.3):
    """
    Convert a regular MLP to Bayesian version by adding dropout layers
    that remain active during inference.
    """
    # Create a new BayesianClassifier with same architecture
    bayesian_model = BayesianClassifier(
        num_classes=model.linear4.out_features,
        in_features=model.linear1.in_features,
        dropout_p=dropout_p
    )
    
    # Copy the weights from original model to Bayesian model
    bayesian_model.linear1.weight.data = model.linear1.weight.data.clone()
    bayesian_model.linear1.bias.data = model.linear1.bias.data.clone()
    bayesian_model.linear2.weight.data = model.linear2.weight.data.clone()
    bayesian_model.linear2.bias.data = model.linear2.bias.data.clone()
    bayesian_model.linear3.weight.data = model.linear3.weight.data.clone()
    bayesian_model.linear3.bias.data = model.linear3.bias.data.clone()
    bayesian_model.linear4.weight.data = model.linear4.weight.data.clone()
    bayesian_model.linear4.bias.data = model.linear4.bias.data.clone()
    
    return bayesian_model

def mc_dropout_predict(model, x, T=30):
    """
    Performs T stochastic forward passes through the Bayesian MLP.
    Returns mean prediction and standard deviation (uncertainty).
    """
    model.eval()
    preds = []
    for _ in range(T):
        preds.append(model(x))
    preds = torch.stack(preds)
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0)  # uncertainty estimate
    return mean_pred, std_pred

class EnsembleBayesianModel(nn.Module):
    def __init__(self, vit_model, original_mlps_list, use_bayesian_dropout=True, mc_samples=30):
        super().__init__()
        self.vit = vit_model
        self.original_mlps = original_mlps_list  # Store original models
        self.use_bayesian_dropout = use_bayesian_dropout
        self.mc_samples = mc_samples

    def forward(self, images):
        final_predictions = []

        # -------------------------------
        # 1. VIT prediction
        # -------------------------------
        vit_output = self.vit(images)
        vit_logits = vit_output
        final_predictions.append(vit_logits)

        # -------------------------------
        # 2. Extract patch embeddings
        # -------------------------------
        x = self.vit.patch_embed(images)
        x_0 = self.vit.pos_drop(x)

        # -------------------------------
        # 3. Run block + corresponding MLP with Bayesian approach
        # -------------------------------
        mlp_logits_list = []
        i = 0
        for original_mlp in self.original_mlps:
            # ViT block → GPU
            x_0 = self.vit.blocks[i](x_0)

            # Move ViT output to CPU for MLP
            features_cpu = x_0.detach().cpu()

            if self.use_bayesian_dropout:
                # Convert original MLP to Bayesian version for this forward pass
                bayesian_mlp = convert_to_bayesian_dropout(original_mlp)
                
                # Use MC Dropout for uncertainty estimation
                mlp_mean, mlp_std = mc_dropout_predict(bayesian_mlp, features_cpu, T=self.mc_samples)
                # Move both mean and std to GPU for stacking
                mlp_logits_list.append(mlp_mean.cuda())
            else:
                # Regular inference with original model
                logits_cpu = original_mlp(features_cpu)
                mlp_logits_list.append(logits_cpu.cuda())

            i += 1

        # Stack all model logits
        # shape: num_models × batch × classes
        all_logits = torch.stack(
            [vit_logits] + mlp_logits_list,
            dim=0
        )

        # Majority voting with logits:
        # sum logits of all models → best class wins
        ensemble_logits = torch.sum(all_logits, dim=0)

        return ensemble_logits

def evaluate_ensemble_bayesian(data_loader, model, original_mlps_list, use_bayesian=True, mc_samples=30):
    ensemble_model = EnsembleBayesianModel(model, original_mlps_list, use_bayesian_dropout=use_bayesian, mc_samples=mc_samples)
    ensemble_model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.cuda()
            labels = labels.cuda()
            
            # Get ensemble predictions
            ensemble_logits = ensemble_model(images)
            predictions = torch.argmax(ensemble_logits, dim=-1)
            
            # Calculate accuracy for this batch
            batch_correct = (predictions == labels).sum().item()
            batch_total = labels.size(0)
            
            correct += batch_correct
            total += batch_total
            
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f'Batch {batch_idx}, Batch Accuracy: {batch_correct/batch_total*100:.2f}%')
    
    overall_accuracy = correct / total
    print(f'Final Accuracy From Bayesian Logit Averaging Ensemble = {overall_accuracy*100:.3f}%')
    return overall_accuracy

def load_models():
    # Load ViT model
    vit_path = r"C:\Users\Muhmmad shaban\Downloads\SEViT-main\SEViT-main\models\vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth"
    vit_model = torch.load(vit_path, map_location='cuda')
    vit_model = vit_model.cuda()
    vit_model.eval()
    print("ViT model loaded!")
    
    # Load original MLP models using your existing function
    mlp_dir = r"C:\Users\Muhmmad shaban\Downloads\SEViT-main\SEViT-main\models\mlp"
    original_mlps_list = get_classifiers_list(MLP_path=mlp_dir)
    print(f"All {len(original_mlps_list)} original MLPs are loaded!")
    
    return vit_model, original_mlps_list

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Bayesian Logit Averaging Ensemble')

    parser.add_argument('--images_type', type=str, choices=['clean', 'adversarial'],
                        help='Type of images')

    parser.add_argument('--image_folder_path', type=str,
                        help='Path to root directory of images')

    parser.add_argument('--attack_name', type=str,
                        help='Attack name')
    
    parser.add_argument('--bayesian', action='store_true',
                        help='Use Bayesian dropout for uncertainty estimation')

    parser.add_argument('--mc_samples', type=int, default=30,
                        help='Number of Monte Carlo samples for Bayesian inference')

    args = parser.parse_args()

    # Load models
    model, original_mlps_list = load_models()

    # CLEAN images
    if args.images_type == "clean":
        loader_, dataset_ = data_loader(
            root_dir=args.image_folder_path,
            batch_size=15  # 👉 Windows fix
        )
        evaluate_ensemble_bayesian(
            data_loader=loader_['test'], 
            model=model, 
            original_mlps_list=original_mlps_list,
            use_bayesian=args.bayesian,
            mc_samples=args.mc_samples
        )

    # ADVERSARIAL images
    else:
        loader_, dataset_ = data_loader_attacks(
            root_dir=args.image_folder_path,
            attack_name=args.attack_name,
            batch_size=15 # 👉 Windows fix
        )
        evaluate_ensemble_bayesian(
            data_loader=loader_, 
            model=model, 
            original_mlps_list=original_mlps_list,
            use_bayesian=args.bayesian,
            mc_samples=args.mc_samples
        )