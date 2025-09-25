import torch
import torch.nn as nn
from dataset import get_data_transforms
import numpy as np
from torch.utils.data import DataLoader
from resnet import wide_resnet50_2
from de_resnet import de_wide_resnet50_2
from dataset import TestDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from vmamba import VSSBlock
from natsort import natsorted
from diffusion import Unet
from epsilon import GaussianDiffusion
import os

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
   
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    
    for i in range(len(ft_list)):
        
        fs = fs_list[i]
        ft = ft_list[i]   

        a_map = 1 - F.cosine_similarity(fs, ft) 
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    
    
    return anomaly_map, a_map_list

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    img_size= 224
    data_transform, gt_transform = get_data_transforms(img_size, img_size)
    
    root = './class type/' 
    first_stage_path = './muweights/'
    diffusion_path = './dmweights/'
    
    test_data = TestDataset(root=root, transform=data_transform, gt_transform=gt_transform)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    encoder_mu = nn.Sequential(nn.Linear(2048*7*7,2048), nn.ReLU())
    encoder_mu = encoder_mu.to(device)
    encoder_var = nn.Sequential(nn.Linear(2048*7*7,2048), nn.ReLU())
    encoder_var = encoder_var.to(device)
    fc1 = nn.Sequential(nn.Linear(2048,2048*7*7), nn.ReLU())
    fc1 = fc1.to(device)
   
    dist_ckp = torch.load(first_stage_path)
    for k, v in list(dist_ckp['bn'].items()):
        if 'memory' in k:
           dist_ckp['bn'].pop(k)

    decoder.load_state_dict(dist_ckp['decoder'])
    bn.load_state_dict(dist_ckp['bn'])
    encoder_mu.load_state_dict(dist_ckp['encoder_mu'])
    encoder_var.load_state_dict(dist_ckp['encoder_var'])
    fc1.load_state_dict(dist_ckp['fc1'])
    bn.eval()
    decoder.eval()
    encoder_mu.eval()
    encoder_var.eval()
    fc1.eval()
    
    diffusion = Unet(
            image_size= 56,
            in_channels= 32,
            model_channels= 128,
            out_channels= 32,
            attention_resolutions= 8,
            channel_mult=(1,2,3,4)
    ) 
    diffusion = diffusion.to(device)
    
    #TBSSM
    vss1_forward = VSSBlock(
                    hidden_dim = 256,
                    norm_layer = nn.LayerNorm,
                    attn_drop_rate = 0.3,
                    d_state = 16,
                    )
    vss1_forward = vss1_forward.to(device)
    vss2_forward = VSSBlock(
                   hidden_dim = 512,
                   norm_layer = nn.LayerNorm,
                   attn_drop_rate = 0.3,
                   d_state = 16, 
                   )
    vss2_forward = vss2_forward.to(device)
    vss3_forward = VSSBlock(
                   hidden_dim = 1024,
                   norm_layer = nn.LayerNorm,
                   attn_drop_rate = 0.3,
                   d_state = 16, 
                   )
    vss3_forward = vss3_forward.to(device)
 
    diffusion_mu = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
    diffusion_mu = diffusion_mu.to(device)
    diffusion_var = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
    diffusion_var = diffusion_var.to(device)
    concate = nn.Sequential(nn.Linear(512+1024+2048, 512), nn.ReLU())
    concate = concate.to(device)
   
    diffusion_ckp = torch.load(diffusion_path)
    diffusion.load_state_dict(diffusion_ckp['diffusion'])
    concate.load_state_dict(diffusion_ckp['concate'])
    vss1_forward.load_state_dict(diffusion_ckp['vss1_forward'])
    vss2_forward.load_state_dict(diffusion_ckp['vss2_forward'])
    vss3_forward.load_state_dict(diffusion_ckp['vss3_forward'])
    diffusion_mu.load_state_dict(diffusion_ckp['diffusion_mu'])

    auc_sp, ap_sp = evaluation(encoder, bn, decoder, encoder_mu, fc1, diffusion, vss1_forward, vss2_forward, vss3_forward, diffusion_mu, concate, test_dataloader, device)

    return auc_sp, ap_sp

def evaluation(encoder, bn, decoder, encoder_mu, fc1, diffusion, vss1_forward, vss2_forward, vss3_forward, diffusion_mu, concate, test_dataloader, device):
  
    diffusion.eval()
    vss1_forward.eval()
    vss2_forward.eval()
    vss3_forward.eval()
    diffusion_mu.eval()
    concate.eval()

    sample = GaussianDiffusion()
    gt_list_sp = []
    pr_list_sp = []
    n_test= 3715

    with torch.no_grad():
        with tqdm(total=n_test, desc=f'Testing', unit='img') as pbar:
            for img, gt, _, _ in test_dataloader:

                img = img.to(device)
                inputs = encoder(img)
                middle = bn(inputs)
                B,C,H,W = middle.shape
                en = middle.view(middle.size(0), -1)
                mu = encoder_mu(en)
                de = fc1(mu)
                de = de.view(B,C,H,W)

                diffusion_input = de.view(-1, 32, 56, 56)
                timestep = torch.full((img.shape[0],), 250, device=device).long()
        
                diffusion_output = sample.ddim_sample_loop(diffusion, diffusion_input.shape, timestep, diffusion_input, inputs, concate, diffusion_mu, vss1_forward, vss2_forward, vss3_forward)
                diffusion_output = diffusion_output.view(-1, 2048, 7, 7)
                outputs = decoder(diffusion_output)

                anomaly_map, a_map_list = cal_anomaly_map(outputs, inputs, img.shape[-1], amap_mode='mul')
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                gt[gt > 0] = 1
                gt[gt <= 0] = 0
          
                gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
                pr_list_sp.append(np.max(anomaly_map))
                pbar.update(img.shape[0])

    auc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 3)
    print('Image Auroc:{:.3f}, Image AP:{:.3f}'.format(auc_sp, ap_sp)) 
       
    return auc_sp, ap_sp


if __name__ == '__main__':

    test()
