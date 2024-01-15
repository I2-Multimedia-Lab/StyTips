import os
import argparse


import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader
import models.stytips as stytips 
from models.swin_transformer import SwinTransformer
from dataset_sampler import SimpleDataset, InfiniteSamplerWrapper



"""Parameters that needs attention
1.epoch           : How many iterations this training has
2.epoch_start        : From which iteration to start the training
3.checkpoint_save_interval : The interval to save checkpoints
4.loss_count_interval    : The interval to calculate the average loss
"""

def train_transform():
    transform_list = [
        transforms.Resize(size=(256, 256)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, default='/mnt/hdd/huying/hy/Swin-Transformer-main/Style_Transfer_224/COCO_224/Train',
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, default='/mnt/hdd/huying/hy/Swin-Transformer-main/Style_Transfer_224/Images/Train',
                    help='Directory path to a batch of style images')
parser.add_argument('--encoder_dir',type=str,default='./pre_trained_models/swin_tiny_patch4_window7_224.pth')
parser.add_argument('--vgg_dir', type=str, default='/mnt/hdd/huying/hy/stc1/pre_trained_models/vgg_normalised.pth')

# training options
parser.add_argument('--base_lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epoch', type=int, default=50000)

parser.add_argument('--content_weight', type=float, default=2)
parser.add_argument('--style_weight', type=float, default=3)
parser.add_argument('--id1_weight', type=float, default=50)
parser.add_argument('--id2_weight', type=float, default=1)


# save and count options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--checkpoint_save_interval', type=int, default=10000)
parser.add_argument('--loss_count_interval', type=int, default=400)
parser.add_argument('--resume_train', type=bool, default=False, help='Use checkpoints to train or not ')
parser.add_argument('--checkpoint_save_path', type=str, default='./pre_trained_models/norm',
                    help='Directory path to save a checkpoint')
parser.add_argument('--checkpoint_import_path', type=str, default='./pre_trained_model/11/checkpoint_40000_epoch.pkl',
                    help='Directory path to the importing checkpoint')

args = parser.parse_args()

# Print args
print('Running args: ')
for k, v in sorted(vars(args).items()):
    print(k, '=', v)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    
epoch_start = 0
loss_count_interval = args.loss_count_interval

content_tf = train_transform()
style_tf = train_transform()

# Datasets
dataset_content = SimpleDataset(args.content_dir, content_tf)
dataset_style = SimpleDataset(args.style_dir, style_tf)
sampler_content = InfiniteSamplerWrapper(dataset_content)
sampler_style = InfiniteSamplerWrapper(dataset_style)
content_iter = iter(DataLoader(dataset_content,
                      batch_size=args.batch_size,
                      sampler=sampler_content,
                      num_workers=0))
style_iter = iter(DataLoader(dataset_style,
                      batch_size=args.batch_size,
                      sampler=sampler_style,
                      num_workers=0))


vgg = stytips.vgg
# Hardware Setting
print(torch.cuda.is_available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
   
# Models
vgg.load_state_dict(torch.load(args.vgg_dir))

encoder = SwinTransformer(
  img_size=256,
  patch_size=4,
  in_chans=3,
  embed_dim=192,
  depths=[2, 2, 6, 2],
  nhead=[3, 6, 12, 24],
  window_size=8,
  mlp_ratio=4.,
  qkv_bias=True,
  qk_scale=None,
  drop_rate=0.,
  attn_drop_rate=0.,
  drop_path_rate=0.1,
  norm_layer=nn.LayerNorm,
  ape=False,
  patch_norm=True,
  use_checkpoint=False,
  fused_window_process=False,
)


decoder = stytips.decoder
transformer = stytips.Transformer()
network = stytips.Net(encoder, decoder, transformer, vgg)

# Optimizer
optimizer = torch.optim.Adam([
    {'params': network.encoder.parameters()},
    {'params': network.decoder.parameters()},
    {'params': network.transformer.parameters()},
], lr=args.base_lr)

network.to(device)

log_c, log_s, log_id1, log_id2, log_all = [],[],[],[],[]
log_c_temp, log_s_temp, log_id1_temp, log_id2_temp, log_all_temp = [],[],[],[],[]

if __name__ == '__main__':
  #txt所在的路径
  loss_save='/mnt/hdd/huying/hy/stc2/loss.txt'
  for i in range(args.epoch):
    i += (epoch_start + 1)

    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    
    # calculate losses
    loss_c, loss_s, loss_id_1, loss_id_2, out = network(content_images, style_images)
    
    loss_all = args.content_weight*loss_c + args.style_weight*loss_s + args.id1_weight*loss_id_1 + args.id2_weight*loss_id_2 
    
    log_c_temp.append(loss_c.item())
    log_s_temp.append(loss_s.item())
    log_id1_temp.append(loss_id_1.item())
    log_id2_temp.append(loss_id_2.item())
    log_all_temp.append(loss_all.item())
    
    # update parameters
    optimizer.zero_grad()
    loss_all.backward()
    optimizer.step()

    
    # calculate average loss
    if i % loss_count_interval == 0:
      log_c.append(np.mean(np.array(log_c_temp)))
      log_s.append(np.mean(np.array(log_s_temp)))
      log_id1.append(np.mean(np.array(log_id1_temp)))
      log_id2.append(np.mean(np.array(log_id2_temp)))
      log_all.append(np.mean(np.array(log_all_temp)))

      print('Epoch {:d}: '.format(i) +str( log_c[-1])+' '+str( log_s[-1])+' '+str( log_id1[-1])+' '+str( log_id2[-1])+' '+ str( log_all[-1]))

      file_save=open(loss_save,mode='a')
      file_save.write('\n'+'Epoch {:d}: '.format(i) +str( log_c[-1])+' '+str( log_s[-1])+' '+str( log_id1[-1])+' '+str( log_id2[-1])+' '+ str( log_all[-1]))
      file_save.close()

      log_c_temp, log_s_temp = [],[]
      log_id1_temp, log_id2_temp = [],[]
      log_all_temp = []

    if (i + 1) % loss_count_interval == 0 or (i + 1) == args.epoch:  
      state_dict = network.encoder.state_dict()
      for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
      torch.save(state_dict,
                '{:s}/encoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
      state_dict = network.transformer.state_dict()
      for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
      torch.save(state_dict,
                '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

      state_dict = network.decoder.state_dict()
      for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device('cpu'))
      torch.save(state_dict,
                '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
      
