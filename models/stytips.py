import torch.nn as nn
import torch
import numpy as np
import torch.nn as nn
from .transformer_tools import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
  """MLP as implemented in timm
  """
  def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
    super().__init__()
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    drops = to_2tuple(drop)

    self.fc1 = nn.Linear(in_features, hidden_features)
    self.act = act_layer()
    self.drop1 = nn.Dropout(drops[0])
    self.fc2 = nn.Linear(hidden_features, out_features)
    self.drop2 = nn.Dropout(drops[1])

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.fc2(x)
    x = self.drop2(x)
    return x


class Self_Attention(nn.Module):
  """Self Attention as implemented in timm
  """
  def __init__(self, d_model, nhead=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
    super().__init__()
    assert d_model % nhead == 0, 'd_model needs to be divisible by nhead'
    self.nhead = nhead
    self.scale = (d_model // nhead) ** -0.5
    self.sc_token_num = 196
    self.sc_token = nn.Parameter(torch.zeros(1,self.sc_token_num,d_model))
    self.sc_token = trunc_normal_(self.sc_token,std=.02)

    self.to_qkv = nn.Linear(d_model, d_model*3, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(d_model, d_model)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x):
    B, N, C = x.size()
    sc_token = self.sc_token.repeat(B,1,1)
    #print()
    x = torch.cat((x,sc_token),dim=1)
    B, N1, C = x.size()
    #print(N1)
    qkv = self.to_qkv(x).reshape(B, N1, 3, self.nhead, C // self.nhead).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    attn = (q @ k.transpose(-1, -2)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    
    #print(x.shape)
    x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
    x = x[:,:N,:]
    x = self.proj(x)
    x = self.proj_drop(x)

    return x


class Cross_Attention(nn.Module):
  """Attention for decoder layer.Some palce may be called "inter attention"
  """
  def __init__(self, d_model, nhead=8, qkv_bias=False, attn_drop=0., proj_drop=0.,kv_channels=[384,768,1536],kv_shape=[56,28,14],q_channel=768,q_shape=28):
    super().__init__()
    assert d_model % nhead == 0, 'd_model needs to be divisible by nhead'
    self.nhead = nhead
    self.scale = (d_model // nhead) ** -0.5
    self.fpn_num = len(kv_channels)
    self.to_q = nn.Linear(d_model, d_model, bias=qkv_bias)
    self.to_kv = nn.Linear(d_model, d_model*2, bias=qkv_bias)
    self.input_proj = nn.ModuleList()
    for i in range(self.fpn_num):
      self.input_proj.append(
        nn.Sequential(
          nn.Linear(kv_channels[i],d_model*2,bias = qkv_bias),
        )
      )
 
    #self.to_kv = nn.Linear(d_model, d_model*2, bias=qkv_bias)
    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(d_model, d_model)
    self.proj_drop = nn.Dropout(proj_drop)

  def forward(self, x, y):
    """
      Args:
        x: output of the former layer
        y: memery of the encoder layer
    """
    
    B, Nx, C = x.size()
    #print(x.shape)
    k = []
    v = []
    # #print(len(y),self.fpn_num)
    for i in range(self.fpn_num):
      
      _,Ny,_ = y[i][0].size() 
      #print(Ny)
      #print(self.input_proj[i](y[i][0]).shape)
      _k, _v = self.input_proj[i](y[i][0]).chunk(2, -1)
      k.append(_k.view(-1, Ny, self.nhead, C // self.nhead).transpose(1, 2))
      v.append(_v.view(-1, Ny, self.nhead, C // self.nhead).transpose(1, 2))
    #_, Ny, _ = y.size()
    q = self.to_q(x).reshape(B, Nx, self.nhead, C // self.nhead).permute(0, 2, 1, 3)
    
    k = torch.cat(k, dim=2)  # (B, hn, L, C)
    v = torch.cat(v, dim=2)  # (B, hn, L, C)
    
    #print(q.shape,k.shape,v.shape)
    #print(q.shape,k.shape,v.shape)
    attn = (q @ k.transpose(-1, -2)) * self.scale
    attn = attn.softmax(dim=-1)
    #attn = attn*(self.nhead**-0.5)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    

    return x

class TransformerDecoderLayer(nn.Module):
  """Transformer Decoder Layer
  """
  def __init__(self, d_model, nhead=8, mlp_ratio=4, qkv_bias=False, attn_drop=0.,
         drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_first=False):
    super().__init__()
    mlp_hidden_dim = int(d_model * mlp_ratio)
    
    self.attn1 = Self_Attention(d_model, nhead=nhead, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
    self.attn2 = Cross_Attention(d_model, nhead=nhead, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
    self.mlp = Mlp(d_model, hidden_features=mlp_hidden_dim, out_features=d_model, act_layer=act_layer, drop=drop)
    
    self.norm_first = norm_first
    self.norm1 = norm_layer(d_model)
    self.norm2 = norm_layer(d_model)
    self.norm3 = norm_layer(d_model)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


  def forward(self, x, y):
    """
      Args:
        x: output of the former layer
        y: memery of the encoder layer
    """
  
    
    if self.norm_first == True:
      #print(x.shape,self.drop_path(self.attn1(self.norm1(x))).shape)

      x = x + self.drop_path(self.attn1(x))
      x = x + self.drop_path(self.attn2(x, y))
      x = x + self.drop_path(self.mlp(x))
      #print(x.shape)
    else:
      x = (x + self.drop_path(self.attn1(x)))
      x = (x + self.drop_path(self.attn2(x, y)))
      x = (x + self.drop_path(self.mlp(x)))
    return x
  

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


# compute channel-wise means and variances of features
def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4, 'The shape of feature needs to be a tuple with length 4.'
    B, C = size[:2]
    feat_mean = feat.reshape(B, C, -1).mean(dim=2).reshape(B, C, 1, 1)
    feat_std = (feat.reshape(B, C, -1).var(dim=2) + eps).sqrt().reshape(B, C, 1, 1)
    return feat_mean, feat_std


# normalize features
def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


########################################## Transfer Module ##########################################

class Transformer(nn.Module):
  """The Transfer Module of Style Transfer via Transformer

  Taking Transformer Decoder as the transfer module.

  Args:
    config: The configuration of the transfer module
  """
  def __init__(self):
    super(Transformer, self).__init__()
    self.layers = nn.ModuleList([
      TransformerDecoderLayer(d_model=384, nhead=8, mlp_ratio=4, qkv_bias=False, attn_drop=0.,
          drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_first=True
          ) \
      for i in range(3)
    ])

  def forward(self, content_feature, style_feature ):
    """
    Args:
      content_feature: Content features，for producing Q sequences. Similar to tgt sequences in pytorch. (Tensor,[Batch,sequence,dim])
      style_feature : Style features，for producing K,V sequences.Similar to memory sequences in pytorch.(Tensor,[Batch,sequence,dim])

    Returns:
      Tensor with shape (Batch,sequence,dim)
    """
    
    for layer in self.layers:
      content_feature = layer(content_feature, style_feature)
    
    return content_feature


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(384, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

class Net(nn.Module):
  def __init__(self, encoder, decoder, transformer, vgg):
    super(Net, self).__init__()
    self.mse_loss = nn.MSELoss()
    self.encoder = encoder
    self.decoder = decoder
    self.transformer = transformer

    # features of intermediate layers
    lossNet_layers = list(vgg.children())
    self.feat_1 = nn.Sequential(*lossNet_layers[:4])  # input -> relu1_1
    self.feat_2 = nn.Sequential(*lossNet_layers[4:11]) # relu1_1 -> relu2_1
    self.feat_3 = nn.Sequential(*lossNet_layers[11:18]) # relu2_1 -> relu3_1
    self.feat_4 = nn.Sequential(*lossNet_layers[18:31]) # relu3_1 -> relu4_1
    self.feat_5 = nn.Sequential(*lossNet_layers[31:44]) # relu3_1 -> relu4_1

    # fix parameters
    for name in ['feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5']:
      for param in getattr(self, name).parameters():
        param.requires_grad = False


  # get intermediate features
  def get_interal_feature(self, input):
    result = []
    for i in range(5):
      input = getattr(self, 'feat_{:d}'.format(i+1))(input)
      result.append(input)
    return result
  

  def calc_content_loss(self, input, target, norm = False):
    assert input.size() == target.size(), 'To calculate loss needs the same shape between input and taget.'
    assert target.requires_grad == False, 'To calculate loss target shoud not require grad.'
    if norm == False:
        return self.mse_loss(input, target) 
    else:
        return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))


  def calc_style_loss(self, input, target):
    assert input.size() == target.size(), 'To calculate loss needs the same shape between input and taget.'
    assert target.requires_grad == False, 'To calculate loss target shoud not require grad.'
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return self.mse_loss(input_mean, target_mean) + \
        self.mse_loss(input_std, target_std)


  # calculate losses
  def forward(self, i_c, i_s):
    f_c = self.encoder(i_c)   
    f_s = self.encoder(i_s)
 
    f_cc, f_c_reso = f_c[0][0], f_c[0][1]
    f_ss, f_s_reso = f_s[0][0], f_s[0][1]
    
    f_cs = self.transformer(f_cc, f_s)
    
    f_cc = self.transformer(f_cc, f_c)
    f_ss = self.transformer(f_ss, f_s)

    B, N, C= f_cs.shape
 
    H = int(np.sqrt(N))
    f_cs = f_cs.permute(0, 2, 1)
    f_cs = f_cs.view(B, C, -1, H)
    f_cc = f_cc.permute(0, 2, 1).view(B, C, -1, H)
    f_ss = f_ss.permute(0, 2, 1).view(B, C, -1, H)

    i_cs = self.decoder(f_cs)
    i_cc = self.decoder(f_cc)
    i_ss = self.decoder(f_ss)

    
    f_c_loss = self.get_interal_feature(i_c)  
    f_s_loss = self.get_interal_feature(i_s)
  
    f_i_cs_loss = self.get_interal_feature(i_cs)
    f_i_cc_loss = self.get_interal_feature(i_cc)
    f_i_ss_loss = self.get_interal_feature(i_ss)
    
    loss_id_1 = self.mse_loss(i_cc, i_c) + self.mse_loss(i_ss, i_s)

    loss_c, loss_s, loss_id_2 = 0, 0, 0
    
    loss_c = self.calc_content_loss(f_i_cs_loss[-2], f_c_loss[-2], norm=True) + \
             self.calc_content_loss(f_i_cs_loss[-1], f_c_loss[-1], norm=True)
    for i in range(1, 5):
      loss_s += self.calc_style_loss(f_i_cs_loss[i], f_s_loss[i])
      loss_id_2 += self.mse_loss(f_i_cc_loss[i], f_c_loss[i]) + self.mse_loss(f_i_ss_loss[i], f_s_loss[i])
    
    return loss_c, loss_s, loss_id_1, loss_id_2, i_cs