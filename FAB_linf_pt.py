''' This is a preliminary implementation of FAB-attack in PyTorch.
    It is only wrt Linf.
'''        

import numpy as np
import time
import torch
import argparse
import sys

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients

torch.set_default_tensor_type('torch.cuda.FloatTensor')
  
def get_diff_logits_grads_batch(model, im3, la):
  model.eval()
  im = Variable(torch.from_numpy(im3).float().to(device), requires_grad=True)
  with torch.enable_grad(): y = model(im)
  g2 = compute_jacobian(im, y).cpu().numpy()
  y2 = model(im.float()).cpu().detach().numpy()
  la = np.squeeze(la)
  df = y2 - np.expand_dims(y2[np.arange(im.shape[0]),la],1)
  dg = g2 - np.expand_dims(g2[np.arange(im.shape[0]),la],1)
  df[np.arange(im.shape[0]), la] = 1e10
  
  return df, dg

def compute_jacobian(inputs, output):
  assert inputs.requires_grad
  
  num_classes = output.size()[1]
  
  jacobian = torch.zeros(num_classes, *inputs.size())
  grad_output = torch.zeros(*output.size())
  if inputs.is_cuda:
  	grad_output = grad_output.cuda()
  	jacobian = jacobian.cuda()
  
  for i in range(num_classes):
  	zero_gradients(inputs)
  	grad_output.zero_()
  	grad_output[:, i] = 1
  	output.backward(grad_output, retain_graph=True)
  	jacobian[i] = inputs.grad.data
  
  return torch.transpose(jacobian, dim0=0, dim1=1)
    
def projection_linf(t2, w2, b2):
    t = t2.clone().float()
    w = w2.clone().float()
    b = b2.clone().float()
    d = torch.zeros(t.shape).float()
    
    ind2 = ((w*t).sum(1) - b < 0).nonzero()
    w[ind2] *= -1
    b[ind2] *= -1
    
    c5 = (w < 0).type(torch.cuda.FloatTensor)
    a = torch.ones(t.shape).cuda()
    d = (a*c5 - t)*(w != 0).type(torch.cuda.FloatTensor)
    a -= a*(1 - c5)
    
    p = torch.ones(t.shape)*c5 - t*(2*c5 - 1)
    indp = torch.argsort(p, dim=1)

    b = b - (w*t).sum(1)
    b0 = (w*d).sum(1)
    b1 = b0.clone()

    counter = 0
    indp2 = indp.unsqueeze(-1).flip(dims=(1,2)).squeeze()
    u = torch.arange(0, w.shape[0])
    ws = w[u.unsqueeze(1), indp2]
    bs2 = - ws*d[u.unsqueeze(1), indp2]
    
    s = torch.cumsum(ws.abs(), dim=1)
    sb = torch.cumsum(bs2, dim=1) + b0.unsqueeze(1)
    
    c = b - b1 > 0
    b2 = sb[u, -1] - s[u, -1]*p[u, indp[u, 0]]
    c_l = (b - b2 > 0).nonzero().squeeze()
    c2 = ((b - b1 > 0) * (b - b2 <= 0)).nonzero().squeeze()
    
    lb = torch.zeros(c2.shape[0])
    ub = torch.ones(c2.shape[0])*(w.shape[1] - 1)
    nitermax = torch.ceil(torch.log2(torch.tensor(w.shape[1]).float()))
    counter2 = torch.zeros(lb.shape).type(torch.cuda.LongTensor)
    
    while counter < nitermax:
      counter4 = torch.floor((lb + ub)/2)
      counter2 = counter4.type(torch.cuda.LongTensor)
      indcurr = indp[c2, -counter2 - 1]
      b2 = sb[c2, counter2] - s[c2, counter2]*p[c2, indcurr]
      c = b[c2] - b2 > 0
      ind3 = c.nonzero().squeeze()
      ind32 = (~c).nonzero().squeeze()
      lb[ind3] = counter4[ind3]
      ub[ind32] = counter4[ind32]
      counter += 1
    
    lb = lb.cpu().numpy().astype(int)
    counter2 = 0
    
    if c_l.nelement != 0:  
      lmbd_opt = (torch.max((b[c_l] - sb[c_l, -1])/(-s[c_l, -1]), torch.zeros(sb[c_l, -1].shape))).unsqueeze(-1)
      d[c_l] = (2*a[c_l] - 1)*lmbd_opt
      
    lmbd_opt = (torch.max((b[c2] - sb[c2, lb])/(-s[c2, lb]), torch.zeros(sb[c2, lb].shape))).unsqueeze(-1)
    d[c2] = torch.min(lmbd_opt, d[c2])*c5[c2] + torch.max(-lmbd_opt, d[c2])*(1-c5[c2])

    return (d*(w != 0).type(torch.cuda.FloatTensor)).cpu()
  
def linear_approximation_search(model, clean_im, clean_im_l, adv, niter):
  a1 = np.copy(clean_im)
  a2 = np.copy(adv)
  u = np.arange(clean_im.shape[0])
  model.eval()
  y1 = model(torch.from_numpy(a1).float().to(device)).cpu().detach().numpy()
  y2 = model(torch.from_numpy(a2).float().to(device)).cpu().detach().numpy()
  la2 = np.argmax(y2, 1)
  
  for counter in range(niter):
    t1 = (y1[u, clean_im_l] - y1[u, la2]).reshape([-1, 1, 1, 1])
    t2 = (-(y2[u, clean_im_l] - y2[u, la2])).reshape([-1, 1, 1, 1])
    
    t3 = t1/(t1 + t2 + 1e-10)
    c3 = np.logical_and(0.0 <= t3, t3 <= 1.0)
    t3[np.logical_not(c3)] = 1.0
    
    a3 = a1*(1.0 - t3) + a2*t3
    
    y3 = model(torch.from_numpy(a3).float().to(device)).cpu().detach().numpy()
    la3 = np.argmax(y3, 1)
    pred = la3 == clean_im_l
    
    y1[pred] = y3[pred] + 0
    a1[pred] = a3[pred] + 0
    y2[np.logical_not(pred)] = y3[np.logical_not(pred)] + 0
    la2[np.logical_not(pred)] = la3[np.logical_not(pred)] + 0
    a2[np.logical_not(pred)] = a3[np.logical_not(pred)] + 0
    
  res = np.amax(np.abs(a2 - clean_im), axis=(1,2,3))

  return res, a2
  
def fab_pt(model, clean_im, clean_im_l):
  model.eval()
  y = torch.from_numpy(clean_im_l)
  logits = model(torch.from_numpy(clean_im).float().to(device)).cpu().detach().numpy()
  pred = np.argmax(logits, axis=1) == clean_im_l
  pred1 = np.copy(pred)
  im2 = clean_im[pred]
  la2 = np.squeeze(clean_im_l[pred])
  bs = np.sum(pred.astype(int))
  u1 = np.arange(bs)
  clean_im_2 = np.copy(clean_im)
  adv = np.copy(im2)
  adv_c = np.copy(clean_im)
  res2 = 1e10*np.ones([bs])
  res_c = np.zeros([clean_im.shape[0]])
  x1 = np.copy(im2)
  x0 = torch.from_numpy(np.reshape(np.copy(im2),[bs, -1])).cuda()
  
  counter_restarts = 0
  
  while counter_restarts < hps.n_restarts:
    if counter_restarts > 0:
      t = np.random.uniform(-1, 1, x1.shape)
      x1 = im2 + np.minimum(res2, hps.eps).reshape([-1,1,1,1])*t/np.amax(np.abs(t), axis=(1,2,3), keepdims=True)*0.5
      x1 = np.clip(x1, 0.0, 1.0)
    
    counter_iter = 0
    
    while counter_iter < hps.n_iter:
      df, dg = get_diff_logits_grads_batch(model, x1, la2)
      dist1 = np.abs(df)/(1e-8 + np.sum(np.abs(dg), axis=(2,3,4)))
      ind = np.argmin(dist1, axis=1)
      b = - df[u1, ind] + np.sum(np.reshape(dg[u1, ind]*x1, [bs, -1]), axis=1)
      w = np.reshape(dg[u1, ind], [bs, -1])
      x2 = torch.from_numpy(np.reshape(x1,[bs, -1])).float().cuda() if hps.dataset == 'ImageNet' else  torch.from_numpy(np.reshape(x1,[bs, -1])).cuda()
      w2, b2 = torch.from_numpy(w).cuda(), torch.from_numpy(b).cuda()
      
      d3 = projection_linf(torch.cat((x2,x0),0), torch.cat((w2, w2), 0), torch.cat((b2, b2),0)).numpy()
      d1 = np.reshape(d3[:bs], x1.shape)
      d2 = np.reshape(d3[-bs:], x1.shape)
      a1 = np.amax(np.abs(d1), axis=(1,2,3), keepdims=True)
      a2 = np.amax(np.abs(d2), axis=(1,2,3), keepdims=True)
      
      alpha = np.minimum(np.maximum(a1/np.maximum(a1 + a2, 1e-20), 0.0), hps.alpha_max)
      x1 = np.clip((x1 + d1*hps.overshooting)*(1 - alpha) + (im2 + d2*hps.overshooting)*alpha, 0.0, 1.0)
      logits = model(torch.from_numpy(x1).float().to(device)).cpu().detach().numpy()
      pred = np.array(np.argmax(logits, axis=1) == la2)
      ind2 = np.where(pred == False)
      
      if np.sum(pred.astype(int)) < im2.shape[0]:
        t = np.amax(np.abs(x1[ind2] - im2[ind2]), axis=(1,2,3))
        adv[ind2] = x1[ind2] * (t < res2[ind2]).astype(int).reshape([-1,1,1,1]) + adv[ind2]*(t >= res2[ind2]).astype(int).reshape([-1,1,1,1])
        res2[ind2] = t * (t < res2[ind2]).astype(int) + res2[ind2]*(t >= res2[ind2]).astype(int)
        
        x1[ind2] = im2[ind2] + (x1[ind2] - im2[ind2])*hps.backward_beta
        
      counter_iter += 1
      
    counter_restarts += 1
  
  ind3 = res2 < 1e10
  print('success rate: {}/{} (on correctly classified points)'.format(np.sum(ind3), np.sum(pred1)))
  
  if hps.las:
    res2t, advt = linear_approximation_search(model, im2, la2, adv, 3)
    res2 = np.copy(res2t)
    adv = np.copy(advt)
  
  ind3 = ind3.astype(float)
  adv_c[pred1] = adv
  res_c[pred1] = res2*ind3 + 1e10*(1 - ind3)
  
  return res2, adv_c
  
if __name__ == '__main__':
  ''' This example assumes that models, checkpoints and datasets from https://github.com/yaodongyu/TRADES
      have been dowanloaded.
  '''
  
  parser = argparse.ArgumentParser(description='Define hyperparameters.')
  parser.add_argument('--bs', type=int, default=500)
  parser.add_argument('--attack', type=str, default='fab')
  parser.add_argument('--model', type=str, default='plain')
  parser.add_argument('--dataset', type=str, default='cifar10')
  parser.add_argument('--sp', type=int, default=0, help='index of the first image on which the attack is run')
  parser.add_argument('--n_restarts', type=int, default=1)
  parser.add_argument('--n_iter', type=int, default=100)
  parser.add_argument('--eps', type=float, default=-1, help='epsilon for the random restarts')
  parser.add_argument('--p', type=str, default='linf', help='Lp-norm of the attack')
  parser.add_argument('--las', type=str, default='False', help='final search')
  parser.add_argument('--alpha_max', type=float, default=0.1, help='param: alpha_max')
  parser.add_argument('--overshooting', type=float, default=1.05, help='param: eta')
  parser.add_argument('--backward_beta', type=float, default=0.9, help='param: beta')
  parser.add_argument('--path_to_save', type=str, default='./results/', help='directory to save the results, must already exist')
  
  hps = parser.parse_args()
  hps.n_labels = 43 if hps.dataset == 'gts' else 10
  hps.las = True if hps.las in ['True', 'true', '1'] else False
  if hps.eps == -1: hps.eps = 0.3 if hps.dataset == 'mnist' else 0.0314
  assert hps.p == 'linf', 'Lp-norm not supported'
  
  ### load models and datasets one can get at https://github.com/yaodongyu/TRADES
  if hps.dataset == 'cifar10':
    from models.wideresnet import WideResNet
    device = torch.device("cuda")
    model = WideResNet().to(device)
    model.load_state_dict(torch.load('./checkpoints/model_cifar_wrn.pt'))
    model.eval()
    
    X_data = np.load('./data_attack/cifar10_X.npy')
    Y_data = np.load('./data_attack/cifar10_Y.npy')
    X_data = np.transpose(X_data, (0, 3, 1, 2))
    
    
  elif hps.dataset == 'mnist':
    from models.small_cnn import SmallCNN
    device = torch.device("cuda")
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load('./checkpoints/model_mnist_smallcnn.pt'))
    model.eval()
    
    X_data = np.load('./data_attack/mnist_X.npy')
    Y_data = np.load('./data_attack/mnist_Y.npy')
    X_data = np.transpose(np.expand_dims(X_data, axis=3), (0, 3, 1, 2))
    
  ### run the attack
  res, adv = fab_pt(model, X_data[hps.sp:hps.sp + hps.bs], Y_data[hps.sp:hps.sp + hps.bs])
    
  np.save(hps.path_to_save + hps.dataset+ '_' + hps.attack + '_' + str(hps.p) + '_niter_'\
    + str(hps.n_iter)+'_nrestarts_'+ str(hps.n_restarts)+ '_eps_'+str(hps.eps)+'_las_' +str(hps.las) + '_'+str(hps.sp)\
    + '_' + str(hps.sp + hps.bs), {'norms_adv': res, 'adv': adv})
    