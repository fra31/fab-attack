import tensorflow as tf
import numpy as np
import time
import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')
  
def get_diff_logits_grads_batch(model, g, im, la, sess, hps):
  y2, g2 = sess.run([model.y, g], {model.x_input: im, model.y_input: la, model.bs: im.shape[0]})
  g2 = np.moveaxis(np.array(g2),0,1)
  la = np.squeeze(la)
  
  df = y2 - np.expand_dims(y2[np.arange(im.shape[0]),la],1)
  dg = g2 - np.expand_dims(g2[np.arange(im.shape[0]),la],1)
  
  df[np.arange(im.shape[0]), la] = 1e10
  
  return df, dg
    
def projection_l2_hyperplane(t2, w2, b2):
    ''' performs the operation described in Equation (4) wrt the l_2 norm '''
    t = t2.clone().float()
    w = w2.clone().float()
    b = b2.clone().float()
    d = torch.zeros(t.shape).float()
    
    c = (w*t).sum(1) - b
    ind2 = (c < 0).nonzero()
    w[ind2] *= -1
    c[ind2] *= -1
    
    u = torch.arange(0, w.shape[0]).unsqueeze(1)
    
    r = torch.max(t/w, (t-1)/w)
    r = torch.min(r, 1e12*torch.ones(r.shape).cuda())
    r = torch.max(r, -1e12*torch.ones(r.shape).cuda())
    r[w.abs() < 1e-8] = 1e12
    r[r == -1e12] = -r[r == -1e12]
    rs, indr = torch.sort(r, dim=1)
    rs2 = torch.cat((rs[:,1:], torch.zeros(rs.shape[0],1)),1)
    rs[rs == 1e12] = 0
    rs2[rs2 == 1e12] = 0
    
    w3 = w**2
    w3s = w3[u, indr]
    w5 = w3s.sum(dim=1, keepdim=True)
    ws = w5 - torch.cumsum(w3s, dim=1)
    d = -(r*w).clone()
    d = d*(w.abs() > 1e-8).float()
    s = torch.cat(((-w5.squeeze()*rs[:,0]).unsqueeze(1), torch.cumsum((-rs2 + rs)*ws, dim=1) - w5*rs[:,0].unsqueeze(-1)), 1)
    
    c4 = (s[:,0] + c < 0)
    c3 = ((d*w).sum(dim=1) + c > 0)
    c6 = c4.nonzero().squeeze()
    c2 = ((1 - c4) * (1 - c3)).nonzero().squeeze()

    counter = 0
    lb = torch.zeros(c2.shape[0])
    ub = torch.ones(c2.shape[0])*(w.shape[1] - 1)
    nitermax = torch.ceil(torch.log2(torch.tensor(w.shape[1]).float()))
    counter2 = torch.zeros(lb.shape).type(torch.cuda.LongTensor)
        
    while counter < nitermax:
      counter4 = torch.floor((lb + ub)/2)
      counter2 = counter4.type(torch.cuda.LongTensor)
      
      c3 = s[c2, counter2] + c[c2] > 0
      ind3 = c3.nonzero().squeeze()
      ind32 = (~c3).nonzero().squeeze()
      lb[ind3] = counter4[ind3]
      ub[ind32] = counter4[ind32]
      
      counter += 1
    
    lb = lb.cpu().numpy().astype(int)
    alpha = torch.zeros([1])
    
    if c6.nelement() != 0:
      alpha = c[c6]/w5[c6].squeeze(-1)
      d[c6] = -alpha.unsqueeze(-1)*w[c6]
    
    if c2.nelement() != 0:
      alpha = (s[c2, lb] + c[c2])/ws[c2, lb] + rs[c2, lb]
      if torch.sum(ws[c2, lb] == 0) > 0:
        ind = (ws[c2,lb] == 0).nonzero().squeeze().cpu().numpy()
        alpha[ind] = 0
      c5 = (alpha.unsqueeze(-1) > r[c2]).float()
      d[c2] = d[c2]*c5 - alpha.unsqueeze(-1)*w[c2]*(1 - c5)

    d = d*(w.abs() > 1e-8).float()
    
    return d
  
def linear_approximation_search(model, clean_im, clean_im_l, adv, niter, sess):
  a1 = np.copy(clean_im)
  a2 = np.copy(adv)
  u = np.arange(clean_im.shape[0])
  y1 = sess.run(model.y, {model.x_input: a1, model.y_input: clean_im_l, model.bs: clean_im.shape[0]})
  y2, la2 = sess.run([model.y, model.predictions], {model.x_input: a2, model.y_input: clean_im_l, model.bs: clean_im.shape[0]})
  
  for counter in range(niter):
    t1 = (y1[u, clean_im_l] - y1[u, la2]).reshape([-1, 1, 1, 1])
    t2 = (-(y2[u, clean_im_l] - y2[u, la2])).reshape([-1, 1, 1, 1])
    
    t3 = t1/(t1 + t2 + 1e-10)
    c3 = np.logical_and(0.0 <= t3, t3 <= 1.0)
    t3[np.logical_not(c3)] = 1.0
    
    a3 = a1*(1.0 - t3) + a2*t3
    a3 = np.clip(a3, 0.0, 1.0)
    
    y3, la3, pred = sess.run([model.y, model.predictions, model.corr_pred], {model.x_input: a3, model.y_input: clean_im_l, model.bs: clean_im.shape[0]})
    y1[pred] = y3[pred]
    a1[pred] = a3[pred]
    y2[np.logical_not(pred)] = y3[np.logical_not(pred)]
    la2[np.logical_not(pred)] = la3[np.logical_not(pred)]
    a2[np.logical_not(pred)] = a3[np.logical_not(pred)]
    
    res = np.sqrt(np.sum((a2 - clean_im)**2, axis=(1,2,3)))
  
  return res, a2
  
def FABattack_l2(model, clean_im, clean_im_l, sess, hps):
  ''' performs FAB attack on correctly classified points wrt the l_2-norm
  
  imput
  model       a TensorFlow model, with
              model.x_input: placeholder for the input images
              model.y_input: placeholder for the labels
              model.bs: placeholder for the batch size
              
              model.predictions: the class predictes
              model.corr_pred: returns (predicted class == true class)
              model.y: logits
              
  clean_im    the original images
  clean_im_l  the original labels
  hps         parameters of the attack
              hps.n_iter: iterations
              hps.n_restarts: restarts
              hps.eps: epsilon for the sampling when using restarts
              hps.alpha_max: alpha_max
              hps.n_labels: number of classes
              hps.targetcl: if -1 untargeted attack
                            if c with c in [2..n_labels] the attack considers only the decision boundary between the orginal class
                            and the c-th most likely according to the classification of the original point
              hps.final_search: if True a final search is performed
              
  output
  res_c       the norm of adversarial perturbations found (1e10 in case no adversarial example is found)
  adv_c       adversarial examples for the correctly classified images in clean_im
  '''
  
  ### creates tensors for the gradient of each logit wrt the input
  grads = [None]*hps.n_labels
  for cl in range(hps.n_labels):
    grads[cl] = tf.gradients(model.y[:,cl], model.x_input)[0]
  
  ### the attack is performed only on the correctly classified points
  pred = sess.run(model.corr_pred, {model.x_input: clean_im, model.y_input: clean_im_l, model.bs: clean_im.shape[0]})
  pred1 = np.copy(pred)
  im2 = clean_im[pred]
  la2 = np.squeeze(clean_im_l[pred])
  bs = np.sum(pred.astype(int))
  u1 = np.arange(bs)
  
  adv = np.copy(im2)
  adv_c = np.copy(clean_im)
  res2 = 1e10*np.ones([bs])
  res_c = np.zeros([clean_im_l.shape[0]])
  
  x1 = np.copy(im2)
  x0 = torch.from_numpy(np.reshape(np.copy(im2),[bs, -1])).cuda()
  
  if hps.targetcl > -1: targetla = np.argsort(y[pred1], axis=1)[:,-hps.targetcl]

  counter3 = 0
  while counter3 < hps.n_restarts:
    if counter3 > 0:
      ### random restarts ###
      t = np.random.randn(x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3])
      x1 = im2 + t/(1e-8 + np.sqrt(np.sum(t**2, axis=(1,2,3), keepdims=True)))*np.minimum(hps.eps, res2.reshape([-1,1,1,1]))*0.5
      x1 = np.clip(x1, 0.0, 1.0)
    
    counter2 = 0
    while counter2 < hps.n_iter:
      ### computation of the decision hyperplane ###
      df, dg = get_diff_logits_grads_batch(model, grads, x1, la2, sess, hps)
      
      if hps.targetcl == -1:
        dist1 = np.abs(df)/(1e-8 + np.sum(np.abs(dg), axis=(2,3,4)))
        ind = np.argmin(dist1, axis=1)
        b = - df[u1, ind] + np.sum(np.reshape(dg[u1, ind]*x1, [bs, -1]), axis=1)
        w = np.reshape(dg[u1, ind], [bs, -1])
      
      else:
        t1 = time.time()
        b = - df[u1, targetla] + np.sum(np.reshape(dg[u1, targetla]*x1, [bs, -1]), axis=1)
        w = np.reshape(dg[u1, targetla], [bs, -1])
        timedist += -(t1 - time.time())
      
      x2 = torch.from_numpy(np.reshape(x1,[bs, -1])).cuda()
      w2, b2 = torch.from_numpy(w).cuda(), torch.from_numpy(b).cuda()
      
      ### projection step ###
      d3 = projection_l2_hyperplane(torch.cat((x2,x0),0), torch.cat((w2, w2), 0), torch.cat((b2, b2),0)).cpu().numpy()
      d3[d3 != d3] = 1e-10
      d1 = np.reshape(d3[:bs], x1.shape)
      d2 = np.reshape(d3[-bs:], x1.shape)
      
      a1 = np.sqrt(np.sum(d1**2, axis=(1,2,3), keepdims=True))
      a2 = np.sqrt(np.sum(d2**2, axis=(1,2,3), keepdims=True))
      
      a3 = 1.05 ### extrapolation parameter
      alpha = np.minimum(a1/np.maximum(a1 + a2, 1e-20), hps.alpha_max)
      x1 = np.clip((x1 + d1*a3)*(1 - alpha) + (im2 + d2*a3)*alpha, 0.0, 1.0)
      pred = sess.run(model.corr_pred, {model.x_input: x1, model.y_input: la2, model.bs: bs})
      ind2 = np.where(pred == False)
      
      if pred[ind2].shape[0] > 0:
        t = np.sqrt(np.sum((x1[ind2] - im2[ind2])**2, axis=(1,2,3)))
        adv[ind2] = x1[ind2] * (t < res2[ind2]).astype(int).reshape([-1,1,1,1]) + adv[ind2]*(t >= res2[ind2]).astype(int).reshape([-1,1,1,1])
        res2[ind2] = t * (t < res2[ind2]).astype(int) + res2[ind2]*(t >= res2[ind2]).astype(int)
        
        ### backward step ###
        x1[ind2] = im2[ind2] + (x1[ind2] - im2[ind2])*0.9
        
      counter2 += 1
      
    counter3 += 1
  
  fl_success = (res2 < 1e10).astype(int)
  
  ### final search ###
  if hps.final_search:
    ind3 = res2 < 1e10
    res2t, advt = linear_approximation_search(model, im2, la2, adv, 3, sess)
    res2 = np.copy(res2t)
    adv = np.copy(advt)
    
  adv_c[pred1] = adv
  res_c[pred1] = res2*fl_success + 1e10*(1 - fl_success)
  
  return res_c, adv_c
  

  
    


    
    