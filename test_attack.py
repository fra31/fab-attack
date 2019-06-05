import tensorflow as tf
import numpy as np
import time
import scipy.io
import argparse
import settings
import sys
    
def saver(hps, adv, res, t1):  
  scipy.io.savemat('results/'+hps.dataset +'_'+ hps.model+'_' + hps.p + '_adv_' + hps.model + '_img_' + str(hps.im) +'_niter_'\
  + str(hps.n_iter)+'_nrestarts_'+ str(hps.n_restarts)+ '_eps_'+str(hps.eps)+'_las_' +str(hps.final_search) +'_alphamax_'+str(hps.alpha_max) +'_targetclass_'+str(hps.targetcl)+ '.mat', {'adv': adv, 'res': res, 'time': t1})

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Define hyperparameters.')
  parser.add_argument('--bs', type=int, default=100, help='batch size')
  parser.add_argument('--im', type=int, default=100, help='number of points to test')
  parser.add_argument('--model', type=str, default='plain', help='plain, l2-at, linf-at')
  parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, mnist')
  parser.add_argument('--n_restarts', type=int, default=1, help='number of restarts')
  parser.add_argument('--n_iter', type=int, default=50, help='number of iterations')
  parser.add_argument('--eps', type=float, default=1, help='epsilon for random restarts sampling')
  parser.add_argument('--p', type=str, default='linf', help='norm of the attack (linf, l2, l1)')
  parser.add_argument('--final_search', type=str, default='False', help='perform or not final search')
  parser.add_argument('--alpha_max', type=float, default=0.1, help='parameter of the attack')
  parser.add_argument('--targetcl', type=int, default=-1,
                      help='if not -1 the attack considers only the decision hyperplane between the original and the c-th most likely class, otherwise the closest one')
                                                
  hps = parser.parse_args()
  hps.n_labels = 10
  hps.final_search = True if hps.final_search in ['True', 'true', '1'] else False
  settings.init(hps)
  
  sess = tf.InteractiveSession()
  
  ### load dataset and create the model
  import utils
  data1, x_test, y_test, y_test_0 = utils.load_dataset(hps)
  conv_l, dense_l = utils.get_weights_conv(data1, hps)
  settings.init(hps)
  settings.init_layers(conv_l, dense_l)
  model = utils.Model(hps)
  
  if hps.p == 'linf': import FAB_linf
  elif hps.p == 'l2': import FAB_l2
  elif hps.p == 'l1': import FAB_l1
  
  ### run the attack in batches of size hps.bs for the first hps.im images of the test set  
  if hps.dataset in ['cifar10']: y_test_0 = y_test_0[0]
  t1 = time.time()
  adv = np.zeros(x_test[:hps.im].shape)
  res = np.zeros([hps.im])
  sp = 0
  while sp < hps.im:
    if hps.p == 'linf': res[sp:sp + hps.bs], adv[sp:sp + hps.bs] = FAB_linf.FABattack_linf(model, x_test[sp:sp+hps.bs], y_test_0[sp:sp+hps.bs], sess, hps)
        
    elif hps.p == 'l2': res[sp:sp + hps.bs], adv[sp:sp + hps.bs] = FAB_l2.FABattack_l2(model, x_test[sp:sp+hps.bs], y_test_0[sp:sp+hps.bs], sess, hps)
        
    elif hps.p == 'l1': res[sp:sp + hps.bs], adv[sp:sp + hps.bs] = FAB_l1.FABattack_l1(model, x_test[sp:sp+hps.bs], y_test_0[sp:sp+hps.bs], sess, hps)
    
    sp += hps.bs
  
  t1 = time.time() - t1
  print('attack performed in {:.2f} s'.format(t1))
  print('misclassified points: {:d}'.format(np.sum(res == 0)))
  print('success rate: {:d}/{:d}'.format(np.sum((res > 0)*(res < 1e10)), np.sum(res > 0)))
  print('average perturbation size: {:.5f}'.format(np.mean(res[(res > 0)*(res < 1e10)])))
  
  pred = sess.run(model.corr_pred, {model.x_input: adv, model.y_input: y_test_0[:hps.im], model.bs: hps.im})
  print('robust accuracy: {:.2f}%'.format(np.mean(pred.astype(int))*100))
  
  saver(hps, adv, res, t1)
  
  sess.close()