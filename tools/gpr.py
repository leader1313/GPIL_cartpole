import numpy as np
import torch

class GPRegression:
  def __init__(self, X, Y, kernel):
    self.X = X
    self.Y = Y
    self.kern = kernel
    # self.sigma = torch.tensor([1.0], requires_grad=True)
    self.sigma = torch.tensor([np.log(0.3)])


  def predict(self, x):
    Kx = self.kern.K(x)
    Kxx = self.kern.K(self.X, x)
    K = self.kern.K(self.X)

    sig = torch.exp(self.sigma)

    mean = Kxx.t().mm(torch.solve(self.Y, K+torch.eye(K.shape[0])*sig)[0])
    sigma = torch.diag(Kx - Kxx.t().mm(torch.solve(Kxx, K+torch.eye(K.shape[0])*sig)[0])).reshape(x.shape[0], -1) + sig
    
    c = 0.45
    for j in range(len(mean)):
        if mean[j]> c :
            mean[j]=1
        else : mean[j] = 0

    return mean, sigma
  
  def compute_grad(self, flag):
    self.sigma.requires_grad = flag
    self.kern.compute_grad(flag)

  def negative_log_likelihood(self):
    K = self.kern.K(self.X) + torch.eye(self.X.shape[0])*torch.exp(self.sigma)

    self.K = K

    invKY = torch.solve(self.Y, K+torch.eye(self.Y.shape[0])*0.000001)[0]
    # logdet = torch.cholesky(K+torch.eye(K.shape[0])*0.000001, upper=False).diag().log().sum()
    sign, logdet = torch.slogdet(K+torch.eye(K.shape[0])*1e-6)
    return (logdet + self.Y.t().mm(invKY))

  def learning(self):
    max_iter = 1000
    # max_iter = 50

    self.compute_grad(True)
    param = self.kern.param() + [self.sigma]
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(param, lr = learning_rate)

    for i in range(max_iter):
      optimizer.zero_grad()
      f = self.negative_log_likelihood() 
      f.backward()
      # def closure():
      #   optimizer.zero_grad()
      #   f = self.negative_log_likelihood() 
      #   f.backward()
      #   return f
      # optimizer.step(closure)
      optimizer.step()
    self.compute_grad(False)
    print('params:', torch.exp(self.kern.param()[0]), torch.exp(self.sigma))
    

if __name__=="__main__":
  import sys, pickle
  from kernel import GaussianKernel
  import matplotlib.pyplot as plt
  plt.style.use("ggplot")
 
  # with open('data/supervisor_demo'+str(episode_num)+'.pickle', 'rb') as handle:
  #   b = pickle.load(handle)
  N = 200
  X = np.linspace(0, np.pi*2, N)[:,None]
  Y = np.sin(X) + np.random.randn(N)[:,None] * 0.2

  X = torch.from_numpy(X).float()
  Y = torch.from_numpy(Y).float()

  kern = GaussianKernel()
  model = GPRegression(X, Y, kern)

  xx = np.linspace(0, np.pi*2, 100)[:,None]
  xx = torch.from_numpy(xx).float()
  mm, ss = model.predict(xx)

  mm = mm.numpy().ravel()
  ss = np.sqrt(ss.numpy().ravel())
  xx = xx.numpy().ravel()
  X = X.numpy().ravel()
  Y = Y.numpy().ravel()

  plt.plot(X, Y, "*")
  line = plt.plot(xx, mm)
  plt.plot(xx, mm+ss, "--", color=line[0].get_color())
  plt.plot(xx, mm-ss, "--", color=line[0].get_color())
  plt.show()

  print("params", torch.exp(model.kern.param()[0]), torch.exp(model.sigma), model.negative_log_likelihood())
  model.learning()
  print("params", torch.exp(model.kern.param()[0]), torch.exp(model.sigma), model.negative_log_likelihood())

  xx = np.linspace(0, np.pi*2, 100)[:,None]
  xx = torch.from_numpy(xx).float()
  mm, ss = model.predict(xx)

  mm = mm.numpy().ravel()
  ss = np.sqrt(ss.numpy().ravel())
  xx = xx.numpy().ravel()

  plt.plot(X, Y, "*")
  line = plt.plot(xx, mm)
  plt.plot(xx, mm+ss, "--", color=line[0].get_color())
  plt.plot(xx, mm-ss, "--", color=line[0].get_color())
  plt.show()

