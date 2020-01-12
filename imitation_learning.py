from tools.gpr import GPRegression
from tools.kernel import GaussianKernel
import sys, pickle, torch
import matplotlib.pyplot as plt
import numpy as np

def main():
    plt.style.use("ggplot")
    episode_num = 20
    graph = {'loss': [],
             'episode_num' : []}

    learning = {'state': [0,0,0,0],
                'action': [],
                'dataset': []}
    for i in range(episode_num) :
        
        #load pickle file
        with open('data/supervisor_demo'+str(i+1)+'.pickle', 'rb') as handle:
            results = pickle.load(handle)
        
        # ==============================================
        # Data slice and preprocess
        # ==============================================
        results['state'] = np.array(results['state'])
        results['action'] = np.array(results['action'])[...,None]
        
        #learning episode each
        learning['state'] = np.array(results['state'])
        learning['action'] = np.array(results['action'])
        
        #learning episode merge
        # if i == 0 :
        #     learning['state'] = np.array(results['state'])
        # else :
        #     learning['state'] = np.append(learning['state'],results['state'], axis= 0)

        # learning['action'] = np.append(learning['action'],results['action'])[...,None]

        # None : (N, 1) 형태를 유지할 수 있게해줌
        print(learning['state'].shape)
        print(learning['action'].shape)
        learning['dataset'] = np.append(learning['state'],learning['action'],axis = 1)
        print(learning['dataset'].shape)
        dataset = torch.from_numpy(learning['dataset']).float()
        
        # validate test and tain data
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train, test = torch.utils.data.random_split(dataset, [train_size, test_size])
        # type change 
        train_X = train[:]
        train_X = train_X[:,:4]
        train_Y = train[:]
        train_Y = train_Y[:,4:5]


        kern = GaussianKernel()
        model = GPRegression(train_X, train_Y, kern)

        print("params", torch.exp(model.kern.param()[0]), torch.exp(model.sigma), model.negative_log_likelihood())
        model.learning()
        print("params", torch.exp(model.kern.param()[0]), torch.exp(model.sigma), model.negative_log_likelihood())
        PATH = 'model/learner_'+str(i+1)
        torch.save(model, PATH)
        
        test_X = test[:]
        test_X = test_X[:,:4]
        test_Y = test[:]
        test_Y = test_Y[:,4:5]
        print(test_X.shape)
        mm , ss = model.predict(test_X)
        # c : Criteria of binary

        loss = (((mm-test_Y)**2).sum())/len(test_Y)
        graph['loss'].append(loss)
        graph['episode_num'].append(i)
    print(graph)
    X = graph['episode_num']
    Y = graph['loss']
    plt.plot(X, Y, "*")
    line = plt.plot(X, Y)
    plt.ylim(0.0, 1.0)
    # plt.plot(xx, mm+ss, "--", color=line[0].get_color())
    # plt.plot(xx, mm-ss, "--", color=line[0].get_color())
    plt.show()
    # print (mm)
    # xx = np.linspace(0, np.pi*2, 100)[:,None]
    # xx = torch.from_numpy(xx).float()
    # mm, ss = model.predict(xx)

    # mm = mm.numpy().ravel()
    # ss = np.sqrt(ss.numpy().ravel())
    # xx = xx.numpy().ravel()

    # plt.plot(X, Y, "*")
    # line = plt.plot(xx, mm)
    # plt.plot(xx, mm+ss, "--", color=line[0].get_color())
    # plt.plot(xx, mm-ss, "--", color=line[0].get_color())
    # plt.show()

if __name__=="__main__":
    main()