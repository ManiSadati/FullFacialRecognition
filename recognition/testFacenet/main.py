import os
import sys
sys.path.insert(1, '../../data/lfw')
import torch
from torchvision import transforms
from PIL import Image
from new_dataset import *
from models import *
from loss import TripletLoss
import pickle
import time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data_direction = '../../data/lfw'
    num_embeddings = 128

    #DATA
    transform = transforms.Compose([transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
    mydataset = LFW_DataSet(data_direction, transform)
    print(len(mydataset))
    print(10586 + 2647)
    train_dataset, test_dataset = torch.utils.data.random_split(mydataset, lengths = [10586, 2647])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=3, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=3, pin_memory=True)
    print('hi')

    sdf = mydataset.labels
    cnt = 1
    asdf = []
    for i in range(1, len(sdf)):
        if(sdf[i] != sdf[i-1]):
            if(cnt > 1 and cnt < 20):
                asdf.append(cnt)
            cnt = 0
        cnt += 1
    asdf.append(cnt)
    print(asdf)
    plt.hist(asdf, bins = 20)
    plt.show()


    # time1 = time.time()
    # for _, (data, target) in enumerate(train_loader):
    #    data = data.cuda()
    #    print(data.mean())
    # time2 = time.time()
    # print(time2 - time1)
    #NETWORK
    net = VGG('VGG11', num_embeddings)
    gpu_device = 0
    if torch.cuda.is_available():
        with torch.cuda.device(gpu_device):
            net.cuda()
    if(True):
        with open('../../checkpoints/checkp.pickle', 'rb') as handle:
            net = pickle.load(handle)

    # #TRAIN
    params = net.parameters()
    learning_rate = 1e-7
    optimizer = torch.optim.Adam(params, lr = learning_rate)
    loss3 = TripletLoss(alpha=0.2)
    num_epochs = 50
    net.train()

    # for epoch in range(num_epochs):
    #     minibatches = make_mini_batch(data)
    #     loss = []
    #     val_loss = 0
    #     cnt = 0
    #     for batch in minibatches:
    #         print('#batch : ',cnt)
    #         if(cnt < 12):
    #             id2embeds = batch2embeddings(batch[0], net, dataloader)
    #             anchor, positive, negative = gen_triplets(batch, id2embeds, num_embeddings)
    #             l = loss3(anchor, positive, negative)
    #             loss.append(l)
    #             l.backward()
    #             optimizer.step()
    #         if(cnt <= 13 and cnt >= 13):
    #             id2embeds = batch2embeddings(batch[0], net, dataloader)
    #             anchor, positive, negative = gen_triplets(batch, id2embeds, num_embeddings)
    #             val_loss += loss3(anchor, positive, negative).item()
    #         if(cnt == 13):
    #             break
    #         cnt += 1
    #     print(torch.mean(torch.Tensor(loss)))
    #     print('VAL_LOSS : ',val_loss)
    #     with open('checkp.pickle', 'wb') as handle:
    #         pickle.dump(net, handle, protocol=pickle.HIGHEST_PROTOCOL)




    #id2embeds = batch2embeddings(mini_batches[0][0], net, dataloader)
    #dis = torch.pow(embeds - embed, 2)

