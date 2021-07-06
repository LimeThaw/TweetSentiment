import torch
import network
import csv
import dictionary
import math
import config
import matplotlib.pyplot as plt
import numpy as np

net = network.Network()
loss_fn = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.Adam(net.parameters())

data = []
with open('stock_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)
tweets, labels = list(zip(*data))
labels = torch.stack([torch.tensor(max(0, int(l)), dtype=torch.float) for l in labels[1:]])
tweets = dictionary.embed_tweets(tweets[1:])

def train_epoch():
    idxs = np.random.permutation(np.arange(len(tweets)))
    losses = []

    for i in range(math.ceil(tweets.shape[0]/config.BATCH_SIZE)):
        min_idx = i*config.BATCH_SIZE
        max_idx = min(tweets.shape[0], (i+1)*config.BATCH_SIZE)
        batch_idxs = idxs[min_idx:max_idx]
        optimizer.zero_grad()

        preds = net(tweets[batch_idxs])
        gt = labels[batch_idxs]

        loss = loss_fn(preds, gt)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.average(losses)


print("Loss:")
losses = []

for _ in range(config.NUM_EPOCHS):
    loss = train_epoch()
    print(loss)
    losses.append(loss)

plt.plot(losses)
plt.yscale('log')
plt.show()