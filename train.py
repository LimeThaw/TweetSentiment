import torch
import network
import csv
import dictionary
import math
import config
import matplotlib.pyplot as plt
import numpy as np

tweets, labels = dictionary.get_data("dataset.csv", "vocab_65536.json", num_tweets=200000)

tweets_test = tweets[:config.BATCH_SIZE]
tweets = tweets[config.BATCH_SIZE:]
labels_test = labels[:config.BATCH_SIZE]
labels = labels[config.BATCH_SIZE:]

def train_epoch():
    idxs = np.random.permutation(np.arange(len(tweets)))
    # idxs = list(range(len(tweets)))
    losses = []

    num_batches = math.ceil(tweets.shape[0]/config.BATCH_SIZE)
    for i in range(num_batches):
        print("[%s%s] %s / %s"%("%"*int(i/5), " "*int((num_batches-i)/5), i, num_batches), end='\r')

        min_idx = i*config.BATCH_SIZE
        max_idx = min(tweets.shape[0], (i+1)*config.BATCH_SIZE)
        batch_idxs = idxs[min_idx:max_idx]
        optimizer.zero_grad()

        preds = net(tweets[batch_idxs]).flatten()
        # print(all([x>=0 and x<=1 for x in preds]))
        gt = labels[batch_idxs]

        loss = loss_fn(preds, gt)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.average(losses)


def accuracy(preds, labels):
    preds = [1 if x>0.5 else 0 for x in preds]
    rights = [1 if x==y else 0 for x, y in zip(preds, labels)]
    return float(sum(rights))/float(len(rights))


print(" Epoch\t Train\t\t\t Test\t\t\t Acc")
losses = []
losses_test = []
accuracies = []

net = network.Network()
loss_fn = torch.nn.BCELoss(reduction='mean')
# loss_fn = torch.nn.HingeEmbeddingLoss(reduction="mean")
# optimizer = torch.optim.SGD(net.embed.parameters(), lr=1e-1)
optimizer = torch.optim.SGD(net.trainable_weights(), lr=1e1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.1,
    patience=3
)

for i in range(config.NUM_EPOCHS):
    net.train()
    loss = train_epoch()
    losses.append(loss)

    net.eval()
    preds_test = net(tweets_test).flatten()
    loss_test = loss_fn(preds_test, labels_test).item()
    losses_test.append(loss_test)
    scheduler.step(loss_test)
    acc = accuracy(preds_test, labels_test)
    accuracies.append(acc)

    print("\033[K", i, "\t", loss, "\t", loss_test, "\t", acc)

torch.save(net.state_dict(), "./state.dict")

plt.plot(losses)
plt.plot(losses_test)
plt.plot(accuracies)
plt.yscale('log')
plt.show()