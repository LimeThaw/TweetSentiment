import torch
import network
import warnings
import dictionary
import math
import config
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=mpl.cbook.mplDeprecation)


dict_list, dict_scores = dictionary._make_vocab('dataset.csv', "vocab_"+str(config.DICT_SIZE)+".json", force_new=False)

words = [0]*len(dict_list)#torch.Tensor(len(dict_list), 1, dtype=torch.long)
word_labels = [0]*len(dict_list)#torch.Tensor(len(dict_list), 1)
# words = torch.tensor(len(dict_list), 1, dtype=torch.long)
# word_labels = torch.tensor((len(dict_list), 1))
for i in range(1, len(dict_list)):
    words[i]=i
    word_labels[i] = 1 if dict_scores[i]>0 else -1
words = torch.tensor(words, dtype=torch.long)
word_labels = torch.tensor(word_labels)


words_test = words[:config.BATCH_SIZE]
# words = words[config.BATCH_SIZE:]
word_labels_test = word_labels[:config.BATCH_SIZE]
# word_labels = word_labels[config.BATCH_SIZE:]

net = network.Network()
net=net.embed
my_distance = torch.nn.PairwiseDistance()
# loss_fn = torch.nn.BCELoss(reduction='sum')
loss_fn = torch.nn.HingeEmbeddingLoss(reduction="mean")
# optimizer = torch.optim.SGD(net.embed.parameters(), lr=1e-1)
optimizer = torch.optim.SGD(net.parameters(), lr=1e2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=0.1,
    patience=3
)

def train_epoch():
    idxs = np.random.permutation(np.arange(len(words)))
    # idxs = list(range(len(tweets)))
    losses = []

    num_batches = math.ceil(words.shape[0]/config.BATCH_SIZE)
    for i in range(num_batches):
        print("[%s%s] %s / %s"%("%"*int(i/5), " "*int((num_batches-i+2)/5), i, num_batches), end='\r')

        min_idx = i*config.BATCH_SIZE
        max_idx = min(words.shape[0], (i+1)*config.BATCH_SIZE)
        if (max_idx-min_idx)%2 != 0:
            max_idx -= 1
        batch_idxs = idxs[min_idx:max_idx]
        optimizer.zero_grad()

        batch_len = len(batch_idxs)
        hb = int(batch_len/2)
        preds = net(words[batch_idxs])
        preds = torch.cdist(preds[:hb], preds[hb:])
        # print(all([x>=0 and x<=1 for x in preds]))
        gt = torch.tensor([1 if word_labels[i]==word_labels[hb+i] else -1 for i in range(hb)])

        loss = loss_fn(preds, gt)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.average(losses)

print("Epoch\tTrain\tTest")
losses = []
losses_test = []

for i in range(64):
    net.train()
    loss = train_epoch()
    losses.append(loss)

    net.eval()
    hb = int(config.BATCH_SIZE/2)
    preds = net(words_test)
    preds = my_distance(preds[:hb], preds[hb:])
    # print(all([x>=0 and x<=1 for x in preds]))
    gt = torch.tensor([1 if word_labels_test[i]==word_labels_test[hb+i] else -1 for i in range(hb)])

    loss_test = loss_fn(preds, gt).item()
    losses_test.append(loss_test)
    scheduler.step(loss_test)

    print("\033[K", i, "\t", loss, "\t", loss_test)

torch.save(net.state_dict(), config.get_embed_dict_name())

plt.plot(losses)
plt.plot(losses_test)
plt.yscale('log')
plt.show()