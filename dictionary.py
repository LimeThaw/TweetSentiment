import torch
import config
import re

def _make_dict(tweets):
    entries = {}
    for t in tweets:
        words = re.split(' |\n',t)
        for w in words:
            if w in entries:
                entries[w] = entries[w]+1
            else:
                entries[w] = 1
    scores = list(entries.items())
    scores.sort(key=lambda x:-x[1])
    scores = list(filter(lambda x: _custom_filter(x[0]), scores))
    dict_list = [w[0] for w in scores[:config.DICT_SIZE-1]]
    return dict_list

def _custom_filter(w):
    fillers = []
    if w == "" or w[:4] == "http":
        return False
    if w in fillers:
        return False
    
    return True

def embed_tweets(tweets):
    dict_list = _make_dict(tweets)

    biglist = []
    for t in tweets:
        tmplist = []
        for w in re.split(' |\n',t):
            if w in dict_list:
                tmplist.append(dict_list.index(w)+1)
            else:
                tmplist.append(0)
        biglist.append(tmplist)

    max_len = max([len(x) for x in biglist])
    biglist = [x+[0]*(max_len-len(x)) for x in biglist]
    biglist = [torch.tensor(x) for x in biglist]

    return torch.stack(biglist)