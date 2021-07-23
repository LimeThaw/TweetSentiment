import torch
import config
import re
import json
import os.path
import csv



def _read_csv(dataset_path, num_tweets):
    data = []
    with open(dataset_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    if num_tweets<0:
        num_tweets=len(data)-1
    tweets, labels = list(zip(*data[1:num_tweets+1]))
    for i in range(1, len(labels)):
        if labels[i] != "1" and labels[i] != "0":
            print(tweets[i], " --- ", labels[i])
            break
    return tweets, labels

def _make_vocab(dataset_path, vocab_file, num_tweets=-1, force_new=False):

    if not force_new and os.path.isfile(vocab_file):
        print("Loading vocabulary from ", vocab_file)
        with open(vocab_file) as json_file:
            json_data = json.load(json_file)
        dict_list = [[x["word"], x["score"]] for x in json_data]
        assert len(dict_list) == config.DICT_SIZE-1
        return zip(*dict_list)

    print("Loading vocab")
    print("\tReading data...")
    tweets, labels = _read_csv(dataset_path, num_tweets)

    entries = {}
    scores = {}
    for i, t in enumerate(tweets):
        print("\tEnumerate: %s/%s"%(i, num_tweets), end="\r")
        words = re.split(' |\n',t)
        for w in words:
            if w in entries:
                entries[w] = entries[w]+1
            else:
                entries[w] = 1
                scores[w] = 0
            
            if labels[i] == "1":
                scores[w] += 1
            else:
                scores[w] -= 1
    # for w in scores:
    #     scores[w] = entries[w] * scores[w]/entries[w]
    scores = list(scores.items())
    print("\n\tSorting...")
    scores.sort(key=lambda x:-abs(x[1]))
    scores = list(filter(lambda x: _custom_filter(x[0]), scores))
    dict_list = scores[:config.DICT_SIZE-1]

    json_data = [
        {"word": x[0], "score": x[1]} for x in dict_list
    ]

    with open(vocab_file, 'w') as outfile:
        json.dump(json_data, outfile)
    print("\tDone")
    return zip(*dict_list)

def _custom_filter(w):
    punctuation = "!|.|,|?|;|/|'|*|>|<".split("|")
    numbers = "1|2|3|4|5|6|7|8|9|0".split("|")
    fillers = [
        "<user>",
        "<url>",
        "a",
        "|",
        "the",
        "this",
        "are",
        "is",
        "in",
    ]+punctuation+numbers
    if w == "" or w[:4] == "http" or len(w)==1:
        return False
    if w in fillers:
        return False
    
    return True

def get_data(dataset_path, vocab_file, num_tweets=-1):
    tweets, labels = _read_csv(dataset_path, num_tweets)
    with open(vocab_file) as json_file:
        json_data = json.load(json_file)
    dict_list = [x["word"] for x in json_data]
    dict_dict = {}
    for i, w in enumerate(dict_list):
        dict_dict[w] = i+1

    max_len = 0
    biglist = []
    for i, t in enumerate(tweets):
        if i%100==0: print("Dictionarizing: ", i, "/", len(tweets), end="\r")
        tmplist = []
        for w in re.split(' |\n',t):
            if w in dict_dict:
                tmplist.append(dict_dict[w])
            else:
                tmplist.append(0)
        biglist.append(tmplist)
        max_len = max(max_len, len(tmplist))

    biglist = [torch.tensor(x+[0]*(max_len-len(x)), dtype=torch.long) for x in biglist]

    # with open("dictionarized.json", 'w') as outfile:
    #     json.dump(biglist, outfile)

    labels = torch.stack([torch.tensor(max(0, int(l)), dtype=torch.float) for l in labels])
    print("\033[KDictionarized")

    return torch.stack(biglist), labels