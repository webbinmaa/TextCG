# -*- coding: utf-8 -*-
# coding: utf-8

from nltk.corpus import stopwords
from collections import Counter
import re
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm


dataset0 = "mr"
dataset1 = 'train'
nlp = StanfordCoreNLP(r'E:\stanford-corenlp-4.5.4')



stop_words = set(stopwords.words('english'))  # 去除停止词
least_freq = 5
if dataset0 == "mr" or dataset0 == "SST1" or dataset0 == "SST2" or dataset0 == "TREC":
    stop_words = set()
    least_freq = 0



def load_dataset(dataset0, dataset1):
    """Get the content and labels of the target document."""
    contents= []
    with open(dataset0 + '/' + f'{dataset1}.txt', 'r', encoding='utf-8') as file:
        texts = file.readlines()
    for l in texts:
        text = l.strip()
        contents.append(text)
    return contents


def clean_text(text: str):
    "cleaning texts"
    text = text.lower()  # tolowercase
    text = re.sub(r"([\w\.-]+)@([\w\.-]+)(\.[\w\.]+)", " ", text)  # Delete email address
    text = re.sub(r"([\w\.-]+)@([\w\.-]+)", " ", text)  # Delete email address
    text = re.sub(r"([\w\.-]+)(\.[\w\.]+)", " ", text)  # Delete urLs
    text = text.replace("'ll ", " will ")
    text = text.replace("'d ", " would ")
    text = text.replace("'m ", " am ")
    text = text.replace("'s ", " is ")
    text = text.replace("'re ", " are ")
    text = text.replace("'ve ", " have ")
    text = text.replace(" can't ", " can not ")
    text = text.replace(" ain't ", " are not ")
    text = text.replace("n't ", " not ")
    text = text.replace(". . .", " . ")
    text = text.replace(". . .", " . ")
    text = text.replace(" '", " ")
    text = re.sub(r"\.{2,}", '.', text)  # Remove redundancy
    text = re.sub(r'\.$', '', text.strip())
    text = re.sub(r'^\.', '', text.strip())
    text = re.sub(r"[^A-Za-z0-9,.!?\'`]", " ", text)
    text = text.replace(",", " , ")
    text = text.replace("!", " ! ")
    text = text.replace("?", " ? ")
    text = text.replace("'", "")

    text = re.sub(r"\s{2,}", " ", text)
    return " ".join(text.strip().split())


def words_list(texts, word2index):
    "Get a list of words for each text based on the cleaned word2index."
    words_list = []

    for t in texts:
        temp = []
        t_split = nlp.word_tokenize(t)
        for i in range(0, len(t_split)):
            if t_split[i] in word2index:
                temp.append(t_split[i])

        words_list.append(temp)

    return words_list


if __name__ == '__main__':
    texts = load_dataset(dataset0, dataset1)

    # handle texts
    texts_clean = [clean_text(t) for t in tqdm(texts)]
    word2count = Counter([w for t in tqdm(list(texts_clean)) for w in nlp.word_tokenize(t)])
    word_count = [[w, c] for w, c in word2count.items() if
                  c >= least_freq and w not in stop_words]
    word2index = {w: i for i, (w, c) in enumerate(word_count)}

    # Get the word list of the cleaned text
    words_list = words_list(texts_clean, word2index)
    texts_remove = [" ".join(ws) for ws in words_list]

    # save
    with open(f"{dataset0}/{dataset1}.texts_clean.txt", "w") as f:
        f.write("\n".join(texts_remove))

    print('done')