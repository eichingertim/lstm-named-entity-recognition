import pandas as pd
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn import cluster

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class PreProcessor:

    @staticmethod
    def stop_words(sentences):
        stopwordsList = stopwords.words('english')
        more_stopwords = ['.', ';', '!', ',', '*', '?', '/', '-', '"', '..']
        stopwordsList.extend(more_stopwords)


        for sen in sentences:
            for i, t in enumerate(sen['tokens']):
                if t in stopwordsList and not t in ['very', 'biggest', 'big', 'highly', 'high', 'not', 'cannot']:
                    sen['labeling'][i] = 'O'

        return sentences

    @staticmethod
    def lemmatize_sen(sen):
        l = WordNetLemmatizer()

        return [l.lemmatize(t) for t in sen]

    @staticmethod
    def getVocabs(data):
        sentences = [PreProcessor.lemmatize_sen(v.get('tokens')) for k, v in data.items()]
        word2vec_model = Word2Vec(sentences, min_count=1)
        return word2vec_model.wv.vocab

    @staticmethod
    def preprocess(data, extraction_of):
        sentences = list()
        for k,v in tqdm(data.items()):
            tokens = PreProcessor.lemmatize_sen(v.get('tokens'))
            labeler = [val for s, val in data[k].items() if s != 'tokens']

            for j in range(0, len(labeler)):
                sentence = dict()
                sentence['sentenceId'] = k+'|'+ str(j)
                sentence['tokens'] = tokens

                eo = ''
                if extraction_of == 'S':
                    eo = 'sentiments'
                elif extraction_of == 'A':
                    eo = 'aspects'
                else:
                    eo = 'modifiers'

                sentence['labeling'] = labeler[j].get(eo)
                sentences.append(sentence)
        
        sentences = PreProcessor.stop_words(sentences)

        words = list()
        for s in sentences:
            for i, t in enumerate(s['tokens']):
                word = dict()
                word['id'] = s['sentenceId']
                word['token'] = t
                word['labeling'] = s['labeling'][i]
                if s['labeling'][i] == 'O' or s['labeling'][i] == extraction_of:
                    words.append(word)
                else:
                    word['labeling'] = 'O'
                    words.append(word)
                words.append(word)

        return pd.DataFrame(words)
