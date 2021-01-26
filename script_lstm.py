import json
from tqdm import tqdm
from lstm_model import LSTMModel
from sklearn.metrics import f1_score

def load_data(filename):
    with open(filename,'r', encoding='utf8') as infile:
        data = json.load(infile)
        data_new = dict()
        for k,v in list(data.items()):
            v_new = data_new.setdefault(k,dict())
            for k1,v1 in list(v.items()):
                v1_new = v_new.setdefault(k1,dict())
                if k1 != 'tokens':
                    for k2,v2 in list(v1.items()):
                        v2_new = [l.replace('B_','').replace('I_','').replace('I','').replace('B','') for l in v2]
                        v1_new[k2] = v2_new
                else:
                    v_new[k1] = v1
    return data_new

def load_test_ids(filename):
    with open(filename, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        ids = list()
        ids.extend(data['test_IDs'])
    
    return ids

data = load_data('data_laptop_absa.json')
testIds = load_test_ids('test_IDs_laptop_absa.json')

sentiments_model = LSTMModel(data, 'S')
aspects_model = LSTMModel(data, 'A')
modifiers_model = LSTMModel(data, 'M')

s_test, s_pred = sentiments_model.fit_predict(testIds)
a_test, a_pred = aspects_model.fit_predict(testIds)
m_test, m_pred = modifiers_model.fit_predict(testIds)

f1_sentiments = f1_score(s_pred, s_test, average='macro')
f1_aspects = f1_score(a_pred, a_test, average='macro')
f1_modifiers = f1_score(m_pred, m_test, average='macro')

print(f"F1-Score Sentiments {f1_sentiments}")
print(f"F1-Score Aspects {f1_aspects}")
print(f"F1-Score Modfiiers {f1_modifiers}")

print(f"F1-Score Average {(f1_sentiments + f1_aspects + f1_modifiers)/3}")