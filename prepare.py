import os
import pickle
from transform import Transformer

'''
all = {'Dysrhythmia', 'Myocarditis', 'Hypertrophy',
        'Myocardial infarction', 'Healthy control', 'Bundle branch block',
        'Valvular heart disease', 'Cardiomyopathy'}
'''

dataset_path = '/media/farkob/OS/ptbdb/'
max_of_any = 100
test_divider = 4
accepted = {'Myocardial infarction', 'Healthy control'}

def get_prepped(db, data, diagnosis):
    a4 = []
    d3 = []
    d4 = []
    dia = []
    for i in range(5, 30, 5):
        a, b, c = db.prepare(d, i)
        a4.append(a)
        d4.append(b)
        d3.append(c)
        dia.append(diagnosis)
    return a4, d4, d3, dia

db = Transformer(dataset_path)

walk = [x for x in os.walk(dataset_path)]
data = { "a4": [], "d4": [], "d3": [], "dia": [] }
test = { "a4": [], "d4": [], "d3": [], "dia": [] }
count = 0
types = dict()


for folder in range(2, len(walk)):

    try:
        folder_name = walk[folder][0].split("/")[-1]
        file = folder_name + "/" + walk[folder][2][0].split(".")[0]

        diagnosis = db.get_diagnosis(file)
        if diagnosis in accepted:

            if diagnosis in types:

                if types[diagnosis] < max_of_any:

                    types[diagnosis] += 1

                    if types[diagnosis] % test_divider == 0:

                        d = db.get_data(file)
                        a, b, c, dia = get_prepped(db, d, diagnosis)
                        test["a4"] += a
                        test["d4"] += b
                        test["d3"] += c
                        test["dia"] += dia

                    else:

                        d = db.get_data(file)
                        a, b, c, dia = get_prepped(db, d, diagnosis)
                        data["a4"] += a
                        data["d4"] += b
                        data["d3"] += c
                        data["dia"] += dia

                    count += 1
                    print(count)

            else:

                types[diagnosis] = 1
                d = db.get_data(file)
                a, b, c, dia = get_prepped(db, d, diagnosis)
                data["a4"] += a
                data["d4"] += b
                data["d3"] += c
                data["dia"] += dia

                count += 1
                print(count)

    except Exception:
        pass

data = [data['a4'], data['d4'], data['d3'], data['dia']]

with open('data.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

test = [test['a4'], test['d4'], test['d3'], test['dia']]

with open('test.pickle', 'wb') as t_handle:
    pickle.dump(test, t_handle, protocol=pickle.HIGHEST_PROTOCOL)