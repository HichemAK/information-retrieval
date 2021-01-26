import json

from utils import read_cacm, preprocess_cacm

dictionary = read_cacm('CACM/cacm.all')
json.dump(dictionary, open('temp.json', 'w'), indent=4)
dictionary = preprocess_cacm(dictionary)
json.dump(dictionary, open('temp2.json', 'w'), indent=4)