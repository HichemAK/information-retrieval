import json

from models import BooleanModel
from utils import read_cacm, preprocess_cacm

dictionary = read_cacm('CACM/cacm.all')
dictionary = preprocess_cacm(dictionary)

#json.dump(dictionary, open('temp.json', 'w'), indent=4)
#json.dump(dictionary, open('temp2.json', 'w'), indent=4)

bm = BooleanModel(dictionary)
print(len(bm._dict))
print(bm.eval("((((((not 'geometry' and 'science'))))))"))
