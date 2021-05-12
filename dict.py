import json
from difflib import get_close_matches

data = json.load(open('data.json'))

def ED(w):
    w = w.lower()

    if w in data:
        return data[w]
        
    elif w.title() in data:
        return data [w.title()]

    elif w.upper() in data:
        return data [w.upper()]
    
    else: #len(get_close_matches(w,data.keys())) > 0:
 
        YN = input(f'did you mean {get_close_matches(w,data.keys())[0]} instead? press Y for Yes, press N for No')
        if YN == "Y"or YN == "y":
            return "meaning is",data[get_close_matches(w,data.keys()) [0]]
        elif YN == "N"or YN == "n":
            return "The word doesnt exist."
        else:
            return "invalid entry"

    #else:
        #return "word doesnt exist, please check the word"

word = input ("Enter word:-")

output = ED(word)

if type(output) == list:
    for item in output:
        print (item)
else:
    print (output)