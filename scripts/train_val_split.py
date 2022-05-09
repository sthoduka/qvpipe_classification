import json
import numpy as np

with open('../classification_track/qv_pipe_train.json', 'r') as fp:
    data = json.load(fp)

all_keys = list(data.keys())
np.random.shuffle(all_keys)
train_keys = all_keys[:4000]
val_keys = all_keys[4000:]

with open('../classification_track/train_keys.json', 'w') as fp:
    json.dump(train_keys, fp)

with open('../classification_track/val_keys.json', 'w') as fp:
    json.dump(val_keys, fp)
