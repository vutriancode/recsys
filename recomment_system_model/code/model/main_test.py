import pickle
import torch 
import torch.nn as nn
import os

from CONFIG import *
#from post_embedding import *
from post_encode import *
from user_encode import *
from post_embedding import *


with open(os.path.join(LINK_DATA,"data.picke"),"rb") as out_put_file:
    user_dict, item_dict, event_dict, ui_dict, iu_dict, ur_dict,ir_dict = pickle.load(out_put_file)
embed_dim = 50
device = "cpu"
with open(os.path.join(LINK_DATA,"data.picke"),"rb") as out_put_file:
    user_dict, item_dict, event_dict, ui_dict, iu_dict, ur_dict,ir_dict = pickle.load(out_put_file)
with open(os.path.join(LINK_DATA,"content.picke"),"rb") as out_put_file:
    content=pickle.load(out_put_file)
u2e = nn.Embedding(len(user_dict), embed_dim).to(device)
i2e = nn.Embedding(len(item_dict), embed_dim).to(device)
r2e = nn.Embedding(len(event_dict), embed_dim).to(device)
postEncode = PostEncode(u2e, r2e,embedding_document, 50, 768)
userEncode = UserEncode(u2e, r2e,i2e, 50)
print("A")
postEncode([10],iu_dict,ir_dict,content)
userEncode([10],ui_dict,ur_dict,content)
postEncode([10],iu_dict,ir_dict,content)
userEncode([10],ui_dict,ur_dict,content)
postEncode([10],iu_dict,ir_dict,content)
userEncode([10],ui_dict,ur_dict,content)
postEncode([10],iu_dict,ir_dict,content)
userEncode([10],ui_dict,ur_dict,content)
postEncode([10],iu_dict,ir_dict,content)
userEncode([10],ui_dict,ur_dict,content)
postEncode([10],iu_dict,ir_dict,content)
userEncode([10],ui_dict,ur_dict,content)
print("b")