#from code.model.final_model import GraphRec
import pickle
import torch 
import torch.nn as nn
import os

from CONFIG import *
#from post_embedding import *
from post_encode import *
from user_encode import *
from post_embedding import *
from final_model import *

with open(os.path.join(LINK_DATA,"data.picke"),"rb") as out_put_file:
    user_dict, item_dict, event_dict, ui_dict, iu_dict, ur_dict,ir_dict = pickle.load(out_put_file)
embed_dim = 50
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
with open(os.path.join(LINK_DATA,"data.picke"),"rb") as out_put_file:
    user_dict, item_dict, event_dict, ui_dict, iu_dict, ur_dict,ir_dict = pickle.load(out_put_file)
with open(os.path.join(LINK_DATA,"content.picke"),"rb") as out_put_file:
    content=pickle.load(out_put_file)
u2e = nn.Embedding(len(user_dict), embed_dim).to(device)
i2e = nn.Embedding(len(item_dict), embed_dim).to(device)
r2e = nn.Embedding(len(event_dict), embed_dim).to(device)
postEncode = PostEncode(u2e, r2e,embedding_document, 50, 768, iu_dict,ir_dict,content,device=device)
userEncode = UserEncode(u2e, r2e,i2e, 50,iu_dict,ir_dict,device=device)
score = GraphRec(userEncode,postEncode,r2e)
print("A")
for i in range(100000):
    print(i)
    print(score([10,12],[10,12]))
print("b")