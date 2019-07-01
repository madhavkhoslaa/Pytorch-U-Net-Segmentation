from flask_restful import Resource, Api
from flask import Flask
from config import Config
import torch
from mode.Unet import UNeT
app= Flask(__name__)
api= Api(app)
class Segmented():
    def __init__(self):
        a= Config('../config.json')
        a= a.load_conf()
        if torch.cuda.is_available():
            model = UNeT(n_class= 1).cuda()
            model.load_state_dict(torch.load(a["Model Save"], map_location="cuda:0"))
        else:
            model= UNeT(n_class= 1)
            model.load_state_dict(torch.load(a["Model Save"]))


        
