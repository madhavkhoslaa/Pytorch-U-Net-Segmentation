import json

class Config():
    """Creates an object which takes a json file and returns a 
        dictionary when load_conf is called on the object"""
    def __init__(self, config_file):
        with open(config_file) as conf_file:
            self.conf= json.load(conf_file)
    def load_conf(self):
        return self.conf