import json

class Config():
    def __init__(self, config_file):
        with open(config_file) as conf_file:
            self.conf= json.load(conf_file)
    def load_conf(self):
        return self.conf