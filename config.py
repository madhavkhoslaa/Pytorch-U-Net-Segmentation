import json

class config():
    def __init__(self, config_file):
        with open(config_file) as conf_file:
            self.conf= json.load(conf_file)
    def get_config():
        return self.conf