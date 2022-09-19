import yaml

class Params:
    def __init__(self, path_to_yaml):
        self.path_to_yaml = path_to_yaml
        with open(self.path_to_yaml, 'r') as file:
            self.yaml = yaml.safe_load(file)

    def log_params(self, logfile):
        print("save params to log file")
        f = open(logfile,'a')
        f.write("\n\nconfig.yaml content:")
        with open(self.path_to_yaml,'r') as firstfile:
            for line in firstfile:
                f.write(line)
        f.write("")
        f.close()
