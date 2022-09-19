# file for logging the parameters information
# save the params and runtime so I can estimate the results
# This might end up being blended in the params.py file

from datetime import datetime
import os
import glob

def flush_logs():
    files = glob.glob('logs/*')
    for f in files:
        os.remove(f)
    print("logs flushed")

class Logger:
    def __init__(self, save_log):
        dateTimeObj = datetime.now()
        self.timestamp = dateTimeObj.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = "logs/" + self.timestamp  + ".log"
        self.save_log = save_log
        if (save_log):
            f = open(self.log_file, "w")
            f.write("Log file from: " + dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)"))
            f.close()


    def log_params(self, params):
        if (self.save_log):
            print("save params to log file")
            f = open(self.log_file,'a')
            f.write("\n---\nParams in config.yaml:\n\n")
            with open(params.path_to_yaml,'r') as firstfile:
                for line in firstfile:
                    f.write(line)
            f.write("\n---\n")
            f.close()
        else:
            print("save log file disabled")


    def log_state(self, state):
        if (self.save_log):
            f = open(self.log_file,'a')
            f.write("Additional informations:\n")
            f.write("number of parameters: " + str(state.number_of_parameters) + "\n")
            f.write("batch size: " + str(state.batch_size) + "\n")
            f.write("device: " + state.device + "\n")
            f.write("loss: " + state.loss + "\n")
            f.write("---\n")
            f.close()


    def log_temp(self, state):
        if (self.save_log):
            f = open(self.log_file,'a')
            f.write("Progress epoch: " + str(state.epoch) + " / " + str(state.epochs) + " . ")
            f.write("Total loss: " + str(state.average_loss) + "\n")
            f.close()
        else:
            print("save log file disabled")


    def stop_logging(self):
        if (self.save_log):
            print("done logging")
