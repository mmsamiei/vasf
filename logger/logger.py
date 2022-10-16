import pathlib

class Logger():
    def __init__(self, log_dir_path):
        pathlib.Path(log_dir_path).mkdir(parents=True, exist_ok=True)
        self.log_path = pathlib.Path(log_dir_path)
        print(self.log_path)
        self.text_file = open(self.log_path.joinpath("log.txt"), 'w+')
    
    def print(self, text):
        self.text_file.write(text+"\n")
        print(text)
    
    def plot(self, fig, path):
        fig.savefig(self.log_path.joinpath(path))
