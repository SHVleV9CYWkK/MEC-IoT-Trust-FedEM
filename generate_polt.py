import os
import time

from utils.plots import make_plot

if __name__ == '__main__':

    # make_plot("./logs/" + args.experiment, "Train/Metric", path)
    # make_plot("./logs/" + args.experiment, "Train/Loss", path)
    # make_plot("./logs/" + args.experiment, "Test/Loss", path)
    path = os.getcwd() + '//' + time.strftime("%H%M-%d%m%Y", time.localtime())
    if not os.path.exists(path):
        os.makedirs(path)
    make_plot("./logs/" + "n-baiot", "Test/Metric", path)