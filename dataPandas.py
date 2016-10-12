from datetime import datetime
from pprint import pprint

__author__ = 'daniel'

import os
import numpy
import pandas
from scipy.stats import kde
from matplotlib import pyplot as plt


class Stream(object):


    def __init__(self, data_path, file_name):
        self.data = pandas.read_csv(os.path.join(data_path,file_name))
        self.data['TIMESTAMP'] = self.data['TIMESTAMP'].astype('datetime64[ns]')
        self.data.ffill(inplace=True)

    def get_time_window(self, column, start, end):
        select = (self.data['TIMESTAMP'] > start) & \
                 (self.data['TIMESTAMP'] < end)
        window = self.data[select]
        return window[column]

    def get_feature_names(self):
        filtr = ['TIMESTAMP', 'rain', 'status', 'avgMeasuredTime', 'extID', 'medianMeasuredTime', '_id',
                  'REPORT_ID']
        feature_names = [fname for fname in self.data.keys() if fname not in filtr]
        return feature_names

    def get_pdf_of_time_window(self, column, start, end):
        window = self.get_time_window(column, start, end)

        # compute distribution
        kernel = kde.gaussian_kde(window.values)
        x_grid = numpy.linspace(min(window), max(window), len(window))
        distribution = kernel(x_grid)
        return distribution, x_grid

    def get_statistics(self, column, start, end):
        window = self. get_time_window(column, start, end)
        if len(window) == 0:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        statistics = {'mean': window.mean().item(), 'std': window.std().item(), 'min': window.min().item(), 'max': window.max().item()}
        return statistics


    def print_pdf_of_time_window(self, column, start, end):
        pdf, x_grid = self.get_pdf_of_time_window(column, start, end)
        fig, ax = plt.subplots()
        ax.plot(x_grid, pdf, "-b")
        betas = self.calculate_betas_custom_distribution(pdf, x_grid)
        pprint(betas)
        for beta in betas:
            plt.axvline(x=beta, color='red')
        # ax.hist(self.get_time_window(column, start, end).values, bins=30, normed=True)
        ax.get_yaxis().set_visible(False)
        plt.show()

    def calculate_betas_custom_distribution(self, pdf, x_grid):
        norm_constant = sum(pdf)
        # normalize values so they are between 0 and 1
        pdf = [float(x)/float(norm_constant) for x in pdf]
        s = 0
        alphabet_size = 5
        quantile = 1./(alphabet_size)
        betas = []
        for i, v in enumerate(pdf):
            s += v
            if s >= quantile:
                betas.append(x_grid[i])
                quantile = float(len(betas)+1)/ alphabet_size
                if quantile == 1:
                    break
        return betas

    def get_start_date(self):
        return self.data['TIMESTAMP'].min()

    def get_end_date(self):
        return self.data['TIMESTAMP'].max()

if __name__ == '__main__':
    time_format = '%Y-%m-%dT%H'
    datapath = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'DataImport', 'Analysis')
    sensor_id = "trafficData187774.csv"
    #
    stream = Stream(datapath, sensor_id)
    pprint(stream.get_time_window('vehicleCount', datetime.strptime("2014-08-01T00", time_format), "2014-08-01T-10"))
    # pprint(sum(stream.get_pdf_of_time_window('vehicleCount', "2014-08-01T-09:00:00", "2014-08-01T-10:00:00")[0]))
    stream.print_pdf_of_time_window('avgSpeed', "2014-08-01T-09:00:00", "2014-08-01T-14:00:00")