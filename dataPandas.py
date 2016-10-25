from datetime import datetime
from pprint import pprint
import os
import numpy
import pandas
from scipy.stats import kde
from matplotlib import pyplot as plt

__author__ = 'daniel'

time_format = '%Y-%m-%dT%H:%M'

class Stream(object):
    def __init__(self, data_path, file_name):
        self.data = pandas.read_csv(os.path.join(data_path, file_name))
        self.data[u'TIMESTAMP'] = self.data[u'TIMESTAMP'].astype('datetime64[ns]')
        self.data.sort_values(by=u'TIMESTAMP', inplace=True)
        self.data.set_index([u'TIMESTAMP'], inplace=True)
        self.data.bfill(inplace=True)

    def get_time_range(self):
        start_date = self.data.first_valid_index()
        end_date = self.data.last_valid_index()
        return start_date, end_date

    def fill_in_missing_values(self, startdate, enddate, freq='5min'):
        date_index = pandas.date_range(startdate, enddate, freq=freq)
        dup = self.data.index.duplicated()
        double_idxes = []
        if any(dup):
            for i, d in enumerate(dup):
                if d:
                   double_idxes.append(self.data.index[i])
        dates = map(lambda x: x.strftime(time_format), self.data.loc[double_idxes])
        self.data.drop(pandas.to_datetime(dates))
        self.data = self.data.reindex(date_index)
        self.data.bfill(inplace=True)

    def get_time_window(self, column, start, end):
        select = (self.data.index >= start) & \
                 (self.data.index < end)
        window = self.data[select]
        return window[column]

    def get_point_in_time(self, start):
        select = (self.data.index >= start)
        point = self.data[select].values[0]
        return point

    def get_points_in_time_frame(self, start, end, column_or_columns=None):
        select = (self.data.index >= start) & \
                 (self.data.index < end)
        if column_or_columns is None:
            window = self.data[select]
        else:
            window = self.data[select][column_or_columns]
        return window

    def transform_time_window_for_neural_network_input(self, start, end, column_or_columns=None):
        window = self.get_points_in_time_frame(start, end, column_or_columns)
        window = numpy.array(map((lambda x: [x]), window.values))
        return window


    def get_feature_names(self):
        filtr = [u'TIMESTAMP', u'rain', u'status', u'avgMeasuredTime', u'extID', u'medianMeasuredTime', u'_id',
                 u'REPORT_ID']
        feature_names = [f_name for f_name in self.data.keys() if f_name not in filtr]
        return feature_names

    def get_pdf_of_time_window(self, column, start, end):
        window = self.get_time_window(column, start, end)

        # compute distribution
        kernel = kde.gaussian_kde(window.values)
        x_grid = numpy.linspace(min(window), max(window), len(window))
        distribution = kernel(x_grid)
        return distribution, x_grid

    def get_statistics(self, column, start, end):
        window = self.get_time_window(column, start, end)
        if len(window) == 0:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        statistics = {'mean': window.mean().item(), 'std': window.std().item(), 'min': window.min().item(),
                      'max': window.max().item()}
        return statistics

    def print_pdf_of_time_window(self, column, start, end):
        pdf, x_grid = self.get_pdf_of_time_window(column, start, end)
        fig, ax = plt.subplots()
        ax.plot(x_grid, pdf, "-b")
        betas = calculate_betas_custom_distribution(pdf, x_grid)
        pprint(betas)
        for beta in betas:
            plt.axvline(x=beta, color='red')
        # ax.hist(self.get_time_window(column, start, end).values, bins=30, normed=True)
        ax.get_yaxis().set_visible(False)
        plt.show()

    def get_start_date(self):
        return self.data.first_valid_index()

    def get_end_date(self):
        return self.data.last_valid_index()



def calculate_betas_custom_distribution(pdf, x_grid):
    norm_constant = sum(pdf)
    # normalize values so they are between 0 and 1
    pdf = [float(x) / float(norm_constant) for x in pdf]
    s = 0
    alphabet_size = 5
    quantile = 1. / (alphabet_size)
    betas = []
    for i, v in enumerate(pdf):
        s += v
        if s >= quantile:
            betas.append(x_grid[i])
            quantile = float(len(betas) + 1) / alphabet_size
            if quantile == 1:
                break
    return betas


def read_in_streams(main_data_path, context_data_path):
    main_streams = {}

    for file_name in os.listdir(main_data_path):
        name, extension = file_name.split('.')
        if extension == 'csv':
            main_streams[name] = Stream(main_data_path, file_name)
    main_features = main_streams.values()[0].get_feature_names()

    context_stream = {}
    for file_name in os.listdir(context_data_path):
        name, extension = file_name.split('.')
        if extension == 'csv':
            context_stream = Stream(context_data_path, file_name)

    context_features = context_stream.get_feature_names()
    return main_streams, main_features, context_stream, context_features


if __name__ == '__main__':
    time_format = '%Y-%m-%dT%H'
    datapath = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'DataImport', 'Analysis')
    sensor_id = "trafficData187774.csv"
    #
    stream = Stream(datapath, sensor_id)
    pprint(stream.get_time_window('vehicleCount', datetime.strptime("2014-08-01T00", time_format), "2014-08-01T-10"))
    # pprint(sum(stream.get_pdf_of_time_window('vehicleCount', "2014-08-01T-09:00:00", "2014-08-01T-10:00:00")[0]))
    stream.print_pdf_of_time_window('avgSpeed', "2014-08-01T-09:00:00", "2014-08-01T-14:00:00")
