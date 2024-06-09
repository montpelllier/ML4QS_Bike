##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

import sys
import copy
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from Python3Code.util.VisualizeDataset import VisualizeDataset
from Python3Code.Chapter3.DataTransformation import LowPassFilter
from Python3Code.Chapter3.DataTransformation import PrincipalComponentAnalysis
from Python3Code.Chapter3.ImputationMissingValues import ImputationMissingValues
from Python3Code.Chapter3.KalmanFilters import KalmanFilters

# Set up the file names and locations.
DATA_PATH = Path('./results/')
DATASET_FNAME = 'dataset1_result_outliers.csv'
RESULT_FNAME = 'dataset1_result_final.csv'
ORIG_DATASET_FNAME = 'dataset1.csv'


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    print_flags()

    # Next, import the data from the specified location and parse the date index.
    try:
        dataset = pd.read_csv(Path(DATA_PATH / DATASET_FNAME), index_col=0)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # We'll create an instance of our visualization class to plot the results.
    DataViz = VisualizeDataset(__file__)

    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]) * 1000

    MisVal = ImputationMissingValues()
    LowPass = LowPassFilter()
    PCA = PrincipalComponentAnalysis()

    if FLAGS.mode == 'imputation':
        # Let us impute the missing values and plot an example.

        imputed_mean_dataset = MisVal.impute_mean(copy.deepcopy(dataset), 'hr_watch_rate')
        imputed_median_dataset = MisVal.impute_median(copy.deepcopy(dataset), 'hr_watch_rate')
        imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'hr_watch_rate')

        DataViz.plot_imputed_values(dataset, ['original', 'mean', 'median', 'interpolation'], 'hr_watch_rate',
                                    imputed_mean_dataset['hr_watch_rate'],
                                    imputed_median_dataset['hr_watch_rate'],
                                    imputed_interpolation_dataset['hr_watch_rate'])

    elif FLAGS.mode == 'kalman':
        # Using the result from Chapter 2, let us try the Kalman filter on the light_phone_lux attribute and study the result.
        try:
            original_dataset = pd.read_csv(
                DATA_PATH / ORIG_DATASET_FNAME, index_col=0)
            original_dataset.index = pd.to_datetime(original_dataset.index)
        except IOError as e:
            print('File not found, try to run previous crowdsignals scripts first!')
            raise e

        KalFilter = KalmanFilters()
        kalman_dataset = KalFilter.apply_kalman_filter(
            original_dataset, 'acc_phone_x')
        DataViz.plot_imputed_values(kalman_dataset, [
            'original', 'kalman'], 'acc_phone_x', kalman_dataset['acc_phone_x_kalman'])
        DataViz.plot_dataset(kalman_dataset, ['acc_phone_x', 'acc_phone_x_kalman'], [
            'exact', 'exact'], ['line', 'line'])

        # We ignore the Kalman filter output for now...

    elif FLAGS.mode == 'lowpass':

        # Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz

        # Determine the sampling frequency.
        fs = float(1000) / milliseconds_per_instance
        cutoff = 1.5
        # Let us study acc_phone_x:
        new_dataset = LowPass.low_pass_filter(copy.deepcopy(
            dataset), 'acc_phone_x', fs, cutoff, order=10)
        DataViz.plot_dataset(new_dataset.iloc[int(0.4 * len(new_dataset.index)):int(0.43 * len(new_dataset.index)), :],
                             ['acc_phone_x', 'acc_phone_x_lowpass'], ['exact', 'exact'], ['line', 'line'])

    elif FLAGS.mode == 'PCA':

        # first impute again, as PCA can not deal with missing values
        for col in [c for c in dataset.columns if not 'label' in c]:
            dataset = MisVal.impute_interpolate(dataset, col)

        selected_predictor_cols = [c for c in dataset.columns if (
            not ('label' in c)) and (not (c == 'hr_watch_rate'))]
        pc_values = PCA.determine_pc_explained_variance(
            dataset, selected_predictor_cols)

        # Plot the variance explained.
        DataViz.plot_xy(x=[range(1, len(selected_predictor_cols) + 1)], y=[pc_values],
                        xlabel='principal component number', ylabel='explained variance',
                        ylim=[0, 1], line_styles=['b-'])

        # We select 7 as the best number of PC's as this explains most of the variance

        n_pcs = 7

        dataset = PCA.apply_pca(copy.deepcopy(
            dataset), selected_predictor_cols, n_pcs)

        # And we visualize the result of the PC's
        DataViz.plot_dataset(dataset, ['pca_', 'label'], [
            'like', 'like'], ['line', 'points'])

    elif FLAGS.mode == 'final':
        # Now, for the final version.
        # We first start with imputation by interpolation

        for col in [c for c in dataset.columns if not 'label' in c]:
            dataset = MisVal.impute_interpolate(dataset, col)

        # And now let us include all LOWPASS measurements that have a form of periodicity (and filter them):
        # periodic_measurements = ['acc_phone_x', 'acc_phone_y', 'acc_phone_z', 'acc_watch_x', 'acc_watch_y',
        #                          'acc_watch_z', 'gyr_phone_x', 'gyr_phone_y',
        #                          'gyr_phone_z', 'gyr_watch_x', 'gyr_watch_y', 'gyr_watch_z', 'mag_phone_x',
        #                          'mag_phone_y', 'mag_phone_z', 'mag_watch_x',
        #                          'mag_watch_y', 'mag_watch_z']
        periodic_measurements = ('Acceleration x (m/s^2),Acceleration y (m/s^2),Acceleration z (m/s^2),Gyroscope x ('
                                 'rad/s),Gyroscope y (rad/s),Gyroscope z (rad/s),Linear Acceleration x (m/s^2),'
                                 'Linear Acceleration y (m/s^2),Linear Acceleration z (m/s^2),Latitude (°),'
                                 'Longitude (°),Height (m),Velocity (m/s),Direction (°),Horizontal Accuracy (m),'
                                 'Vertical Accuracy (m)')
        periodic_measurements = periodic_measurements.split(',')
        print(periodic_measurements)

        # Let us apply a lowpass filter and reduce the importance of the data above 1.5 Hz

        # Determine the sampling frequency.
        fs = float(1000) / milliseconds_per_instance
        cutoff = 1.5

        for col in periodic_measurements:
            dataset = LowPass.low_pass_filter(
                dataset, col, fs, cutoff, order=10)
            dataset[col] = dataset[col + '_lowpass']
            del dataset[col + '_lowpass']

        # We used the optimal found parameter n_pcs = 7, to apply PCA to the final dataset

        selected_predictor_cols = [c for c in dataset.columns if (not ('label' in c)) and (not (c == 'hr_watch_rate'))]

        n_pcs = 7

        dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_predictor_cols, n_pcs)

        # And the overall final dataset:
        DataViz.plot_dataset(dataset,
                             ['Acceleration', 'Gyroscope', 'Linear Acceleration', "Height", "Velocity", "Direction",
                              'label'],
                             ['like', 'like', 'like', 'like', 'like', 'like', 'like'],
                             ['line', 'line', 'line', 'line', 'line', 'line', 'points'])

        # Store the final outcome.

        dataset.to_csv(DATA_PATH / RESULT_FNAME)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, imputation, lowpass or PCA \
                        'lowpass' applies the lowpass-filter to a single variable \
                        'imputation' is used for the next chapter \
                        'PCA' is to study the effect of PCA and plot the results\
                        'final' is used for the next chapter", choices=['lowpass', 'imputation', 'PCA', 'final'])

    FLAGS, unparsed = parser.parse_known_args()

    main()
