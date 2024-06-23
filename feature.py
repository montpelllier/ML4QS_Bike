import argparse
import copy
import time
from pathlib import Path
import pandas as pd

from Python3Code.Chapter4.FrequencyAbstraction import FourierTransformation
from Python3Code.Chapter4.TemporalAbstraction import CategoricalAbstraction
from Python3Code.Chapter4.TemporalAbstraction import NumericalAbstraction
from Python3Code.util.VisualizeDataset import VisualizeDataset

# Read the result from the previous chapter, and make sure the index is of the type datetime.
DATA_PATH = Path('./results/')
DATASET_FNAME = 'merged_dataset.csv'
RESULT_FNAME = 'dataset_result_feature.csv'
# DATASET_FNAME = 'dataset1_result_imputation.csv'
# RESULT_FNAME = 'dataset1_result_feature.csv'


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    print_flags()

    start_time = time.time()
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = dataset.index.astype(float)
    except IOError as e:
        print('File not found, try to run previous scripts first!')
        raise e

    # Let us create our visualization class again.
    DataViz = VisualizeDataset(__file__)

    # Compute the number of milliseconds covered by an instance based on the first two rows
    # 500 milliseconds per instance here
    milliseconds_per_instance = 500

    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    if FLAGS.mode == 'aggregation':
        # Chapter 4: Identifying aggregate attributes.

        # Set the window sizes to the number of instances representing 5 seconds, 30 seconds and 5 minutes
        window_sizes = [int(float(5000) / milliseconds_per_instance),
                        int(float(0.5 * 60000) / milliseconds_per_instance),
                        int(float(5 * 60000) / milliseconds_per_instance)]

        # please look in Chapter4 TemporalAbstraction.py to look for more aggregation methods or make your own.

        for ws in window_sizes:
            dataset = NumAbs.abstract_numerical(dataset, ['Acceleration x (m/s^2)'], ws, 'mean')
            dataset = NumAbs.abstract_numerical(dataset, ['Acceleration x (m/s^2)'], ws, 'std')

        DataViz.plot_dataset(dataset, ['Acceleration x (m/s^2)', 'Acceleration x (m/s^2)_temp_mean',
                                       'Acceleration x (m/s^2)_temp_std', 'labelnormal'],
                             ['exact', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))

    if FLAGS.mode == 'frequency':
        fs = float(1000) / milliseconds_per_instance
        ws = int(float(10000) / milliseconds_per_instance)
        dataset = FreqAbs.abstract_frequency(dataset, ['Acceleration x (m/s^2)'], ws, fs)
        # Spectral analysis.
        DataViz.plot_dataset(dataset, ['Acceleration x (m/s^2)_max_freq', 'Acceleration x (m/s^2)_freq_weighted',
                                       'Acceleration x (m/s^2)_pse', 'labelnormal'], ['like', 'like', 'like', 'like'],
                             ['line', 'line', 'line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))

    if FLAGS.mode == 'final':
        ws = int(float(0.5 * 3000) / milliseconds_per_instance)
        fs = float(1000) / milliseconds_per_instance
        print(f"Final mode - Window size: {ws}, Sampling frequency: {fs}")

        # selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]
        selected_predictor_cols = [
            'Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
            'Gyroscope x (rad/s)', 'Gyroscope y (rad/s)', 'Gyroscope z (rad/s)',
            'Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)'
        ]

        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')

        # Calculate the vector module of three-axis acceleration, which can reflect the overall motion intensity
        dataset['total_acceleration'] = (dataset['Acceleration x (m/s^2)'] ** 2 + dataset[
            'Acceleration y (m/s^2)'] ** 2 + dataset[
                                             'Acceleration z (m/s^2)'] ** 2) ** 0.5

        # Calculate the vector module of three-axis linear acceleration, which can reflect the overall motion intensity
        dataset['total_linear_acceleration'] = (dataset['Linear Acceleration x (m/s^2)'] ** 2 + dataset[
            'Linear Acceleration y (m/s^2)'] ** 2 + dataset['Linear Acceleration z (m/s^2)'] ** 2) ** 0.5

        # Calculate the vector module of three-axis gyroscope, which can reflect the overall motion intensity
        dataset['total_gyroscope'] = (dataset['Gyroscope x (rad/s)'] ** 2 + dataset['Gyroscope y (rad/s)'] ** 2 +
                                      dataset[
                                          'Gyroscope z (rad/s)'] ** 2) ** 0.5

        DataViz.plot_dataset(dataset, ['Acceleration x (m/s^2)', 'Gyroscope x (rad/s)', 'Linear Acceleration x (m/s^2)',
                                       'Velocity (m/s)', 'pca_1', 'labelnormal'],
                             ['like', 'like', 'like', 'like', 'like', 'like'],
                             ['line', 'line', 'line', 'line', 'line', 'points'])

        CatAbs = CategoricalAbstraction()

        # label_cols = ['labelnormal', 'labelturnright', 'labelturnleft', 'labelbrake', 'labelstop', 'labelaccelerate']
        # dataset = CatAbs.abstract_categorical(dataset, label_cols, ['like'] * len(label_cols), 0.03,
        #                                      int(float(5 * 60000) / milliseconds_per_instance), 2)

        # periodic_predictor_cols = [
        #     'Acceleration x (m/s^2)', 'Acceleration y (m/s^2)', 'Acceleration z (m/s^2)',
        #     'Gyroscope x (rad/s)', 'Gyroscope y (rad/s)', 'Gyroscope z (rad/s)',
        #     'Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)'
        # ]
        #
         # dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), periodic_predictor_cols,
        #                                      int(float(10000) / milliseconds_per_instance), fs)
        #
        # # Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.
        # # The percentage of overlap we allow
        # window_overlap = 0.9
        # skip_points = int((1 - window_overlap) * ws)
        # if skip_points == 0:
        #     skip_points = 1
        #
        # # dataset = dataset.iloc[20:, :]
        # dataset = dataset.iloc[::skip_points, :]

        missing_columns = [col for col in dataset.columns if 'temp_std_ws' in col]
        dataset[missing_columns] = dataset[missing_columns].interpolate(method='bfill', axis=0)

        dataset.index.name = 'timestamp'
        dataset.to_csv(DATA_PATH / RESULT_FNAME)

        DataViz.plot_dataset(dataset, ['Acceleration x (m/s^2)', 'Gyroscope x (rad/s)', 'Linear Acceleration x (m/s^2)',
                                       'Velocity (m/s)', 'pca_1', 'labelnormal'],
                             ['like', 'like', 'like', 'like', 'like', 'like'],
                             ['line', 'line', 'line', 'line', 'line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, aggregation or freq \
                        'aggregation' studies the effect of several aggeregation methods \
                        'frequency' applies a Fast Fourier transformation to a single variable \
                        'final' is used for the next chapter ", choices=['aggregation', 'frequency', 'final'])

    FLAGS, unparsed = parser.parse_known_args()

    main()
