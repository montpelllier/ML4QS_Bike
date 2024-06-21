import copy
from pathlib import Path

from Python3Code.Chapter2.CreateDataset import CreateDataset
from Python3Code.util import util
from Python3Code.util.VisualizeDataset import VisualizeDataset

# Set up the data file path.
DATASET_PATH = Path('./datasets/1/')
# Set up the result file path.
RESULT_PATH = Path('./results/')
# Set up the result filename.
RESULT_FNAME = 'dataset1.csv'

# Define the granularity of the data
# GRANULARITIES = [5000, 500]
GRANULARITIES = [500]
# Create the result path if it doesn't exist
[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]

# Define the columns that we are going to parse
datasets = []
directions = ['x', 'y', 'z']
acc_cols = [f'Acceleration {v} (m/s^2)' for v in directions]
gyr_cols = [f'Gyroscope {v} (rad/s)' for v in directions]
lacc_cols = [f'Linear Acceleration {v} (m/s^2)' for v in directions]
gps_cols = ["Latitude (°)", "Longitude (°)", "Height (m)", "Velocity (m/s)", "Direction (°)", "Horizontal Accuracy (m)",
            "Vertical Accuracy (m)"]

for milliseconds_per_instance in GRANULARITIES:
    print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

    # Create an initial dataset object with the base directory for our data and a granularity
    dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)
    time_col = 'Time (s)'

    dataset.add_numerical_dataset('Accelerometer.csv', time_col, acc_cols, 'avg', '')
    dataset.add_numerical_dataset('Gyroscope.csv', time_col, gyr_cols, 'avg', '')
    dataset.add_numerical_dataset('Linear Acceleration.csv', time_col, lacc_cols, 'avg', '')
    dataset.add_numerical_dataset('Location.csv', time_col, gps_cols, 'avg', '')
    dataset.add_event_dataset('label.csv', 'label_start', 'label_end', 'label', 'binary')

    # Get the resulting pandas data table
    dataset = dataset.data_table

    # Plot the data
    DataViz = VisualizeDataset(__file__)

    # Boxplot
    DataViz.plot_dataset_boxplot(dataset, acc_cols)

    # Plot all data
    # DataViz.plot_dataset(dataset, ['acc_', 'gyr_', 'hr_watch_rate', 'light_phone_lux', 'mag_', 'press_phone_', 'label'],
    #                      ['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'],
    #                      ['line', 'line', 'line', 'line', 'line', 'line', 'points', 'points'])
    DataViz.plot_dataset(dataset,
                         ['Acceleration', 'Gyroscope', 'Linear Acceleration', "Height", "Velocity", "Direction",
                          'label'],
                         ['like', 'like', 'like', 'like', 'like', 'like', 'like'],
                         ['line', 'line', 'line', 'line', 'line', 'line', 'points'])
    # And print a summary of the dataset.
    print(dataset.columns)
    print(dataset.dtypes)
    print(dataset.describe())
    util.print_statistics(dataset)
    datasets.append(copy.deepcopy(dataset))

    # If needed, we could save the various versions of the dataset we create in the loop with logical filenames:
    # dataset.to_csv(RESULT_PATH / f'chapter2_result_{milliseconds_per_instance}')

# Make a table like the one shown in the book, comparing the two datasets produced.
# util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])

# Finally, store the last dataset we generated (250 ms).
dataset.to_csv(RESULT_PATH / RESULT_FNAME)

# Lastly, print a statement to know the code went through
print('The code has run through successfully!')
