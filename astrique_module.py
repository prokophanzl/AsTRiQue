import numpy as np
from sklearn.linear_model import LogisticRegression

# ==============
# DEFAULT CONFIG
# ==============

PREDICTOR1 = 'voicing'                    # first predictor column name
PREDICTOR2 = 'duration'                   # second predictor column name
FILENAME_COL = 'filename'                 # filename column name
LABEL_MAPPING = {'s': 0, 'z': 1}          # binary output label mapping

TARGET = 'answer_batch'                   # target column name
DATA_PATH = 'data/data.csv'               # sound info data file path
PARTICIPANT_CSV_DIR = 'data/participants' # participant CSV directory
PROCESSED_PATH = 'data_processed.csv'     # processed data file path; leave blank to disable

INIT_RANDOM_SAMPLES = 10                  # initial random samples to collect
MIN_ITERATIONS = 0                        # minimum number of iterations
CLEANSER_FREQUENCY = 0                    # insert a high-certainty sample every nth iteration to prevent participant fatigue (irrelevant for virtual agents); 0 to disable
MODEL_CERTAINTY_CUTOFF = 0.95             # stopping certainty threshold
PARTICIPANT_TO_MODEL = 'p01'              # participant ID to simulate

# ==============
# SHARED FUNCTIONS
# ==============

def initialize_dataframe(stimuli):
    """
    Makes sure the stimuli dataframe is correctly formatted.
    """
    # add columns for classification order, classification type, real class, predicted class, and prediction certainty if they don't exist
    for col in ['classification_order', 'classification_type', 'participant_classification', 'predicted_class', 'prediction_certainty']:
        if col not in stimuli.columns:
            stimuli[col] = None

def get_sample(stimuli, iteration, active_learning_iteration):
    """
    Returns a sample from the stimuli dataframe (uncertainty sampling with cleanser).
    Takes a dataframe with only unlabeled samples and the current iteration in the active learning phase.
    """

    # check if it is time for a cleanser (the single highest-certainty) sample, otherwise select the sample with the lowest certainty
    if CLEANSER_FREQUENCY > 0 and active_learning_iteration % CLEANSER_FREQUENCY == 0:
        print(f"Iteration {iteration}: Cleanser (AL {active_learning_iteration})")
        # return stimuli[stimuli['prediction_certainty'] == stimuli['prediction_certainty'].max()].sample(1)
        # include "cleanser" in the return statement
        return stimuli[stimuli['prediction_certainty'] == stimuli['prediction_certainty'].max()].sample(1), 'cleanser'
    else:
        print(f"Iteration {iteration}: Uncertainty sampling (AL {active_learning_iteration})")
        return stimuli[stimuli['prediction_certainty'] == stimuli['prediction_certainty'].min()].sample(1), 'uncertainty'

def train_model(stimuli):
    """
    Trains a logistic regression model based on the current state of the stimuli dataframe.
    Updates predicted_class and prediction_certainty for unlabeled samples.
    """

    # filter to only labeled samples
    valid = stimuli['participant_classification'].isin([0, 1])
    if valid.sum() < 2:
        raise ValueError("Not enough labeled samples to train the model.")

    # features and labels
    X_train = stimuli.loc[valid, [PREDICTOR1, PREDICTOR2]]
    y_train = stimuli.loc[valid, 'participant_classification'].astype(int)

    # define and train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # apply model to unlabeled data
    unknown = stimuli['participant_classification'].isna()
    if unknown.sum() == 0:
        print("No unknown samples to predict.")
        return model

    X_test = stimuli.loc[unknown, [PREDICTOR1, PREDICTOR2]]
    probs = model.predict_proba(X_test)

    # predicted class (0 or 1) and associated certainty
    predicted = model.predict(X_test)
    certainty = probs.max(axis=1)

    # store predictions in dataframe
    stimuli.loc[unknown, 'predicted_class'] = predicted
    stimuli.loc[unknown, 'prediction_certainty'] = certainty

    return model

def export_data(stimuli, path):
    """
    Exports the stimuli dataframe to a CSV file.
    """
    # in all rows where participant_classification is known, remove the predicted_class and prediction_certainty values
    stimuli.loc[~stimuli['participant_classification'].isna(), ['predicted_class', 'prediction_certainty']] = np.nan

    # save dataframe
    stimuli.to_csv(path, index = False)
    print(f"Processed data saved to {path}")
