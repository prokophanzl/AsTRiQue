import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

# ==============
# DEFAULT CONFIG
# ==============

# PREDICTOR1 = 'voicing'                    # first predictor column name
# PREDICTOR2 = 'duration'                   # second predictor column name
# FILENAME_COL = 'filename'                 # filename column name
# LABEL_MAPPING = {'s': 0, 'z': 1}          # binary output label mapping

# TARGET = 'answer_batch'                   # target column name
# DATA_PATH = 'data/data.csv'               # sound info data file path
# PARTICIPANT_CSV_DIR = 'data/participants' # participant CSV directory
# PROCESSED_PATH = 'data_processed.csv'     # processed data file path; empty string to disable
# AUDIO_FOLDER = 'data/audio'               # audio file directory

# INIT_RANDOM_SAMPLES = 10                  # initial random samples to collect
# MIN_ITERATIONS = 0                        # minimum number of iterations
# CLEANSER_FREQUENCY = 0                    # insert a high-certainty sample every nth iteration to prevent participant fatigue (irrelevant for virtual agents); 0 to disable
# MODEL_CERTAINTY_CUTOFF = 0.95             # stopping certainty threshold
# PARTICIPANT_TO_MODEL = 'p01'              # participant ID to simulate

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

def get_sample(stimuli, iteration, cleanser_frequency, random_samples):
    """
    Returns a sample from the stimuli dataframe (uncertainty sampling with cleanser).
    Takes a dataframe with only unlabeled samples and the current iteration in the active learning phase.
    """
    # check if it is time for a cleanser (the single highest-certainty) sample, otherwise select the sample with the lowest certainty
    if cleanser_frequency > 0 and (iteration - random_samples) % cleanser_frequency == 0:
        print(f"Iteration {iteration}:\tCleanser\tCertainty: {stimuli['prediction_certainty'].max()}")
        return stimuli[stimuli['prediction_certainty'] == stimuli['prediction_certainty'].max()].sample(1), 'cleanser'
    else:
        print(f"Iteration {iteration}: Uncertainty\tCertainty: {stimuli['prediction_certainty'].min()}")
        return stimuli[stimuli['prediction_certainty'] == stimuli['prediction_certainty'].min()].sample(1), 'uncertainty'

def train_model(stimuli, predictor1, predictor2):
    """
    Trains a logistic regression model based on the current state of the stimuli dataframe.
    Updates predicted_class and prediction_certainty for unlabeled samples.
    """

    # filter to only labeled samples
    valid = stimuli['participant_classification'].isin([0, 1])
    if valid.sum() < 2:
        raise ValueError("Not enough labeled samples to train the model.")

    # features and labels
    X_train = stimuli.loc[valid, [predictor1, predictor2]]
    y_train = stimuli.loc[valid, 'participant_classification'].astype(int)

    # define and train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # apply model to unlabeled data
    unknown = stimuli['participant_classification'].isna()
    if unknown.sum() == 0:
        print("No unknown samples to predict.")
        return model

    X_test = stimuli.loc[unknown, [predictor1, predictor2]]
    probs = model.predict_proba(X_test)

    # predicted class (0 or 1) and associated certainty
    predicted = model.predict(X_test)
    certainty = probs.max(axis=1)

    # store predictions in dataframe
    stimuli.loc[unknown, 'predicted_class'] = predicted
    stimuli.loc[unknown, 'prediction_certainty'] = certainty

    return model

def plot_results(stimuli, model, plot_title, predictor1, predictor2, label_mapping):
    """Visualize results with decision boundary and improved legend/color bar"""
    
    # split answered and unanswered data
    answered_data = stimuli[stimuli['participant_classification'].notna()]
    unanswered_data = stimuli[stimuli['participant_classification'].isna()]

    plt.figure(figsize=(10, 6), dpi=300)
    
    # convert answers to numeric if necessary
    if answered_data['participant_classification'].dtype == 'object':
        answered_data = answered_data.copy()
        answered_data['participant_classification'] = answered_data['participant_classification'].astype(int)
        
    # plot answered points, split by class
    for label_char, label_num in label_mapping.items():
        subset = answered_data[answered_data['participant_classification'] == label_num]
        if not subset.empty:
            plt.scatter(
                subset[predictor1],
                subset[predictor2],
                c='blue' if label_num == 0 else 'red',
                label=f"answered ({label_char})",
                edgecolors='k'
            )

    # plot unanswered points
    if not unanswered_data.empty:
        plt.scatter(
            unanswered_data[predictor1], 
            unanswered_data[predictor2],
            c='gray',
            alpha=0.5,
            label='unanswered'
        )
    
    # decision boundary grid
    x_min, x_max = stimuli[predictor1].min() - 1, stimuli[predictor1].max() + 1
    y_min, y_max = stimuli[predictor2].min() - 1, stimuli[predictor2].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    grid_points = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()],
        columns=[predictor1, predictor2]
    )
    
    Z = model.predict_proba(grid_points)[:, 1]
    Z = Z.reshape(xx.shape)
    
    # show background decision gradient
    contour = plt.contourf(xx, yy, Z, alpha=0.3, levels=20, cmap='coolwarm')
    
    # custom color bar with labels s and z
    cbar = plt.colorbar(contour, ticks=[0, 1])
    rev_label_map = {v: k for k, v in label_mapping.items()}
    cbar.ax.set_yticklabels([rev_label_map[0], rev_label_map[1]])
    cbar.set_label('predicted answer')
    
    plt.xlabel(predictor1)
    plt.ylabel(predictor2)
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_model(stimuli, filename_col, query_participant_classification):
    """
    Evaluate model predictions on the unanswered data by comparing them to real labels
    obtained via query_participant_classification().
    """
    # get unanswered data
    unanswered = stimuli[stimuli['participant_classification'].isna()].copy()
    
    if unanswered.empty:
        print("No unanswered data to evaluate.")
        return

    # query the actual class for evaluation
    true_labels = []
    predicted_labels = unanswered['predicted_class'].tolist()

    print("Evaluating model predictions on unanswered data...")

    for filename in unanswered[filename_col]:
        # print which sound is being evaluated out of how many - count filenames from 1
        print(f"Evaluating sound {unanswered[filename_col].tolist().index(filename) + 1} out of {len(unanswered[filename_col])}")
        true_label = int(query_participant_classification(filename))
        true_labels.append(true_label)

    # calculate metrics
    acc = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels)

    print("\n=== Evaluation on Unanswered Data ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

def export_data(stimuli, path):
    """
    Exports the stimuli dataframe to a CSV file if the path is provided.
    """

    if not path:
        print("Processed data not saved - PROCESSED_PATH is empty")
        return

    # in all rows where participant_classification is known, remove the predicted_class and prediction_certainty values
    stimuli.loc[~stimuli['participant_classification'].isna(), ['predicted_class', 'prediction_certainty']] = np.nan

    # save dataframe
    stimuli.to_csv(path, index = False)
    print(f"Processed data saved to {path}")
