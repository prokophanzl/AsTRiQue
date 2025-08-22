import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

rng = np.random.default_rng()

# ==============
# CONFIG
# ==============


@dataclass
class Config:
    # predictor columns (flexible)
    PREDICTORS: tuple[str, ...] = ("voicing", "duration")

    # column names
    FILENAME_COL: str = "filename"
    LABEL_MAPPING: dict[str, int] = field(default_factory=lambda: {"s": 0, "z": 1})
    TARGET: str = "answer_batch"

    # data paths
    DATA_PATH: str = "data/data.csv"
    AUDIO_FOLDER: str = "data/audio"
    PARTICIPANT_CSV_DIR: str = "data/participants"
    PROCESSED_PATH: str = "data_processed.csv"

    # parameters
    STRATIFIED_SAMPLING_RESOLUTION: int = 3
    MIN_ITERATIONS: int = 0
    CLEANSER_FREQUENCY: int = 0
    MODEL_CERTAINTY_CUTOFF: float = 0.95
    PARTICIPANT_TO_MODEL: str = "p01"


# ==============
# SHARED FUNCTIONS
# ==============


def set_seed(seed: int | None):
    """
    Sets the seed for the random number generator.
    """
    global rng
    rng = np.random.default_rng(seed)


def initialize_dataframe(stimuli: pd.DataFrame) -> None:
    """
    Makes sure the stimuli dataframe is correctly formatted.
    """
    # add columns for classification order, classification type, real class, predicted class, and prediction certainty if they don't exist
    for col in [
        "classification_order",
        "classification_type",
        "participant_classification",
        "predicted_class",
        "prediction_certainty",
    ]:
        if col not in stimuli.columns:
            stimuli[col] = None


def get_stratified_samples(stimuli: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Returns a dataframe with a specified number of 2D-stratified random samples from the stimuli dataframe.
    """
    predictor1, predictor2 = config.PREDICTORS[:2]
    # create 2D bins
    stimuli = stimuli.copy()
    stimuli["bin1"] = pd.cut(
        stimuli[predictor1], bins=config.STRATIFIED_SAMPLING_RESOLUTION, labels=False
    )
    stimuli["bin2"] = pd.cut(
        stimuli[predictor2], bins=config.STRATIFIED_SAMPLING_RESOLUTION, labels=False
    )

    # drop rows with NaN bins (e.g., from NaN values or empty bins)
    stimuli = stimuli.dropna(subset=["bin1", "bin2"])

    def stratified_sample(group: pd.DataFrame) -> pd.DataFrame:
        return group.sample(n=1, random_state=rng.integers(0, 1_000_000))

    # group by the 2D bin and sample one row randomly from each group
    grouped = stimuli.groupby(["bin1", "bin2"], group_keys=False)
    sampled = grouped.apply(stratified_sample)

    # drop the bin columns before returning
    return sampled.drop(columns=["bin1", "bin2"]).reset_index(drop=True)


def get_sample(
    stimuli: pd.DataFrame, iteration: int, config: Config, init_samples: int
) -> tuple[pd.DataFrame, str]:
    """
    Returns a sample from the stimuli dataframe (uncertainty sampling with cleanser).
    Takes a dataframe with only unlabeled samples and the current iteration in the active learning phase.
    """
    # check if it is time for a cleanser (the single highest-certainty) sample, otherwise select the sample with the lowest certainty
    if (
        config.CLEANSER_FREQUENCY > 0
        and (iteration - init_samples) % config.CLEANSER_FREQUENCY == 0
    ):
        print(
            f"Iteration {iteration}:\tCleanser\tCertainty: {stimuli['prediction_certainty'].max()}"
        )
        return (
            stimuli[
                stimuli["prediction_certainty"] == stimuli["prediction_certainty"].max()
            ].sample(1),
            "cleanser",
        )
    else:
        print(
            f"Iteration {iteration}: Uncertainty\tCertainty: {stimuli['prediction_certainty'].min()}"
        )
        return (
            stimuli[
                stimuli["prediction_certainty"] == stimuli["prediction_certainty"].min()
            ].sample(1),
            "uncertainty",
        )


def train_model(stimuli: pd.DataFrame, config: Config) -> LogisticRegression:
    """
    Trains a logistic regression model based on the current state of the stimuli dataframe.
    Updates predicted_class and prediction_certainty for unlabeled samples.
    """

    predictors = list(config.PREDICTORS)

    # filter to only labeled samples
    valid = stimuli["participant_classification"].isin([0, 1])
    if valid.sum() < 2:
        raise ValueError("Not enough labeled samples to train the model.")

    # features and labels
    X_train = stimuli.loc[valid, predictors]
    y_train = stimuli.loc[valid, "participant_classification"].astype(int)

    # define and train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # apply model to unlabeled data
    unknown = stimuli["participant_classification"].isna()
    if unknown.sum() == 0:
        print("No unknown samples to predict.")
        return model

    X_test = stimuli.loc[unknown, predictors]
    probs = model.predict_proba(X_test)

    # predicted class (0 or 1) and associated certainty
    predicted = model.predict(X_test)
    certainty = probs.max(axis=1)

    # store predictions in dataframe
    stimuli.loc[unknown, "predicted_class"] = predicted
    stimuli.loc[unknown, "prediction_certainty"] = certainty

    return model


def plot_results(
    stimuli: pd.DataFrame, model: LogisticRegression, plot_title: str, config: Config
) -> None:
    predictor1, predictor2 = config.PREDICTORS[:2]
    """Visualize results with decision boundary and predicted classifications for unanswered data"""

    # split answered and unanswered data
    answered_data = stimuli[stimuli["participant_classification"].notna()]
    unanswered_data = stimuli[stimuli["participant_classification"].isna()]

    # set font
    # plt.rcParams['font.family'] = ['Arial', 'sans-serif']

    plt.figure(figsize=(10, 6), dpi=300)

    # convert answers to numeric if necessary
    if answered_data["participant_classification"].dtype == "object":
        answered_data = answered_data.copy()
        answered_data["participant_classification"] = answered_data[
            "participant_classification"
        ].astype(int)

    # plot answered points, split by class
    for label_char, label_num in config.LABEL_MAPPING.items():
        subset = answered_data[answered_data["participant_classification"] == label_num]
        if not subset.empty:
            plt.scatter(
                subset[predictor1],
                subset[predictor2],
                c="blue" if label_num == 0 else "red",
                label=f"Answered ({label_char})",
                edgecolors="k",
            )

    # plot unanswered points, split by predicted class
    if not unanswered_data.empty:
        for label_num, color in zip([0, 1], ["blue", "red"]):
            predicted_subset = unanswered_data[
                unanswered_data["predicted_class"] == label_num
            ]
            if not predicted_subset.empty:
                label_char = [
                    k for k, v in config.LABEL_MAPPING.items() if v == label_num
                ][0]
                plt.scatter(
                    predicted_subset[predictor1],
                    predicted_subset[predictor2],
                    c=color,
                    alpha=0.3,
                    label=f"Predicted ({label_char})",
                )

    # decision boundary grid
    x_min, x_max = stimuli[predictor1].min() - 1, stimuli[predictor1].max() + 1
    y_min, y_max = stimuli[predictor2].min() - 1, stimuli[predictor2].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    grid_points = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()], columns=[predictor1, predictor2]
    )

    Z = model.predict_proba(grid_points)[:, 1]
    Z = Z.reshape(xx.shape)

    # show background decision gradient
    contour = plt.contourf(xx, yy, Z, alpha=0.3, levels=20, cmap="coolwarm")

    # draw n by n grid overlay
    step_x = (x_max - x_min) / config.STRATIFIED_SAMPLING_RESOLUTION
    step_y = (y_max - y_min) / config.STRATIFIED_SAMPLING_RESOLUTION

    for i in range(config.STRATIFIED_SAMPLING_RESOLUTION + 1):
        plt.axvline(x_min + i * step_x, color="gray", linestyle="--", linewidth=0.5)
        plt.axhline(y_min + i * step_y, color="gray", linestyle="--", linewidth=0.5)

    # custom color bar with labels s and z
    cbar = plt.colorbar(contour, ticks=[0, 1])
    rev_label_map = {v: k for k, v in config.LABEL_MAPPING.items()}
    cbar.ax.set_yticklabels([rev_label_map[0], rev_label_map[1]])
    cbar.set_label("Predicted Answer")

    plt.xlabel(predictor1)
    plt.ylabel(predictor2)
    plt.title(plot_title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_model(
    stimuli: pd.DataFrame, query_participant_classification: callable, config: Config
) -> pd.DataFrame:
    """
    Evaluate model predictions on the unanswered data by comparing them to real labels
    obtained via query_participant_classification(). The true labels are saved into
    the 'reference_participant_classification' column of the stimuli dataframe.
    """
    # create 'reference_participant_classification' column and set it to 'participant_classification' values
    stimuli["reference_participant_classification"] = stimuli[
        "participant_classification"
    ]

    # select rows with missing participant classification
    unanswered_mask = stimuli["reference_participant_classification"].isna()
    unanswered = stimuli[unanswered_mask].copy()

    if unanswered.empty:
        print("No unanswered data to evaluate.")
        return stimuli  # Return original unchanged DataFrame

    # prepare lists for true and predicted labels
    true_labels = []
    predicted_labels = unanswered["predicted_class"].tolist()

    print("Evaluating model predictions on unanswered data...")

    true_label_values = []
    for idx, row in unanswered.iterrows():
        filename = row[config.FILENAME_COL]
        print(
            f"Evaluating sound {list(unanswered.index).index(idx) + 1} out of {len(unanswered)}"
        )
        true_label = int(query_participant_classification(filename))
        true_labels.append(true_label)
        true_label_values.append(true_label)

    # batch update the main DataFrame
    stimuli.loc[unanswered.index, "reference_participant_classification"] = (
        true_label_values
    )

    # compute and print evaluation metrics
    acc = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    report = classification_report(true_labels, predicted_labels)

    print("\n=== Evaluation on Unanswered Data ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return stimuli


def export_data(stimuli: pd.DataFrame, config: Config) -> None:
    """
    Exports the stimuli dataframe to a CSV file if the path is provided.
    """

    if not config.PROCESSED_PATH:
        print("Processed data not saved - PROCESSED_PATH is empty")
        return

    # in all rows where participant_classification is known, remove the predicted_class and prediction_certainty values
    stimuli.loc[
        ~stimuli["participant_classification"].isna(),
        ["predicted_class", "prediction_certainty"],
    ] = np.nan

    # save dataframe
    stimuli.to_csv(config.PROCESSED_PATH, index=False)
    print(f"Processed data saved to {config.PROCESSED_PATH}")
