import os
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

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

    # virtual agent setup
    USE_VIRTUAL_AGENT: bool = True
    PARTICIPANT_CSV_DIR: str = "data/participants"  # only for virtual agent mode
    PROCESSED_PATH: str = "data_processed.csv"  # only for virtual agent mode

    # data paths
    DATA_PATH: str = "data/data.csv"
    AUDIO_FOLDER: str = "data/audio"  # only for live participant mode

    # parameters
    STRATIFIED_SAMPLING_RESOLUTION: int = 3
    MIN_ITERATIONS: int = 0
    MAX_ITERATIONS: int = (
        0  # only taken into account after initial model is built (after stratified sampling + class diversity-ensuring random sampling)
    )
    CLEANSER_FREQUENCY: int = 0
    MODEL_CERTAINTY_CUTOFF: float = 0.95
    PARTICIPANT_TO_MODEL: str = "p01"

    # miscellaneous
    DEBUG_MODE: bool = False
    RNG_SEED: int = 42  # None for random seed


# ==============
# SHARED FUNCTIONS
# ==============


def print_debug(print_value: str, config: Config) -> None:
    """
    Prints a debug message if DEBUG_MODE is True.
    """

    if config.DEBUG_MODE:
        print(print_value)


def validate_config(config: Config) -> None:
    """
    Validates a Config object and logs warnings for non-fatal issues.
    Raises ValueError for fatal configuration problems.
    """

    # --- DATA_PATH ---
    if not os.path.isfile(config.DATA_PATH):
        raise ValueError(f"DATA_PATH file does not exist: {config.DATA_PATH}")

    # load CSV to check columns and row count
    df = pd.read_csv(config.DATA_PATH)
    total_rows = len(df)

    required_columns = [config.FILENAME_COL] + list(config.PREDICTORS)
    missing_cols = [col for col in required_columns if col not in df.columns]
    if len(missing_cols) == 1:
        raise ValueError(f"DATA_PATH CSV is missing a required column: {missing_cols}")
    elif len(missing_cols) > 1:
        raise ValueError(
            f"DATA_PATH CSV is missing required columns: {', '.join(missing_cols)}"
        )

    # --- LABEL_MAPPING ---
    if not isinstance(config.LABEL_MAPPING, dict):
        raise ValueError("LABEL_MAPPING is not in the correct format (dict).")
    if set(config.LABEL_MAPPING.values()) != {0, 1}:
        raise ValueError(
            "LABEL_MAPPING must assign values 0 and 1 to two distinct labels."
        )
    if len(config.LABEL_MAPPING) != 2:
        raise ValueError("LABEL_MAPPING must have exactly two entries.")

    # --- string parameters ---
    string_params = [
        "TARGET",
        "FILENAME_COL",
        "PROCESSED_PATH",
    ]

    for param in string_params:
        value = getattr(config, param)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{param} must be a non-empty string.")

    # --- integer parameters ---
    numeric_params = [
        "STRATIFIED_SAMPLING_RESOLUTION",
        "MIN_ITERATIONS",
        "MAX_ITERATIONS",
        "CLEANSER_FREQUENCY",
    ]

    for param in numeric_params:
        value = getattr(config, param)
        if not isinstance(value, int):
            raise ValueError(f"{param} must be an integer.")
        if value < 0:
            raise ValueError(f"{param} cannot be negative.")

    # --- boolean parameters ---
    bool_params = [
        "DEBUG_MODE",
    ]

    for param in bool_params:
        value = getattr(config, param)
        if not isinstance(value, bool):
            raise ValueError(f"{param} must be a boolean.")

    print_debug("[INFO]\tConfig validation passed.", config)

    # --- MIN_ITERATIONS ---
    if config.MIN_ITERATIONS > total_rows:
        raise ValueError(
            f"MIN_ITERATIONS ({config.MIN_ITERATIONS}) cannot exceed the total number of stimuli available ({total_rows})."
        )

    # --- MAX_ITERATIONS ---
    if config.MAX_ITERATIONS != 0 and config.MAX_ITERATIONS < config.MIN_ITERATIONS:
        raise ValueError(
            f"MAX_ITERATIONS ({config.MAX_ITERATIONS}) cannot be smaller than MIN_ITERATIONS ({config.MIN_ITERATIONS})."
        )

    # --- CLEANSER_FREQUENCY ---
    if config.CLEANSER_FREQUENCY > total_rows:
        warnings.warn(
            f"CLEANSER_FREQUENCY ({config.CLEANSER_FREQUENCY}) exceeds the total number of stimuli available ({total_rows}). The cleanser will never trigger."
        )

    # --- MODEL_CERTAINTY_CUTOFF ---
    if not (
        isinstance(config.MODEL_CERTAINTY_CUTOFF, (int, float))
        and 0 <= config.MODEL_CERTAINTY_CUTOFF <= 1
    ):
        raise ValueError("MODEL_CERTAINTY_CUTOFF must be a number between 0 and 1")
    if config.MODEL_CERTAINTY_CUTOFF == 1:
        warnings.warn(
            "MODEL_CERTAINTY_CUTOFF = 1. The oracle will need to answer every stimulus; AsTRiQue offers no advantage."
        )

    # --- participant workflow ---
    if not config.USE_VIRTUAL_AGENT:
        if not os.path.isdir(config.AUDIO_FOLDER):
            raise ValueError(f"AUDIO_FOLDER does not exist: {config.AUDIO_FOLDER}.")

        missing_files = [
            f
            for f in df[config.FILENAME_COL]
            if not os.path.isfile(os.path.join(config.AUDIO_FOLDER, f))
        ]
        if missing_files:
            raise ValueError(f"AUDIO_FOLDER is missing these stimuli: {missing_files}.")

    # --- virtual agent workflow ---
    if config.USE_VIRTUAL_AGENT:
        if not os.path.isdir(config.PARTICIPANT_CSV_DIR):
            raise ValueError(
                f"PARTICIPANT_CSV_DIR does not exist: {config.PARTICIPANT_CSV_DIR}."
            )

        agent_csv_path = os.path.join(
            config.PARTICIPANT_CSV_DIR, f"{config.PARTICIPANT_TO_MODEL}.csv"
        )
        if not os.path.isfile(agent_csv_path):
            raise ValueError(f"Virtual agent CSV does not exist: {agent_csv_path}.")

        agent_df = pd.read_csv(agent_csv_path)
        required_agent_cols = [config.FILENAME_COL, config.TARGET] + list(
            config.PREDICTORS
        )
        missing_agent_cols = [
            col for col in required_agent_cols if col not in agent_df.columns
        ]
        if missing_agent_cols:
            raise ValueError(
                f"Virtual agent CSV is missing required columns: {missing_agent_cols}."
            )

    # --- miscellaneous ---

    if not (isinstance(config.RNG_SEED, int) or config.RNG_SEED == None):
        raise ValueError("RNG_SEED must be an integer or None.")


def query_virtual_agent_classification(filename: str, config: Config = None) -> int:
    """
    Queries a virtual agent for a classification of a given sample.
    """
    # look into the participant's answer lookup table - PARTICIPANT_CSV_DIR/PARTICIPANT_TO_MODEL.csv and return the real class based on LABEL_MAPPING
    participant_answers = pd.read_csv(
        config.PARTICIPANT_CSV_DIR + "/" + config.PARTICIPANT_TO_MODEL + ".csv"
    )
    real_answer = participant_answers[
        participant_answers[config.FILENAME_COL] == filename
    ][config.TARGET].values[0]
    return config.LABEL_MAPPING[real_answer]


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
    Returns a dataframe with a specified number of nD-stratified random samples
    from the stimuli dataframe, where n = len(config.PREDICTORS).
    Rows are returned in a randomized order based on config.RNG_SEED.
    """

    stimuli = stimuli.copy()
    bin_cols = []

    # create bins for each predictor
    for i, predictor in enumerate(config.PREDICTORS):
        bin_col = f"bin{i}"
        stimuli[bin_col] = pd.cut(
            stimuli[predictor], bins=config.STRATIFIED_SAMPLING_RESOLUTION, labels=False
        )
        bin_cols.append(bin_col)

    # drop rows with NaN bins
    stimuli = stimuli.dropna(subset=bin_cols)

    def stratified_sample(group: pd.DataFrame) -> pd.DataFrame:
        return group.sample(1, random_state=config.RNG_SEED)

    # group by all bin columns and sample one row per group
    grouped = stimuli.groupby(bin_cols, group_keys=False)
    sampled = grouped.apply(stratified_sample)

    # drop bin columns
    sampled = sampled.drop(columns=bin_cols).reset_index(drop=True)

    # shuffle rows
    sampled = sampled.sample(frac=1, random_state=config.RNG_SEED).reset_index(
        drop=True
    )

    return sampled


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
        sample = stimuli[
            stimuli["prediction_certainty"] == stimuli["prediction_certainty"].max()
        ].sample(1, random_state=config.RNG_SEED)
        filename = sample["filename"].values[0]

        print_debug(
            f"[{iteration}]\tCleanser sample: {filename} (certainty: {sample['prediction_certainty'].values[0]})",
            config,
        )
        return (
            sample,
            "cleanser_sample",
        )
    else:
        sample = stimuli[
            stimuli["prediction_certainty"] == stimuli["prediction_certainty"].min()
        ].sample(1, random_state=config.RNG_SEED)
        filename = sample["filename"].values[0]

        print_debug(
            f"[{iteration}]\tUncertainty sampling: {filename} (certainty: {sample['prediction_certainty'].values[0]})",
            config,
        )
        return (
            sample,
            "uncertainty_sampling",
        )


def evaluate_model(
    stimuli: pd.DataFrame, query_oracle_classification: callable, config: Config
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
        print_debug("[INFO]\tNo unanswered data to evaluate.", config)
        return stimuli  # Return original unchanged DataFrame

    # prepare lists for true and predicted labels
    true_labels = []
    predicted_labels = unanswered["predicted_class"].tolist()

    print_debug("[INFO]\tEvaluating model predictions on unanswered data...", config)

    # query the oracle for all remaining classifications
    true_label_values = []
    for idx, row in unanswered.iterrows():
        filename = row[config.FILENAME_COL]
        print_debug(
            f"[{list(unanswered.index).index(idx) + 1}/{len(unanswered)}]\tEvaluation: {filename}",
            config,
        )
        true_label = int(query_oracle_classification(filename, config))
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


def export_data(stimuli: pd.DataFrame, config: Config) -> None:
    """
    Exports the stimuli dataframe to a CSV file if the path is provided.
    """

    # in all rows where participant_classification is known, remove the predicted_class and prediction_certainty values
    stimuli.loc[
        ~stimuli["participant_classification"].isna(),
        ["predicted_class", "prediction_certainty"],
    ] = np.nan

    # save dataframe
    stimuli.to_csv(config.PROCESSED_PATH, index=False)
    print_debug(f"Processed data saved to {config.PROCESSED_PATH}", config)


def run(
    config: Config,
    query_oracle_classification: callable = None,
) -> pd.DataFrame:

    # =======================
    # PREPARE ENVIRONMENT
    # =======================

    # validate configuration
    validate_config(config)

    # set up oracle query function; either virtual agent or participant
    if config.USE_VIRTUAL_AGENT:
        query_oracle_classification = query_virtual_agent_classification
    elif (
        query_oracle_classification == None
    ):  # if virtual agent is not used and no query function is provided
        raise ValueError("A query function must be provided.")

    # load and initialize the stimuli dataframe
    stimuli = pd.read_csv(config.DATA_PATH)
    initialize_dataframe(stimuli)

    # print runtime info
    if config.USE_VIRTUAL_AGENT:
        mode_name = "virtual agent mode"
    else:
        mode_name = "live participant mode"
    total_stimuli = len(stimuli)
    print_debug(
        f"[INFO]\tStarting runtime in {mode_name} with {total_stimuli} total stimuli.",
        config,
    )

    # =======================
    # INITIAL STRATIFIED SAMPLING PHASE WITH CLASS BALANCE
    # =======================

    iteration = 0

    collected_classes = set()  # set of classes that have been collected

    # get stratified samples (already in randomized order)
    stratified_samples = get_stratified_samples(stimuli, config)

    # get oracle classification for each of the selected samples
    for _, sample in stratified_samples.iterrows():
        iteration += 1

        filename = sample[config.FILENAME_COL]
        print_debug(f"[{iteration}]\tStratified sampling: {filename}", config)

        # get classification by querying with filename
        filename = sample[config.FILENAME_COL]
        classification = int(query_oracle_classification(filename, config))

        collected_classes.add(classification)

        # update row in dataframe
        idx = stimuli[config.FILENAME_COL] == filename
        stimuli.loc[idx, "classification_order"] = iteration
        stimuli.loc[idx, "participant_classification"] = classification
        stimuli.loc[idx, "classification_type"] = "stratified_sampling"

    # ensure class diversity
    while len(collected_classes) < 2:
        iteration += 1

        # select a random stimulus where real class is unknown using the RNG seed
        sample = stimuli[stimuli["participant_classification"].isna()].sample(
            1, random_state=config.RNG_SEED
        )

        filename = sample[config.FILENAME_COL].values[0]

        print_debug(
            f"[{iteration}]\tRandom sampling (ensuring class diversity): {filename}",
            config,
        )

        # get classification, querying filename
        classification = int(
            query_oracle_classification(sample[config.FILENAME_COL].values[0], config)
        )

        collected_classes.add(classification)

        # update row in dataframe
        idx = stimuli[config.FILENAME_COL] == sample[config.FILENAME_COL].values[0]
        stimuli.loc[idx, "classification_order"] = iteration
        stimuli.loc[idx, "classification_type"] = "random_sampling_class_balance"
        stimuli.loc[idx, "participant_classification"] = classification

    init_samples = iteration

    # train initial model
    model = train_model(stimuli, config)

    # =======================
    # MAIN ACTIVE LEARNING LOOP
    # =======================

    while not (config.MAX_ITERATIONS > 0 and iteration >= config.MAX_ITERATIONS):
        iteration += 1

        # get updated unanswered subset and check stopping criteria

        unanswered = stimuli[stimuli["participant_classification"].isna()]

        below_cutoff = (
            unanswered["prediction_certainty"] < config.MODEL_CERTAINTY_CUTOFF
        )

        if below_cutoff.sum() == 0 and iteration > config.MIN_ITERATIONS:
            print_debug(
                "[STOP]\tAll predictions above certainty threshold "
                f"({config.MODEL_CERTAINTY_CUTOFF}) and minimum iterations met ({config.MIN_ITERATIONS}).",
                config,
            )
            break

        # select next sample using uncertainty sampling (with optional cleanser)
        sample, sample_type = get_sample(unanswered, iteration, config, init_samples)

        # query real classification
        classification = int(
            query_oracle_classification(sample[config.FILENAME_COL].values[0], config)
        )

        # update row in dataframe
        idx = stimuli[config.FILENAME_COL] == sample[config.FILENAME_COL].values[0]
        stimuli.loc[idx, "classification_order"] = iteration
        stimuli.loc[idx, "classification_type"] = sample_type
        stimuli.loc[idx, "participant_classification"] = classification

        # retrain model to get up-to-date predictions on remaining unlabeled samples
        model = train_model(stimuli, config)

        # check if MAX_ITERATIONS has been reached
        if config.MAX_ITERATIONS > 0 and iteration >= config.MAX_ITERATIONS:
            print_debug(
                f"[INFO]\tMax iterations reached ({config.MAX_ITERATIONS}).", config
            )
            break

    # evaluate model
    evaluate_model(stimuli, query_oracle_classification, config)

    # plot the results if desired
    if config.DEBUG_MODE:
        if config.USE_VIRTUAL_AGENT:
            plot_title = (
                f"AsTRiQue Results (Virtual Agent: {config.PARTICIPANT_TO_MODEL})"
            )
        else:
            plot_title = f"AsTRiQue Results (Live Participant Mode)"

        plot_results(stimuli, model, plot_title, config)

    # export data if desired
    export_data(stimuli, config)
