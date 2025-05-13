# AsTRiQue
#### **AS**k **T**he **RI**ght **QUE**stions: An active machine learning framework for perception experiments

AsTRiQue is a tool designed to streamline perception experiments that involve large amounts of stimuli. In traditional perception experiments, participants often have to classify a large number of items, which can be time-consuming and tiring. AsTRiQue minimizes this burden through active learning; a kind of machine learning where a model is continuously updated based on the participant’s input to decide what to ask next.

The experiment starts by randomly selecting a few stimuli for the participant ("oracle") to classify. Based on the answers, a logistic regression model is trained to predict how the participant might respond to all the remaining stimuli. Then, the model picks the stimulus it is most uncertain about (uncertainty sampling). By targeting these hard-to-predict cases, the model learns much faster.

To keep things balanced, there is also an option to occasionally include a very easy (high-certainty) stimulus as a “cleanser” to reduce participant fatigue from repetitive difficult stimuli.

This process repeats: the participant answers, the model updates, and a new uncertain stimulus is selected. Once the model is confident enough in its predictions of all the stimuli the participant hasn't been presented with (based on a pre-determined confidence value), the experiment ends.


## 🧑🏾 Human Workflow Showcase

If you'd like to see AsTRiQue in action with yourself as the participant, you can run the `participant.ipynb` notebook locally or check out the [showcase notebook online](https://colab.research.google.com/github/prokophanzl/AsTRiQue/blob/main/colab-showcase/participant_showcase.ipynb).

## 🤖 Virtual Agent Showcase
Not feeling like doing the perception experiment yourself? You can still see AsTRiQue in action by running the `virtual_agent.ipynb` notebook locally or checking out the [virtual agent showcase notebook online](https://colab.research.google.com/github/prokophanzl/AsTRiQue/blob/main/colab-showcase/virtual_agent_showcase.ipynb). Instead of querying you as the oracle, the model will query a lookup table of a real participant who took part in the original experiment.

### 📊 Dataset
The showcase makes use of data from Bořil (YEAR), where he investigated the categorization of Czech sibilants /s/ vs. /z/ and /ʃ/ vs. /ʒ/ as a function of two acoustic parameters: voicing (quantified as the percentage of the segment exhibiting periodic vocal fold vibration) and segmental duration (in ms). For the purposes of the showcase, /s/ and /ʃ/ were batched together, as were /z/ and /ʒ/.

### 🗂️ Data Structure
#### 📁 data/data.csv
This spreadsheet contains the filenames of all recordings used in the experiment, as well as their parameters (voicing and duration).

| filename             | voicing | duration |
| -------------------- | ------- | -------- |
| NJ8_sC_0_File017.wav | 0       | 72       |
| NJ1_zV_8_File036.wav | 8       | 127      |
| ...                  | ...     | ...      |

#### 📁 data/participants/p01.csv

This spreadsheet contains participant 1's answers in the real experiment.
| filename             | answer_batch |
| -------------------- | ------------ |
| NJ8_sC_0_File017.wav | s            |
| ...                  | ...          |


### ⚙️ Config & Technical Stuff

The model takes two numbers (e.g. voicing and duration) as predictors for a binary classification (e.g. batched /s/ /ʃ/ vs. batched /z/ /ʒ/)

```python
# data structure config
PREDICTOR1 = 'voicing'                    # first predictor column name
PREDICTOR2 = 'duration'                   # second predictor column name
TARGET = 'answer'                         # target column name
FILENAME_COL = 'filename'                 # filename column name
LABEL_MAPPING = {'s': 0, 'z': 1}          # binary output label mapping
DATA_PATH = 'data/data.csv'               # sound info data file path
PARTICIPANT_CSV_DIR = 'data/participants' # participant CSV directory

# model parameters
INIT_RANDOM_SAMPLES = 10                  # initial random samples to collect
MIN_ITERATIONS = 20                       # minimum number of iterations
MODEL_CERTAINTY_CUTOFF = 0.95             # stopping certainty threshold
PARTICIPANT_TO_MODEL = 'p01'              # participant ID to simulate
```

### 🔄 Customization Tips
* See how `MODEL_CERTAINTY_CUTOFF` affects the number of samples collected and prediction quality
* Human participant: see how `CLEANSER_FREQUENCY` affects fatigue (by preventing long stretches of ambiguous stimuli)
* Virtual agent: simulate other participants by changing `PARTICIPANT_TO_MODEL`


TODO: Bořil citation
