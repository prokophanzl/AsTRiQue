# AsTRiQue
**AS**k **T**he **RI**ght **QUE**stions: An active machine learning framework for perception experiments

## Virtual Agent Showcase
If you'd like to see AsTRiQue in action using a virtual agent, you can run `virtual_agent.ipynb` in a notebook locally or check out the [Google Colab notebook online](https://colab.research.google.com/github/prokophanzl/AsTRiQue/blob/main/colab-notebooks/virtual_agent_showcase.ipynb).

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


### ⚙️ Constants and Configs

```python
# data structure config (don't change this)
PREDICTOR1 = 'voicing'                    # first predictor column name
PREDICTOR2 = 'duration'                   # second predictor column name
TARGET = 'answer'                         # target column name
FILENAME_COL = 'filename'                 # filename column name
LABEL_MAPPING = {'s': 0, 'z': 1}          # binary output label mapping
DATA_PATH = 'data/data.csv'               # sound info data file path
PARTICIPANT_CSV_DIR = 'data/participants' # participant CSV directory

# model parameters (tweak this)
INIT_RANDOM_SAMPLES = 10                  # initial random samples to collect
MIN_ITERATIONS = 20                       # minimum number of iterations
MODEL_CERTAINTY_CUTOFF = 0.95             # stopping certainty threshold
PARTICIPANT_TO_MODEL = 'p01'              # participant ID to simulate
```

### 🔄 Customization Tips
* Simulate other participants by changing `PARTICIPANT_TO_MODEL`
* See how `MODEL_CERTAINTY_CUTOFF` affects the number of samples collected and prediction quality



TODO: Bořil citation
