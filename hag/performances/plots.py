import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette

# ---------------------------------------------------------------------------
# Style of the figures: name, color and order of each method, name of each dataset
# ---------------------------------------------------------------------------
_blues = color_palette("Blues", 5)
_oranges = color_palette("Oranges", 2)
_greens = color_palette("Greens", 2)
_reds = color_palette("Reds", 2)
_greys = color_palette("Greys", 3)
_purples = color_palette("Purples", 3)

# function name (studies, results) -> label in the figures
function_mapping = {
    'random_ee':    'E-ESN',
    'random_ei':    'ESN',
    'ip_correct':   'IP',
    'anti-oja':     'Anti-Oja',
    'ip-anti-oja':  'IP +\nAnti-Oja',
    'mean_hag':     'mean HAG',
    'var_hag':      'variance HAG',
    'lstm_last':    'LSTM',
    'gru':          'GRU',
    'rnn':          'RNN',
    'rnn-mean_hag': 'RNN-HAG',
    'hsp':          'HSP',
    'short-hag':    'short HAG',
    'diag_ee':      'diag EE',
    'diag_ei':      'diag EI',
}

# label -> color
function_colors = {
    'E-ESN':          _greens[0],
    'ESN':            _greens[1],
    'IP':             _blues[0],
    'Anti-Oja':       _blues[1],
    'IP +\nAnti-Oja': _blues[3],
    'mean HAG':       _oranges[0],
    'variance HAG':   _oranges[1],
    'LSTM':           _greys[0],
    'RNN':            _greys[1],
    'GRU':            _greys[2],
    'RNN-HAG':        _greys[2],
    'diag EE':        _reds[0],
    'diag EI':        _reds[1],
    'HSP':            _purples[0],
    'short HAG':      _purples[2],
}

# labels shown in the figures, in this order (add e.g. 'HSP' or 'RNN' to show them)
functions_order = [
    'E-ESN',
    'ESN',
    'IP',
    'Anti-Oja',
    'IP +\nAnti-Oja',
    'mean HAG',
    'variance HAG',
    'LSTM',
    'GRU',
]

# dataset name -> label in the figures
dataset_label_map = {
    "JapaneseVowels": "Japanese Vowels",
    "CatsDogs": "Cats vs Dogs",
    "FSDD": "FSDD",
    "SpokenArabicDigits": "Spoken Arabic Digits",
    "SPEECHCOMMANDS": "Speech Commands",
    "MackeyGlass": "Mackey-Glass",
    "Lorenz": "Lorenz",
    "Sunspot_daily": "Sunspot Daily",
}


def plot_prediction_vs_actual(y_pred, y_test, start=0, end=500):
    sample = slice(start, end)
    x_corrd = np.arange(start, end)
    fig = plt.figure(figsize=(15, 7))
    plt.subplot(211)
    plt.plot(x_corrd, y_pred[sample], lw=3, label="ESN prediction")
    plt.plot(x_corrd, y_test[sample], linestyle="--", lw=2, label="True value")
    plt.plot(x_corrd, np.abs(y_test[sample] - y_pred[sample]), label="Absolute deviation")

    #plot legend outside the graph
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
