# .ipynb version is available.
The raw data can be found at https://www.dropbox.com/sh/iwdmcfpooxjfy6x/AACc6IcICIc7RbnRjTkid6bTa?dl=0
Reminder: these codes need to be packed up for testing. Upon training, you must generate graph datasets from raw datasets. Then, using both model and training files. 

## Rolling feature selection

The `utils/feature_selection.py` module provides a helper `walk_forward_selector`
function that re-evaluates important features inside a time-series split.  By
default it runs the Boruta-SHAP algorithm on each training window, but HSIC-Lasso,
elastic-net with exponential forgetting or simple online selectors can also be
used.  Only the training slice is seen by the selector so validation leakage is
avoided.
