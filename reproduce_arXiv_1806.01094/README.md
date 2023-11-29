Code to reproduce the experimental results presented in

Robustifying Independent Component Analysis by Adjusting for Group-Wise Stationary Noise\
N Pfister*, S Weichwald*, P Bühlmann, B Schölkopf\
https://arxiv.org/abs/1806.01094

Refer to the [project website](https://sweichwald.de/coroICA/) for more details and instructions on how to install the provided open-source Python/R/Matlab implementations of the coroICA algorithm.

Niklas Pfister and Sebastian Weichwald are the sole authors of the code files packed within this very .tar.gz file.
Refer to the LICENSE file for the AGPL-3.0 license under which this code archive is licensed.

December 2018



# Preparation

1. make sure to run all commands from within this directory using python3/pip3
2. pip install -r requirements.txt

# Overview - EEG

There are two different datasets, covertattention and BCICompIV2a.

## covertattention

1. download the data .mat files for "14. Covert shifts of attention (005-2015)" from http://bnci-horizon-2020.eu/database/data-sets into ./covertattention
2. Prepare the data using
   `python covertattention_preparedata.py`
3. Run the ICAs\
   `python covertattention_runICAs.py k`\
   for k=0..253.
   (These commands compute and save the respective unmixing matrices and the computations are costly.)
4. Score the ICAs\
   `python covertattention_scorer.py k`\
   for k=0..253, followed by
   `python covertattention_scorer.py collect`
  5. Generate stability plots\
   `python covertattention_plotter.py`
  6. Classification experiment\
  `python covertattention_classification.py`\
    followed by
  `python covertattention_classification_plot.py`
7. The topographic maps can be plotted using\
   `python covertattention_topo.py`

## BCICompIV2a data

1. download the data .mat files for "1. Four class motor imagery (001-2014)" from http://bnci-horizon-2020.eu/database/data-sets into ./BCICompIV2a
2. Prepare the data using\
   `python BCICompIV2a_preparedata.py`
3. Run the ICAs\
   `python BCICompIV2a_runICAs.py k`\
   for k=0..509
   (These commands compute and save the respective unmixing matrices and the computations are costly.)
 4. Score the ICAs\
   `python BCICompIV2a_scorer.py k`\
   for k=0..253, followed by
   `python BCICompIV2a_scorer.py collect`
 5. Generate stability plots\
  `python BCICompIV2a_plotter.py`
 6. Classification experiment\
  `python BCICompIV2a_classification.py`\
  followed by
  `python BCICompIV2a_classification_plot.py`

# Overview - simulations

There are eight simulation experiments
* experiment1 -- signal-strength simulation (block-var signal)
* experiment2 -- zero-confounding simulation (block-var signal)
* experiment3 -- GARCH model (var signal, ar noise)
* experiment4 -- GARCH model (TD signal, ar noise)
* experiment5 -- GARCH model (TD & var signal, ar noise)
* experiment6 -- GARCH model (var signal, iid noise)
* experiment7 -- GARCH model (TD signal, iid noise)
* experiment8 -- GARCH model (TD & var signal, iid noise)

1. To reproduce the results first run the experiments\
   `python simulations_run.py experiment1`\
   `python simulations_run.py experiment2`\
   `python simulations_run.py experiment3`\ 
   `python simulations_run.py experiment4`\
   `python simulations_run.py experiment5`\
   `python simulations_run.py experiment6`\
   `python simulations_run.py experiment7`\
   `python simulations_run.py experiment8`
2. After this has been run the results can be plotted with the following commands\
   `python simulations_plot.py experiment1`\
   `python simulations_plot.py experiment2`\
   `python simulations_plot.py experiment345678`    

# Overview - climate experiment

See the file climate_example.py for detailed instructions on
downloading the data and reconstructing the results.

# Overview - picture example

Run `python picture_example.py`
