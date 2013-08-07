Prototyping calorimeter clustering algorithms for E989, the Muon g-2 experiment
-------------------------------------------------------------------------------

This code has a few pieces.

## Signal generation

Generate simulated calorimeter signals. Save both the generated waveforms
and the truth information.

The main interface to this is through `gm2_clustering/wf_generator/save_waveform.py`. This
will generate waveforms and save them to a .npz file. Run it without any arguments to see
the command line help.

## Testing

Run an algorithm on some generated signals and test its performance. This code is located in
`gm2_clustering/algo_tests`, and is intented to be called by a clustering script which tests
itself.

## Clustering

The actual algorithms that do the clustering. Code is in `algos`. To write a new algorithm,
define a function that takes in a waveform and outputs a list of length 3 tuples which
give the algorithm's guess for x0, y0, and t0 for each electron in the waveform. There should
be one tuple per electron if the algorithm thinks there are multiple electrons.

Code to test algorithm performance is in `gms_clustering/algo_tests`, but should be called when
the algorithm script is run. See `algos/e821.py` for an example.