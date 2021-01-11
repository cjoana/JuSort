from collections import OrderedDict, defaultdict


# file information
datasetdir = "/home/..."
spikedir = "{}/PREPROCESSED/tmp/new".format(datasetdir)
taskdir = "{}/RAWDATA".format(datasetdir)
savedir = "."

params = OrderedDict()
# data properties
params["SamplingRate"] = 20000.0
params["NumChannels"] = 24
# unit selection criteria
params["MinNumSpikes"] = 500  # minimum number of spikes to be selected
params["MinNumTrials"] = 30  # minimun number of trials to be selected
# periodization parameters
params["SpikeBinSize"] = params["MinNumSpikes"]  # bin width in number of spikes
params["SpikeBinStep"] = params["SpikeBinSize"] / 10  # bin step in number of spikes
params["GapNRefs"] = 1  # number of references for the gap statistics computation
params["FDR-q"] = 0.05  # q-value for the false discovary rate control
# segmentation parameters
params["TrialBinSize"] = params["MinNumTrials"] / 2  # bin width in number of trials
params["TrialBinStep"] = 1  # bin step in number of trials
# quality check parameters
params["RPVThreshold"] = 0.001  # ISI refractory period violation threshold in sec

# odML parameters
odml_units = defaultdict(lambda: None)
odml_units["SamplingRate"] = "Hz"
odml_units["Start"] = "s"
odml_units["End"] = "s"
odml_units["RPVThreshold"] = "s"
odml_units["NoiseLevel"] = "uV"
odml_dtypes = defaultdict(lambda: None)
odml_dtypes["Channel"] = "int"
odml_dtypes["NumSpikes"] = "int"
odml_dtypes["NumTrials"] = "int"
odml_dtypes["Start"] = "float"
odml_dtypes["End"] = "float"
odml_dtypes["MeanUnimodality"] = "float"
odml_dtypes["MeanSurprise"] = "float"
odml_dtypes["RPV"] = "float"
odml_dtypes["SNR"] = "float"

odml_author = ""
odml_version = 0.1

# session information
datasets = [
    # format: [sbj, sess, rec, blk, site]
    ["..."],
]

