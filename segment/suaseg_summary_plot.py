import numpy as np
import scipy
import sklearn.cluster
import scipy.spatial.distance
import scipy.stats as spstats
import matplotlib.pyplot as plt
from matplotlib import gridspec

from odml.tools.xmlparser import XMLWriter, XMLReader

from suaseg import load_class_header, find_filenames, load_task, identify_trial_time_ranges


if __name__ == "__main__":
    # file information
    odmldir = "."
    savedir = "."

    # plot parameters
    bin_size = 50.0
    bin_step = 5.0

    savefig = True
    # savefig = False

    from suaseg_conf import *

    for sbj, sess, rec, blk, site in datasets:
        dataset_name = "{}_rec{}_blk{}_{}_h".format(sess, rec, blk, site)
        odml_name = "{}_rec{}_blk{}_{}_h_SUA".format(sess, rec, blk, site)
        filename_odml = "{}/{}.odml".format(odmldir, odml_name)
        metadata = XMLReader().fromFile(filename_odml)
        if metadata["Dataset"]["SpikeData"].properties["NumUnits"].value.data == 0:
            print "\n{sbj}:{odml_name} contains no units.\n".format(**locals())
            continue

        print "\n{sbj}:{odml_name} ".format(**locals())

        num_ch = metadata["Dataset"]["SpikeData"].properties["NumChannels"].value.data
        rpv_threshold = metadata["Dataset"]["SpikeData"].properties["RPVThreshold"].value.data * 1000
        noise_level = metadata["Dataset"]["SpikeData"].properties["NoiseLevel"].value.data

        # load spike data and extract necessary information
        filename_class = metadata["Dataset"]["SpikeData"].properties["File"].value.data
        print filename_class
        print "\tLoading spike data..."
        spike_data = np.genfromtxt(filename_class, skip_header=2, dtype=None, names=True)
        print "\t...done.\n"
        spike_times = spike_data['event_time']
        spike_covs = np.array([spike_data["ch{}".format(i_ch)] for i_ch in range(params["NumChannels"])])
        spike_types = spike_data['type']

        # load task event data and extract necessary information
        filename_task = find_filenames(taskdir, sbj, sess, rec, 'task')[0]
        print "\tLoading task event data..."
        task_events, task_params = load_task(filename_task, blk)
        print "\t...done.\n"
        # time stamps in the task data file are relative to the beginning of the recording. Here the onset time of the
        # block is subtracted from the time stamps so that they are relative to the beginning of the block.
        task_events["evtime"] -= task_events["evtime"][0]
        block_dur = task_events["evtime"][-1] / params["SamplingRate"]
        trial_time_ranges = identify_trial_time_ranges(task_events, task_params, params["SamplingRate"])

        unitIDs = [int(x.data) for x in metadata["Dataset"]["SpikeData"].properties["UnitIDs"].values]
        for unitID in unitIDs:
            mask_unit = (spike_types == unitID)
            unit_label = "Unit{}".format(unitID)
            sect_unit = metadata["Dataset"]["SpikeData"][unit_label]
            unit_ch = sect_unit.properties["Channel"].value.data
            num_spike = sect_unit.properties["NumSpikes"].value.data
            num_trial = sect_unit.properties["NumTrials"].value.data
            num_period = sect_unit.properties["NumPeriods"].value.data

            # compute time resolved firing rate and mean covariances
            spike_times_unit = spike_times[mask_unit]
            spike_covs_unit = spike_covs[:, mask_unit]
            bin_times = np.arange(spike_times_unit[0] + bin_size / 2, spike_times_unit[-1], bin_step)
            firing_rates = np.empty(bin_times.size)
            mean_covs = np.zeros((num_ch, bin_times.size))
            for i, t_ini in enumerate(bin_times):
                idxs_spikes_in_bin = (t_ini-bin_size/2 <= spike_times_unit) & (spike_times_unit < t_ini+bin_size/2)
                firing_rates[i] = (idxs_spikes_in_bin).sum() / bin_size
                mean_covs[:, i] = spike_covs_unit[:, idxs_spikes_in_bin].mean(1)

            # compute trial-wise firing rate
            num_trial_all = len(trial_time_ranges)
            trial_firing_rates = np.empty(num_trial_all)
            for i_trial, (t_ini, t_fin) in enumerate(trial_time_ranges):
                mask_trial = (t_ini <= spike_times_unit) & (spike_times_unit < t_fin)
                trial_firing_rates[i_trial] = mask_trial.sum() / (t_fin-t_ini)

            # organize axes
            fig = plt.figure(figsize=(10, 8))
            fig.subplots_adjust(left=0.08, right=0.96)
            title = "{}:{}".format(sbj, dataset_name)
            title += "\nUnit {} (Ch {}, {} spikes, {} trials)".format(unitID, unit_ch, num_spike, num_trial)
            fig.suptitle(title)
            gs = gridspec.GridSpec(5, 2, width_ratios=[4, 1])
            ax_meancovs = fig.add_subplot(gs[0])
            ax_meancovs.set_xlabel("Time (s)")
            ax_meancovs.set_ylabel("Channel")
            ax_covs = fig.add_subplot(gs[2])
            ax_covs.set_xlabel("Time (s)")
            ax_covs.set_ylabel("Spike size (cov)")
            ax_covs_hist = fig.add_subplot(gs[3])
            ax_covs_hist.set_xlabel("Count")
            ax_covs_hist.set_ylabel("Spike size (cov)")
            ax_periods = fig.add_subplot(gs[4])
            ax_periods.set_xlabel("Time (s)")
            ax_periods.set_ylabel("Firing rate (1/s)")
            ax_periods2 = ax_periods.twinx()
            ax_periods2.set_xlabel("Time (s)")
            ax_periods2.set_ylabel("Unimodality")
            ax_segs = fig.add_subplot(gs[6])
            ax_segs.set_xlabel("Time (s)")
            ax_segs.set_ylabel("Firing rate (1/s)")
            ax_segs2 = ax_segs.twinx()
            ax_segs2.set_xlabel("Time (s)")
            ax_segs2.set_ylabel("RPV")
            ax_isis = fig.add_subplot(gs[8])
            ax_isis.set_xlabel("Time (s)")
            ax_isis.set_ylabel("ISI (ms)")
            ax_isis_hist = fig.add_subplot(gs[9])
            ax_isis_hist.set_xlabel("Count")
            ax_isis_hist.set_ylabel("ISI (log10(ms))")

            # make plots
            X, Y = np.meshgrid(
                bin_times,
                np.linspace(-0.5, num_ch-0.5, num_ch+1)
            )
            vmax = np.abs(mean_covs).max()
            ax_meancovs.pcolormesh(X, Y, mean_covs, vmax=vmax, vmin=-vmax, cmap="bwr")
            ax_meancovs.set_xlim(0, block_dur)
            ax_meancovs.set_ylim(23.5, -0.5)
            ax_meancovs.grid(color="gray")

            ax_covs.plot(spike_times_unit, spike_covs_unit[unit_ch], "k,")
            for t_ini, t_fin in trial_time_ranges:
                ax_covs.axvspan(t_ini, t_fin, color='gray', alpha=0.2, linewidth=0)
            ax_covs.set_xlim(0, block_dur)
            ax_covs.set_ylim(0, 200)
            ax_covs.grid(color="gray")
            ax_covs_hist.hist(spike_covs[unit_ch, mask_unit], bins=200, range=[0, 200], orientation="horizontal", linewidth=0, color="black")
            ax_covs_hist.grid(color="gray")
            ax_covs.axhline(y=noise_level*4.5, color="red", linestyle="--")

            ax_periods.plot(bin_times, firing_rates, 'k-')
            for t_ini, t_fin in trial_time_ranges:
                ax_periods.axvspan(t_ini, t_fin, color='gray', alpha=0.2, linewidth=0)
            ax_periods.set_xlim(0, block_dur)
            ax_periods.set_ylim(ymin=0)
            ax_periods.grid(color="gray")
            ax_periods2.axhline(y=1, color="green", linestyle="--")

            # ax_segs.plot(bin_times, firing_rates, 'k-')
            ax_segs.plot(trial_time_ranges.mean(1), trial_firing_rates, 'k+-')
            for t_ini, t_fin in trial_time_ranges:
                ax_segs.axvspan(t_ini, t_fin, color='gray', alpha=0.2, linewidth=0)
            ax_segs.set_xlim(0, block_dur)
            ax_segs.set_ylim(ymin=0)
            ax_segs.grid(color="gray")
            ax_segs2.axhline(y=0.01, color="blue", linestyle="--")

            for t_ini, t_fin in trial_time_ranges:
                ax_isis.axvspan(t_ini, t_fin, color='gray', alpha=0.2, linewidth=0)
            isis = np.diff(spike_times_unit) * 1000
            ax_isis.plot(spike_times_unit[:-1], isis, "k,")
            # ax_isis.axhline(rpv_threshold, color="red", linestyle='--')
            ax_isis.axhspan(0.1, rpv_threshold, color="blue", alpha=0.1)
            ax_isis.set_yscale('log')
            ax_isis.set_xlim(0, block_dur)
            ax_isis.set_ylim(0.1, 1000)
            ax_isis.grid(color="gray")
            ax_isis_hist.hist(np.log10(isis), bins=40, range=[np.log10(0.1), np.log10(1000)], orientation="horizontal", linewidth=0, color="black")
            ax_isis_hist.axhspan(np.log10(0.1), np.log10(rpv_threshold), color="blue", alpha=0.1)
            ax_isis_hist.set_ylim(np.log10(0.1), np.log10(1000))
            ax_isis_hist.grid(color="gray")

            periodIDs = [] if num_period ==0 else [int(x.data) for x in sect_unit.properties["PeriodIDs"].values]
            for periodID in periodIDs:
                period_label = "Period{}".format(periodID)
                sect_period = metadata["Dataset"]["SpikeData"][unit_label][period_label]
                t_ini = sect_period.properties["Start"].value.data
                t_fin = sect_period.properties["End"].value.data
                unimodality = sect_period.properties["MeanUnimodality"].value.data
                num_seg = sect_period.properties["NumSegments"].value.data
                mask_period = mask_unit & (t_ini <= spike_times) & (spike_times <= t_fin)

                ax_periods2.plot([t_ini, t_fin], [unimodality, unimodality], 'g-')
                plotcolor = 'red' if periodID >= 0 else 'blue'
                ax_periods2.axvline(t_ini, color=plotcolor, alpha=0.5)
                ax_periods2.axvline(t_fin, color=plotcolor, alpha=0.5)
                ax_periods2.axvspan(t_ini, t_fin, color=plotcolor, linewidth=2, alpha=0.1)
                ax_periods2.set_xlim(0, block_dur)
                ax_periods2.set_ylim(0, 2)

                segIDs = [] if num_seg == 0 else [int(x.data) for x in sect_period.properties["SegmentIDs"].values]
                for segID in segIDs:
                    seg_label = "Segment{}".format(segID)
                    sect_seg = metadata["Dataset"]["SpikeData"][unit_label][period_label][seg_label]
                    t_ini = sect_seg.properties["Start"].value.data
                    t_fin = sect_seg.properties["End"].value.data
                    rpv = sect_seg.properties["RPV"].value.data
                    mask_seg = mask_period & (t_ini <= spike_times) & (spike_times <= t_fin)

                    ax_segs2.plot([t_ini, t_fin], [rpv, rpv], 'b-')
                    plotcolor = 'red' if segID >= 0 else 'blue'
                    ax_segs2.axvline(t_ini, color=plotcolor, alpha=0.5)
                    ax_segs2.axvline(t_fin, color=plotcolor, alpha=0.5)
                    ax_segs2.axvspan(t_ini, t_fin, color=plotcolor, linewidth=2, alpha=0.1)
                    ax_segs2.set_xlim(0, block_dur)
                    ax_segs2.set_ylim(0, 0.1)

            if savefig:
                filename_fig = "{}/{}_unit{}.png".format(savedir, odml_name, unitID)
                plt.savefig(filename_fig)
                print "\tFigure saved as {}\n".format(filename_fig)
                plt.close("all")
            else:
                plt.show()
