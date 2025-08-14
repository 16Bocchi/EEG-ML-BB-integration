import matplotlib.pyplot as plt

from classes import xdf_loader as xdfl


def main():
    bb_eeg_data = xdfl.EEGXDFLoader("data/P001_S001_BB.xdf")
    bb_eeg_data.load()

    bb_eeg_data.list_streams()

    bb_eeg_data.select_stream(eeg_name="LiveAmpSN-055606-0358", marker_name="CCPT-V")

    # bb_eeg_data.summary()
    # bb_eeg_data.plot_raw()
    # bb_eeg_data.plot_psd()

    bb_eeg_data.preprocess()
    bb_eeg_data.extract_events()

    bb_eeg_data.summary()
    # bb_eeg_data.plot_raw()
    # bb_eeg_data.plot_psd()
    bb_epoch_data = bb_eeg_data.get_epochs()
    bb_epoch_data.compute_psd(fmax=50, fmin=1).plot()
    eeg_data = xdfl.EEGXDFLoader("data/P001_S001.xdf")
    eeg_data.load()

    eeg_data.list_streams()

    eeg_data.select_stream(eeg_name="LiveAmpSN-055606-0358", marker_name="CCPT-V")

    # eeg_data.summary()
    # eeg_data.plot_raw()
    # eeg_data.plot_psd()

    eeg_data.preprocess()
    eeg_data.extract_events()

    eeg_data.summary()
    # eeg_data.plot_raw()
    # eeg_data.plot_psd()
    epoch_data = eeg_data.get_epochs()
    epoch_data.compute_psd(fmax=50, fmin=1).plot()
    plt.show()


if __name__ == "__main__":
    main()
