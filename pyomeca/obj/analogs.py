from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyomeca import signal as pyosignal
from pyomeca.obj.frame_dependent import FrameDependentNpArray


class Analogs3d(FrameDependentNpArray):
    def __new__(cls, data=np.ndarray((1, 0, 0)), names=list(), *args, **kwargs):
        """
        Parameters
        ----------
        data : np.ndarray
            1xNxF matrix of analogs data
        names : list of string
            name of the analogs that correspond to second dimension of the matrix
        """
        if data.ndim == 2:
            data = Analogs3d.from_2d(data)

        if data.ndim == 3:
            s = data.shape
            if s[0] != 1:
                raise IndexError('Analogs3d must have a length of 1 on the first dimension')
            analog = data
        else:
            raise TypeError('Data must be 2d or 3d matrix')

        return super(Analogs3d, cls).__new__(cls, array=analog, *args, **kwargs)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        # Allow slicing
        if obj is None or not isinstance(obj, Analogs3d):
            return

    # --- Get metadata methods

    def get_num_analogs(self):
        """
        Returns
        -------
        The number of analogs
        """
        s = self.shape
        return s[1]

    def get_2d_labels(self):
        """
        Takes a Analogs style labels and returns 2d style labels
        Returns
        -------
        2d style labels
        """
        return self.get_labels

    # --- Fileio methods (from_*)

    @staticmethod
    def from_2d(m):
        """
        Takes a tabular matrix and returns a Vectors3d
        Parameters
        ----------
        m : np.array
            A CSV tabular matrix (Fx3*N)
        Returns
        -------
        Vectors3d of data set
        """
        s = m.shape
        return Analogs3d(np.reshape(m.T, (1, s[1], s[0]), 'F'))

    # --- Fileio methods (to_*)

    def to_2d(self):
        """
        Takes a Analogs3d style matrix and returns a tabular matrix
        Returns
        -------
        Tabular matrix
        """
        return np.squeeze(self.T, axis=2)

    # --- Signal processing methods

    def moving_rms(self, window_size, method='filtfilt'):
        return Analogs3d(pyosignal.moving_rms(self, window_size, method))

    def moving_average(self, window_size, method='filtfilt'):
        return Analogs3d(pyosignal.moving_average(self, window_size, method))

    def moving_median(self, window_size):
        return Analogs3d(pyosignal.moving_median(self, window_size))

    def low_pass(self, freq, order, cutoff):
        return Analogs3d(pyosignal.low_pass(self, freq, order, cutoff))

    def band_pass(self, freq, order, cutoff):
        return Analogs3d(pyosignal.band_pass(self, freq, order, cutoff))

    def band_stop(self, freq, order, cutoff):
        return Analogs3d(pyosignal.band_stop(self, freq, order, cutoff))

    def high_pass(self, freq, order, cutoff):
        return Analogs3d(pyosignal.high_pass(self, freq, order, cutoff))

    def time_normalization(self, time_vector=np.linspace(0, 100, 101), axis=-1):
        return Analogs3d(pyosignal.time_normalization(self, time_vector, axis=axis))

    def fill_values(self, axis=-1):
        return Analogs3d(pyosignal.fill_values(self, axis))


class MVC:

    def __init__(self, directories, channels, plot_trials=False, plot_mva=False):
        """
        Return the Maximal Voluntary Contraction (MVA) array.
        MVA is computed as follow:
            for trial:
                1. read the file
                2. process trial (band-pass, center, rectify, low-pass). You can modify the parameters of these steps in
                    self.params
                3. detect onset
                4. remove data that are more than three standard deviations from the average of the onset
                2. get mean of the highest sorted_values activation during `time` seconds

        Parameters
        ----------
        directories : list
            List of directories containing the trials to be processed
        channels : list
            List (or list of lists) of string associated with each channel
        plot_trials : bool
            If the plot of each trial must be displayed
        plot_mva : bool
            If the plot of each mva must be displayed
        """
        self.trials_path = []
        for idir in directories:
            idir = Path(idir)
            if not idir.is_dir():
                raise ValueError(f'{str(idir)} does not exist.')
            for ifile in idir.glob('*.c3d'):
                self.trials_path.append(ifile)
        # make a nested list if not already nested
        if not any(isinstance(i, list) for i in channels):
            channels = [channels]
        self.channels = channels
        self.plot_trials = plot_trials
        self.plot_mva = plot_mva

        self.params = {
            'band_pass': {'order': 4, 'cutoff': [10, 425]},
            'low_pass': {'order': 4, 'cutoff': 5},
            'outlier': 3
        }

        self.trials = self.read_files()
        self.concatenated = self.process_trials()

    def read_files(self):
        """Read c3d files and append them to a list"""
        trials = []
        for itrial in self.trials_path:
            for iassign in self.channels:
                # get index where assignment are empty
                nan_idx = [i for i, v in enumerate(iassign) if not v]
                if nan_idx:
                    iassign_without_nans = [i for i in iassign if i]
                else:
                    iassign_without_nans = iassign

                try:
                    emg = Analogs3d.from_c3d(itrial, names=iassign_without_nans, prefix=':')
                    if nan_idx:
                        # if there is any empty assignment, fill the dimension with nan
                        for i in nan_idx:
                            emg = np.insert(emg, i, np.nan, axis=1)
                        # check if nan dimension are correctly inserted
                        n = np.isnan(emg).sum(axis=2).ravel()
                        if not np.array_equal(n.argsort()[-len(nan_idx):], nan_idx):
                            raise ValueError('NaN dimensions misplaced')
                        print(f'trial: {itrial.parts[-1]} (NaNs: {nan_idx})')
                    else:
                        print(f'trial: {itrial.parts[-1]}')

                    # check if dimensions are ok
                    if not emg.shape[1] == len(iassign):
                        raise ValueError('Wrong dimensions')
                except IndexError:
                    emg = []

                if emg is None:
                    raise ValueError(f'no assignments were found for the trial {itrial.parts[-1]}')
                else:
                    trials.append(emg)

        self.rate = emg.get_rate
        return trials

    def process_trials(self):
        """Process trials from a list and concatenate them in a single dict"""
        print('Processing trials...')
        concatenated = {imuscle: np.array([]) for imuscle in range(self.trials[0].shape[1])}

        for i, itrial in enumerate(self.trials):
            # emg processing
            itrial = itrial \
                .band_pass(freq=self.rate, order=self.params['band_pass']['order'],
                           cutoff=self.params['band_pass']['cutoff']) \
                .center() \
                .rectify() \
                .low_pass(freq=self.rate, order=self.params['low_pass']['order'],
                          cutoff=self.params['low_pass']['cutoff'])

            for imuscle in range(itrial.shape[1]):
                if self.channels[0][imuscle] == '':
                    concatenated[imuscle] = np.append(concatenated[imuscle], np.nan)
                else:
                    x = itrial[0, imuscle, :]
                    # onset detection
                    idx = pyosignal.detect_onset(x,
                                                 threshold=np.nanmean(x[..., :int(self.rate)]),
                                                 above=int(self.rate) / 2,
                                                 below=3,
                                                 threshold2=np.nanmean(x[..., :int(self.rate)]) * 2,
                                                 above2=5)
                    # outliers detection
                    x_without_outliers = self.detect_outlier(x, onset_idx=idx, threshold=self.params['outlier'])

                    concatenated[imuscle] = np.append(concatenated[imuscle], np.ma.compressed(x_without_outliers))

                    if self.plot_trials:
                        plt.plot(x_without_outliers, 'k-')
                        if x_without_outliers.mask.any():
                            plt.plot(np.ma.masked_array(x, ~x_without_outliers.mask), 'r-', label='outlier')
                            plt.legend()
                        plt.title(f'{self.channels[0][imuscle]} | {self.trials_path[i].parts[-1]}')
                        plt.show()
        return concatenated

    def get_mva(self, time=2):
        """
        Return the Maximal Voluntary Contraction (MVA) array.
        MVA is computed as follow:
            for each muscle:
                1. sort the vector of all the concatenated trials
                2. get mean of the highest sorted_values activation during `time` seconds

        Parameters
        ----------
        time : int
            Time during which the average is calculated

        Returns
        -------
        numpy.ndarray
        """
        seconds = int(time * self.rate)
        mva = np.full((len(self.channels[0])), np.nan)

        for imuscle, values in self.concatenated.items():
            if not np.isnan(values).all():
                sorted_values = np.sort(values)
                mu = np.nanmean(sorted_values[-seconds:])
                mva[imuscle] = mu

                if self.plot_mva:
                    plt.plot(sorted_values[-seconds:], 'b-', label='sorted activation')
                    plt.axhline(y=mu, c='k', ls='--', label='MVA')
                    plt.title(f'Last {time} seconds | {self.channels[0][imuscle]}')
                    plt.legend()
                    plt.show()
        return mva

    @classmethod
    def detect_outlier(cls, x, onset_idx=None, threshold=3):
        """
        Detects data that is `threshold` times the standard deviation calculated on the `onset_idx`

        Parameters
        ----------
        x : numpy.ndarray
            Vector
        onset_idx : numpy.ndarray
            Array of onset (first column) and offset (second column)
        threshold : int
            Multiple of standard deviation from which data is considered outlier

        Returns
        -------
        numpy masked array
        """
        if np.any(onset_idx):
            mask = np.zeros(x.shape, dtype='bool')
            for (inf, sup) in onset_idx:
                mask[inf:sup] = 1
            sigma = np.nanstd(x[mask])
            mu = np.nanmean(x[mask])
        else:
            sigma = np.nanstd(x)
            mu = np.nanmean(x)
        y = np.ma.masked_where(np.abs(x) > mu + (threshold * sigma), x)
        return y


if __name__ == '__main__':
    DIR = ['/media/romain/E/Projet_MVC/data/C3D_original_files/irsst_hf/ArsT',
           "/media/romain/F/Data/Shoulder/RAW/IRSST_ArsTd/trials"]
    channels = [
        "1-Deltoid Ant",
        "2-Deltoid Mid",
        "3-Deltoid Post",
        "9-Biceps",
        "10-Triceps",
        "6-Trapezius Up",
        "8-Trapezius Low",
        "11-Serratus Ante",
        "",
        "",
        "",
        "5-Pectoralis maj",
        "4-Lassitimus Dor"
    ]
    mvc = MVC(directories=DIR, channels=channels, plot_mva=True)
    mva = mvc.get_mva()
    print('')
