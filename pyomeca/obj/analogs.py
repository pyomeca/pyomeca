from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

    @staticmethod
    def _parse_c3d(c3d, prefix):
        """
        Implementation on how to read c3d header and parameter for analogs
        Parameters
        ----------
        c3d : ezc3d

        prefix : str, optional
            Participant's prefix

        Returns
        -------
        metadata, channel_names, data
        """
        channel_names = [i.c_str().split(prefix)[-1] for i in
                         c3d.parameters().group('ANALOG').parameter('LABELS').valuesAsString()]
        metadata = {
            'get_num_analogs': c3d.header().nbAnalogs(),
            'get_num_frames': c3d.header().nbAnalogsMeasurement(),
            'get_first_frame': c3d.header().firstFrame() * c3d.header().nbAnalogByFrame(),
            'get_last_frame': c3d.header().lastFrame() * c3d.header().nbAnalogByFrame(),
            'get_rate': c3d.header().frameRate() * c3d.header().nbAnalogByFrame(),
            'get_unit': []
        }
        data = c3d.get_analogs()
        return data, channel_names, metadata

    def rectify(self):
        """
        Rectify a signal (i.e., get absolute values)

        Returns
        -------
        FrameDependentNpArray
        """
        return self.abs()


class MVC:
    """
    Return the Maximal Voluntary Contraction (MVA) array.
    MVA is computed as follow:
            1. read the files
            2. process trial (band-pass, center, rectify, low-pass). You can modify the parameters of these steps in
                self.params
            3. detect onset
            4. remove data that are more than `outlier` standard deviations from the average of the onset
            5. concatenate all trials for a given muscle
            6. get mean of the highest sorted_values activation during `time` seconds

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
    outlier : int
        Multiple of standard deviation from which data is considered outlier
    band_pass_cutoff : list
        Band-pass cut-off frequencies
    low_pass_cutoff : int
        Low-pass cut-off frequencies
    order : int
        Order of the filter
    """

    def __init__(self, directories, channels, plot_trials=False, plot_mva=False, outlier=3,
                 band_pass_cutoff=None, low_pass_cutoff=None, order=4):
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

        self.band_pass_cutoff = band_pass_cutoff if band_pass_cutoff else [10, 425]
        self.low_pass_cutoff = low_pass_cutoff if low_pass_cutoff else 5
        self.order = order
        self.outlier = outlier

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
                    emg = Analogs3d.from_c3d(f'{itrial}', prefix=':', names=iassign_without_nans)
                    if nan_idx:
                        # if there is any empty assignment, fill the dimension with nan
                        emg.get_nan_idx = np.array(nan_idx)
                        for i in nan_idx:
                            emg = np.insert(emg, i, np.nan, axis=1)
                        # check if nan dimension are correctly inserted
                        n = np.isnan(emg).sum(axis=2).ravel()
                        if not np.array_equal(n.argsort()[-len(nan_idx):], nan_idx):
                            raise ValueError('NaN dimensions misplaced')
                        print(f'\t{itrial.stem} (NaNs: {nan_idx})')
                    else:
                        print(f'\t{itrial.stem}')

                    # check if dimensions are ok
                    if not emg.shape[1] == len(iassign):
                        raise ValueError('Wrong dimensions')
                    break
                except IndexError:
                    emg = []

            if np.any(emg):
                trials.append(emg)
            else:
                raise ValueError(f'no assignments were found for the trial {itrial.stem}')

        return trials

    def process_trials(self):
        """Process trials from a list and concatenate them in a single dict"""
        print('Processing trials...')
        concatenated = {imuscle: np.array([]) for imuscle in range(self.trials[0].shape[1])}

        for i, itrial in enumerate(self.trials):
            # emg processing
            itrial = itrial \
                .band_pass(freq=itrial.get_rate, order=self.order,
                           cutoff=self.band_pass_cutoff) \
                .center() \
                .rectify() \
                .low_pass(freq=itrial.get_rate, order=self.order,
                          cutoff=self.low_pass_cutoff)

            for imuscle in range(itrial.shape[1]):
                if self.channels[0][imuscle] == '':
                    concatenated[imuscle] = np.append(concatenated[imuscle], np.nan)
                else:
                    x = itrial[0, imuscle, :]
                    # onset detection
                    idx = x.detect_onset(
                        threshold=np.nanmean(x[..., :int(itrial.get_rate)]),
                        above=int(itrial.get_rate) / 2,
                        below=3,
                        threshold2=np.nanmean(x[..., :int(itrial.get_rate)]) * 2,
                        above2=5
                    )

                    # outliers detection
                    x_without_outliers = x.detect_outliers(onset_idx=idx, threshold=self.outlier)

                    # append the current trial to a dictionary concatenating all trials for each of the muscles
                    concatenated[imuscle] = np.append(concatenated[imuscle], np.ma.compressed(x_without_outliers))

                    if self.plot_trials:
                        plt.plot(x_without_outliers, 'k-')
                        if x_without_outliers.mask.any():
                            plt.plot(np.ma.masked_array(x, ~x_without_outliers.mask), 'r-', label='outlier')
                            plt.legend()
                        plt.title(f'{self.channels[0][imuscle]} | {self.trials_path[i].stem}')
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
        seconds = int(time * self.trials[0].get_rate)
        mva = []
        for imuscle, values in self.concatenated.items():
            if not np.isnan(values).all():
                sorted_values = np.sort(values)
                mu = np.nanmean(sorted_values[-seconds:])
                mva.append(mu)

                if self.plot_mva:
                    plt.plot(sorted_values[-seconds:], 'b-', label='sorted activation')
                    plt.axhline(y=mu, c='k', ls='--', label='MVA')
                    plt.title(f'Last {time} seconds | {self.channels[0][imuscle]}')
                    plt.legend()
                    plt.show()
            else:
                mva.append(None)
        return mva
