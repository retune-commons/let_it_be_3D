import imageio as iio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from abc import abstractmethod, ABC
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path, PosixPath, WindowsPath


class FramedropFinder:
    """
    Input: Filepath
    Output: Dataframe


    """

    @abstractmethod
    def find_correct_phases(self):
        """
        Determine which phases are correct and which phases are not correct
        """
        pass

    def __init__(self, filepath: Path, LED_coords, framerate):
        self.filepath = filepath
        self.LED_coords = LED_coords
        self.framerate = framerate

        # loads LED timeseries as pd.df
        self.df_LED = self._load_data()

        # calculates timepoints based on frame rate and frame number
        self._compute_time()

    def _load_data(self):
        if self.filepath.endswith(".mp4"):
            return pd.DataFrame(
                {
                    "LED_lightintensity": self._extract_LED_timeseries_from_video(
                        self.filepath
                    )
                }
            )
        elif self.filepath.endswith(".csv"):
            return pd.read_csv(self.filepath, index_col=0)

    def _compute_time(self):
        # Creating column 'time' in 'processed_DataFrame', setting base value to np.NaN
        self.df_LED["time"] = np.NaN
        # Calculating time and adding it as a column to the df
        self.df_LED["time"] = (
            self.df_LED.iloc[:, 0].index / self.framerate + 1 / self.framerate
        )

    def _extract_LED_timeseries_from_video(self) -> np.array:
        led_timeseries = []
        for image in iio.imiter(self.filepath):
            led_timeseries.append(image[self.LED_coords].mean())
        return led_timeseries

    def plot_timeseries(self, beginning_of_slice: int, end_of_slice: int) -> None:
        plt.plot(self.df_LED["LED_lightintensity"][beginning_of_slice:end_of_slice])

    def check_LED_on(self, LED_threshold: int) -> None:
        self.df_LED["LED_on"] = False
        self.df_LED.loc[
            self.df_LED["LED_lightintensity"] > LED_threshold, ["LED_on"]
        ] = True

    def check_LED_switching(self):
        """
        Creates two new columns in the df:
            ['switch'] for when the LED switches in any direction
            ['switch_on'] for when the LED switches from off to on
            ['switch_off'] for when the LED switches from on to off
        """
        self.df_LED.loc[
            (self.df_LED["LED_on"] != self.df_LED["LED_on"].shift(1)), ["switch"]
        ] = True

    def compute_period_length(self):
        times_where_switch_true = self.df_LED.index[
            np.where(self.df_LED["switch"] == True)
        ]
        indices = times_where_switch_true
        lower_end = (indices + 1)[:-1]
        upper_end = indices[1:]
        interval_ranges = np.column_stack([lower_end, upper_end])

        for first_idx, last_idx in interval_ranges:
            phase_duration = (
                last_idx - first_idx + 1
            )  # +1 to include row in which LED switches

            # -1 to start with row in which the LED switches on or off
            self.df_LED.loc[
                first_idx - 1 : last_idx - 1, "phase_duration"
            ] = phase_duration

    def exclude_framedrop_phases(self):
        self.df_LED["exclude_frame"] = True
        self.df_LED.loc[self.df_LED["correct_phase"] == True, "exclude"] = False

    def count_amount_of_phase_duration(self):
        set_of_phase_durations = set(
            [x for x in self.df_LED["phase_duration"] if np.isnan(x) == False]
        )
        for elem in set_of_phase_durations:
            self.df_LED.loc[
                self.df_LED["phase_duration"] == elem, "occurance_of_phase_duration"
            ] = len(self.df_LED.loc[self.df_LED["phase_duration"] == elem])

    def set_occurance_threshold(self, occurance_threshold):
        self.occurance_threshold = occurance_threshold


class PhaseOccuranceExcluder(FramedropFinder):
    def _count_amount_of_phase_duration(self):
        set_of_phase_durations = set(
            [x for x in self.df_LED["phase_duration"] if np.isnan(x) == False]
        )
        for elem in set_of_phase_durations:
            self.df_LED.loc[
                self.df_LED["phase_duration"] == elem, "occurance_of_phase_duration"
            ] = len(self.df_LED.loc[self.df_LED["phase_duration"] == elem])

    def set_occurance_threshold(self, occurance_threshold):
        self.occurance_threshold = occurance_threshold

    def find_correct_phases(self):
        self._count_amount_of_phase_duration()
        self.df_LED["correct_phase"] = False
        if self.occurance_threshold:
            self.df_LED.loc[
                self.df_LED["occurance_of_phase_duration"] > self.occurance_threshold,
                "correct_phase",
            ] = True

            # set the last phase to true -> no full cycle -> no chance to be True otherwise
            self.df_LED.loc[
                self.df_LED["occurance_of_phase_duration"].isna(), "correct_phase"
            ] = True
        else:
            raise KeyError(
                "You have to define the minimum of how often a phase needs to occur for this step using .set_occurance_threshold"
            )


class PatternExcluder(FramedropFinder):
    """
    Checks code for seasonality (-> patterns),
    slices the video into the motifs
    and then looks at the pattern length to determine correct phases
    input: amount of pattern motif switches
    output: correct LED phases
    """

    def check_motif_switches(self, framerange: int = 2):
        """Checks for switches between motifs, """

        self.df_LED["motif_switch"] = False

        # create new df (off_switches) by slicing df_LED for LED off periods,
        # reindex to then calculate difference between off phases with .shift(1),
        # but keep the index as col in new df to later get the indexes for the df
        off_switches = self.df_LED.loc[
            (self.df_LED["LED_on"] == False) & (self.df_LED["switch"] == True),
            "phase_duration",
        ].reset_index()

        # calculate difference between current and previous off phase duration
        # save that difference as column 'difference_to_previous_off_phase' in off_switches
        off_switches["difference_to_previous_off_phase"] = off_switches[
            "phase_duration"
        ] - off_switches["phase_duration"].shift(1)

        # select for motif switches by checking where the difference between two off phases is bigger than 2 frames
        self.indices_motif_switches = off_switches.loc[
            (
                (off_switches["difference_to_previous_off_phase"] < -framerange)
                | (off_switches["difference_to_previous_off_phase"] > framerange)
            ),
            "index",
        ].values
        return self.indices_motif_switches

    def find_correct_phases(self):

        # create two columns in df_LED: phase_duration_change and framedrop phase
        self.df_LED["phase_duration_change"] = False
        self.df_LED["correct_phase"] = True

        for index in self.indices_motif_switches:

            # if phase change: check how often the phase duration occurs
            if (
                self.df_LED.iloc[
                    index, self.df_LED.columns.get_loc("occurance_of_phase_duration")
                ]
                < self.occurance_threshold
            ):
                # set correct phase as False for index and the whole phase -> exclude
                self.df_LED.iloc[
                    index : int(index + self.df_LED.iloc[index]["phase_duration"]),
                    self.df_LED.columns.get_loc("correct_phase"),
                ] = False
            else:
                self.df_LED.iloc[
                    index, self.df_LED.columns.get_loc("phase_duration_change")
                ] = True

    def _check_no_switch_previous_and_past_rows(self, index):
        if (
            not self.df_LED[index - 2 : index]["phase_duration_change"].any()
            and not self.df_LED[index + 1 : index + 6]["phase_duration_change"].any()
        ):
            return True
        else:
            print("no")
            return False

    def fix_framedrop_phases(self):
        """
        To do:

        Calculate amount of frames missing for whole video -> maybe as input?
        calculate difference between framedrop phase and proper phase -> how many rows need to be added
        define behaviour if more than one framedrop phase
        """
        pass
