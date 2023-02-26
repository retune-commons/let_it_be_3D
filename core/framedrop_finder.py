import imageio as iio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path, PosixPath, WindowsPath


class FramedropFinder:
    """
    Input: Filepath
    Output: Dataframe


    """

    def __init__(self, filepath: Path, LED_coords: List[int:int]):
        self.filepath = filepath
        self.LED_coords = LED_coords

        # loads LED timeseries as pd.df
        self.df_LED = self._load_data()

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

    def _extract_LED_timeseries_from_video(self) -> np.array():
        led_timeseries = []
        for image in iio.imiter(self.filepath):
            led_timeseries.append(image[self.LED_coords].mean())
        return led_timeseries

    def plot_timeseries(
        self, df: pd.DataFrame, beginning_of_slice: int, end_of_slice: int
    ) -> None:
        plt.plot(df["LED_lightintensity"][beginning_of_slice:end_of_slice])

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

    # to do: check if period is more than 1 frame different compared to

    def _calculate_correct_phase_duration(self):
        phase_duration_LED_on = np.median(
            self.df_LED.loc[self.df_LED["LED_on"] == True, "phase_duration"]
        )
        self.correct_phase_duration_LED_on = [
            phase_duration_LED_on - 1,
            phase_duration_LED_on,
            phase_duration_LED_on + 1,
        ]
        phase_duration_LED_off = np.median(
            self.df_LED.loc[self.df_LED["LED_on"] == False, "phase_duration"]
        )
        self.correct_phase_duration_LED_off = [
            phase_duration_LED_off - 1,
            phase_duration_LED_off,
            phase_duration_LED_off + 1,
        ]

    def check_correct_phase_duration(self):
        # calculates correct phase duration
        self._calculate_correct_phase_duration()

        self.df_LED["correct_phase_duration"] = False

        # set the last phase duration to true -> no full cycle -> no chance to be True otherwise
        self.df_LED.loc[
            self.df_LED["phase_duration"].isna(), "correct_phase_duration"
        ] = True

        # set all values where the phase duration is correct to True
        notna_mask = self.df_LED["phase_duration"].notna()
        correct_on_mask = (
            notna_mask
            & (self.df_LED["phase_duration"] > (self.correct_phase_duration_LED_on - 2))
            & (self.df_LED["phase_duration"] < (self.correct_phase_duration_LED_on + 2))
            & (self.df_LED["LED_on"] == True)
        )
        correct_off_mask = (
            notna_mask
            & (
                self.df_LED["phase_duration"]
                > (self.correct_phase_duration_LED_off - 2)
            )
            & (
                self.df_LED["phase_duration"]
                < (self.correct_phase_duration_LED_off + 2)
            )
            & (self.df_LED["LED_on"] == False)
        )
        self.df_LED.loc[
            correct_on_mask | correct_off_mask, "correct_phase_duration"
        ] = True

        # set the last phase duration to true -> no full cycle -> no chance to be True otherwise
        self.df_LED.loc[
            self.df_LED["phase_duration"].isna(), "correct_phase_duration"
        ] = True

    def exclude_incorrect_phase_duration_values(self):
        self.df_LED["exclude_frame"] = True
        self.df_LED.loc[
            self.df_LED["correct_phase_duration"] == True, "exclude"
        ] = False
