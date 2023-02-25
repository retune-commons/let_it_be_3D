from typing import List
from pathlib import Path

import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
import imageio as iio

from utils import check_if_same, convert_to_path


class Framenumberchecker:
    """Checks if framenumbers of all videos are the same
    Input: List of videofilepaths
    """

    def check_framenumber(self, videofilepaths: List) -> bool:
        """checks if all videos in videofilepaths list have the same number of frames"""
        video_framenumbers = []
        for elem in videofilepaths:
            video = Framenumberreader()
            framenumber = video.run(elem)
            video_framenumbers.append(framenumber)
        if check_if_same(video_framenumbers):
            print("All videos have the same amount of frames!")
        else:
            print("Not all videos have the same number of frames!")
            pd.DataFrame(video_framenumbers, index=[videofilepaths])


class Framenumberreader:

    """
    Reads a video, checks & returns the number of frames
    """

    def run(self, videofilepath: Path) -> int:
        return self.read_framenr(self.read_video(videofilepath))

    def read_video(self, videofilepath: Path) -> None:
        return iio.v2.get_reader(videofilepath)

    def read_framenr(self, video) -> int:
        return video.count_frames()
