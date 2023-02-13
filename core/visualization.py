import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Polygon, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
import math

"""
Usage example:
%matplotlib widget
marker_ids_to_connect = test_positions_gt.marker_ids_to_connect_in_3D_plot
"""


class Visualization:
    def __init__(
        self,
        frame_idx: int,
        filepath_to_3D_csv: Path,
        marker_ids_to_connect: List[Tuple[str]],
        paradigm: Optional[str] = None,
        return_frame: bool = False,
        plot_in_cm: bool = False,
    ):
        self._read_csv(filepath_to_3D_csv=filepath_to_3D_csv)
        self.plot_in_cm = plot_in_cm
        self._show_3D_plot(
            frame_idx=frame_idx,
            marker_ids_to_connect=marker_ids_to_connect,
            paradigm=paradigm,
            return_frame=return_frame,
        )

    def _read_csv(self, filepath_to_3D_csv: Path):
        self.df = pd.read_csv(filepath_to_3D_csv)
        self.bodyparts = []
        for key in self.df.keys():
            bodypart = key.split("_")[0]
            if bodypart not in self.bodyparts and bodypart not in set(
                ["M", "center", "fnum"]
            ):
                self.bodyparts.append(bodypart)

    def _show_3D_plot(
        self,
        frame_idx: int,
        marker_ids_to_connect: List[Tuple[str]],
        paradigm: Optional[str] = None,
        return_frame: bool = False,
    ) -> None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        if self.plot_in_cm:
            ax.scatter(
                [-25, -25, 55, 55, -25, -25, 55, 55],
                [-25, 55, 55, -25, -25, 55, 55, -25],
                [-25, -25, -25, -25, 55, 55, 55, 55],
                s=100,
                c="white",
                alpha=0,
            )
            # the line above fixes axes

        for bodypart in self.bodyparts:
            if bodypart not in set(["LED5"]):
                if not math.isnan(self.df.loc[frame_idx, f"{bodypart}_x"]):
                    ax.text(
                        self.df.loc[frame_idx, f"{bodypart}_x"],
                        self.df.loc[frame_idx, f"{bodypart}_y"],
                        self.df.loc[frame_idx, f"{bodypart}_z"],
                        bodypart,
                        size=7,
                    )
                    ax.scatter(
                        self.df.loc[frame_idx, f"{bodypart}_x"],
                        self.df.loc[frame_idx, f"{bodypart}_y"],
                        self.df.loc[frame_idx, f"{bodypart}_z"],
                        s=10,
                        alpha=1,
                    )

        self._connect_all_marker_ids(
            ax=ax, scheme=marker_ids_to_connect, frame_idx=frame_idx
        )
        if self.plot_in_cm:
            self._add_maze_shape(ax=ax, paradigm=paradigm)

        if return_frame:
            # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
            ax.axis("off")
            fig.tight_layout(pad=0)
            ax.margins(0)
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            return frame
        else:
            plt.show()

    def _connect_all_marker_ids(
        self, ax: plt.Figure, scheme: List[Tuple[str]], frame_idx: int
    ) -> List[plt.Figure]:
        # ToDo: correct type hints
        cmap = plt.get_cmap("tab10")
        bp_dict = dict(zip(self.bodyparts, range(len(self.bodyparts))))
        for i, bps in enumerate(scheme):
            self._connect_one_set_of_marker_ids(
                ax=ax, bps=bps, bp_dict=bp_dict, color=cmap(i)[:3], frame_idx=frame_idx
            )

    def _connect_one_set_of_marker_ids(
        self,
        ax: plt.Figure,
        bps: List[str],
        bp_dict: Dict,
        color: np.ndarray,
        frame_idx: int,
    ) -> plt.Figure:
        # ToDo: correct type hints
        x = [f"{bodypart}_x" for bodypart in bps]
        y = [f"{bodypart}_y" for bodypart in bps]
        z = [f"{bodypart}_z" for bodypart in bps]
        ax.plot(
            self.df.loc[frame_idx, x],
            self.df.loc[frame_idx, y],
            self.df.loc[frame_idx, z],
            color=color,
        )

    def _add_maze_shape(self, ax: plt.Figure, paradigm: Optional[str] = None) -> None:
        if paradigm == "OTR":
            sideright = Rectangle((0, 0), 35, 30, color="red", alpha=0.4)
            sideleft = Rectangle((0, 0), 35, 30, color="red", alpha=0.4)

        if paradigm == "OTT":
            sideright = Polygon(
                np.array([[0, 0], [0, 30], [30, 0]]),
                closed=True,
                color="red",
                alpha=0.4,
            )
            sideleft = Polygon(
                np.array([[0, 0], [0, 30], [30, 0]]),
                closed=True,
                color="red",
                alpha=0.4,
            )

        if paradigm == "OTE":
            Path = mpath.Path
            path_data = [
                (Path.MOVETO, (0, 0)),
                (Path.LINETO, (0, 30)),
                # (Path.CURVE3, (1.3, 27)),
                (Path.CURVE4, (13, 11.0)),
                (Path.CURVE4, (33.8, 2.1)),
                (Path.CURVE4, (35, 1)),
                (Path.LINETO, (35, 0)),
                (Path.LINETO, (0, 0)),
            ]
            codes, verts = zip(*path_data)
            path = mpath.Path(verts, codes)
            sideright = mpatches.PathPatch(path, color="red", fill=True, alpha=0.4)
            sideleft = mpatches.PathPatch(path, color="red", fill=True, alpha=0.4)

        if paradigm != None:
            ax.add_patch(sideright)
            art3d.pathpatch_2d_to_3d(sideright, z=0, zdir="y")
            ax.add_patch(sideleft)
            art3d.pathpatch_2d_to_3d(sideleft, z=5, zdir="y")

        base = Rectangle((0, 0), 50, 5, color="gray", alpha=0.1)
        ax.add_patch(base)
        art3d.pathpatch_2d_to_3d(base, z=0, zdir="z")
        sideback = Rectangle((0, 0), 5, 30, color="gray", alpha=1)
        ax.add_patch(sideback)
        art3d.pathpatch_2d_to_3d(sideback, z=0, zdir="x")
