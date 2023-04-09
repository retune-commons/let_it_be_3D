import itertools as it
import math
from typing import List, Tuple, Dict, Union, Any

import numpy as np
import pandas as pd


def get_xyz_distance_in_triangulation_space(
        marker_ids: Tuple[str, str], df_xyz: pd.DataFrame
) -> Union[pd.Series, float]:
    """
    Calculate the distance between two markers in 3D.

    Parameters
    ----------
    marker_ids: tuple of str
        The two marker_ids to calculate the distance between.
    df_xyz: pd.DataFrame
        DataFrame of triangulated data.

    Returns
    -------
    pd.Series or float
        Distance between the two markers. float, if only one frame, else pd.Series.
    """
    squared_differences = [
        (df_xyz[f"{marker_ids[0]}_{axis}"] - df_xyz[f"{marker_ids[1]}_{axis}"]) ** 2
        for axis in ["x", "y", "z"]
    ]
    return sum(squared_differences) ** 0.5


def add_reprojection_errors_of_all_calibration_validation_markers(anipose_io: Dict, df_xyz: pd.DataFrame) -> Dict:
    anipose_io["reprojection_errors_calibration_validation_markers"] = {}
    all_reprojection_errors = []
    for key in df_xyz.iloc[0].keys():
        if "error" in key:
            reprojection_error = df_xyz[key].iloc[0]
            marker_id = key[: key.find("_error")]
            anipose_io["reprojection_errors_calibration_validation_markers"][
                marker_id
            ] = reprojection_error
            if type(reprojection_error) != np.nan:
                # ToDo:
                # confirm that it would actually be a numpy nan
                # or as alternative, use something like this after blindly appending all errors to drop the nan´s:
                # anipose_io['reprojerr'][np.logical_not(np.isnan(anipose_io['reprojerr']))]
                all_reprojection_errors.append(reprojection_error)
    anipose_io["reprojection_errors_calibration_validation_markers"]["mean"] = np.asarray(
        all_reprojection_errors
    ).mean()
    return anipose_io


def set_distances_and_angles_for_evaluation(parameters: Dict, anipose_io: Dict, df_xyz: pd.DataFrame) -> Dict:
    """
    Compute angles and distances and add them to anipose_io.

    Parameters
    ----------
    parameters: dict
        ground_truth_config
    anipose_io: dict
        Containing information to validate calibration in comparison with
        ground truth.
    df_xyz:
        DataFrame of triangulated data.

    Returns
    -------
    anipose_io: dict
        Added "distances_in_cm" and "computed_angles".

    See Also
    ________
    core.utils.KEYS_TO_CHECK_PROJECT
    """
    if "distances" in parameters:
        anipose_io = _set_distances_from_configuration(
            parameters["distances"], anipose_io, df_xyz=df_xyz
        )
    else:
        print(
            "WARNING: No distances were computed. If this is unexpected, "
            "please edit the ground truth file accordingly"
        )

    if "angles" in parameters:
        anipose_io = _set_computed_angles(parameters["angles"], anipose_io, df_xyz=df_xyz)
    else:
        print(
            "WARNING: No angles were computed. If this is unexpected, "
            "please edit the ground truth file accordingly"
        )

    return anipose_io


def load_distances_from_ground_truth(distances: Dict) -> Dict:
    """ Return distances from ground_truth_config["distances"]. """
    filled_d = {}
    for key, value in distances.items():
        filled_d[key] = value
        for k, v in value.items():
            if k in filled_d.keys():
                filled_d[k][key] = v
            else:
                filled_d[k] = {}
                filled_d[k][key] = v
    return filled_d


def add_errors_between_computed_and_ground_truth_distances_for_different_references(
        anipose_io: Dict, ground_truth_distances: Dict
) -> Dict:
    """
    Calculate errors between computed distances compared to ground_truth.

    Parameters
    ----------
    anipose_io: dict
        Containing information to validate calibration in comparison with
        ground truth, such as "bodyparts".
    ground_truth_distances:
        Distances defined in ground truth as returned by
        load_distances_from_ground_truth.

    Returns
    -------
    anipose_io: dict
        Added "distance_errors_in_cm".
    """
    anipose_io = _add_distance_errors_for_different_references(anipose_io=anipose_io,
                                                               gt_distances=ground_truth_distances)
    return anipose_io


def add_errors_between_computed_and_ground_truth_angles(gt_angles: Dict, anipose_io: Dict) -> Dict:
    """
    Set the errors between the computed and ground truth angles.

    Parameters
    ----------
    gt_angles:
        Angles from ground_truth = ground_truth_config["angles"].
    anipose_io:
        Containing information to validate calibration in comparison with
        ground truth.

    Returns
    -------
    anipose_io: dict
        Added "angles_error_ground_truth_vs_triangulated".
    """
    anipose_io[
        "angles_error_ground_truth_vs_triangulated"
    ] = _compute_differences_between_triangulated_and_gt_angles(gt_angles, anipose_io)
    return anipose_io


def _get_conversion_factors_from_different_references(
        ground_truth_distances: Dict, df_xyz: pd.DataFrame
) -> Dict:
    all_conversion_factors = {}
    for reference, markers in ground_truth_distances.items():
        for m in markers:
            reference_marker_ids = (reference, m)
            distance_in_cm = ground_truth_distances[reference][m]
            reference_distance_id = reference + "_" + m
            distance_to_cm_conversion_factor = _get_xyz_to_cm_conversion_factor(
                reference_marker_ids=reference_marker_ids,
                distance_in_cm=distance_in_cm,
                df_xyz=df_xyz,
            )
            all_conversion_factors[
                reference_distance_id
            ] = distance_to_cm_conversion_factor

    return all_conversion_factors


def _get_xyz_to_cm_conversion_factor(
        reference_marker_ids: Tuple[str, str],
        distance_in_cm: Union[int, float],
        df_xyz: pd.DataFrame,
) -> float:
    distance_in_triangulation_space = get_xyz_distance_in_triangulation_space(marker_ids=reference_marker_ids,
                                                                              df_xyz=df_xyz)
    return distance_in_triangulation_space / distance_in_cm


def _add_distances_in_cm_for_each_conversion_factor(
        anipose_io: Dict, conversion_factors: Dict, df_xyz: pd.DataFrame
) -> Dict:
    anipose_io["distances_in_cm"] = {}
    for reference_distance_id, conversion_factor in conversion_factors.items():
        anipose_io["distances_in_cm"][
            reference_distance_id
        ] = _convert_all_xyz_distances(
            anipose_io=anipose_io, conversion_factor=conversion_factor, df_xyz=df_xyz
        )
    return anipose_io


def _add_distance_errors_for_different_references(anipose_io: Dict, gt_distances: Dict) -> Dict:
    anipose_io["distance_errors_in_cm"] = {}
    for reference_distance_id, triangulated_distances in anipose_io[
        "distances_in_cm"
    ].items():
        anipose_io["distance_errors_in_cm"][reference_distance_id] = {}
        marker_ids_with_distance_error = (
            _compute_differences_between_triangulated_and_gt_distances(
                triangulated_distances=triangulated_distances, gt_distances=gt_distances
            )
        )
        all_distance_errors = [
            distance_error
            for marker_id_a, marker_id_b, distance_error, percentage_error in marker_ids_with_distance_error
        ]
        mean_distance_error = np.nanmean(np.asarray(all_distance_errors))
        all_percentage_errors = [
            percentage_error
            for marker_id_a, marker_id_b, distance_error, percentage_error in marker_ids_with_distance_error
        ]
        mean_percentage_error = np.nanmean(np.asarray(all_percentage_errors))
        anipose_io["distance_errors_in_cm"][reference_distance_id] = {
            "individual_errors": marker_ids_with_distance_error,
            "mean_error": mean_distance_error,
            "mean_percentage_error": mean_percentage_error,
        }

    return anipose_io


def _set_distances_from_configuration(distances_to_compute: Dict, anipose_io: Dict, df_xyz: pd.DataFrame) -> Dict:
    conversion_factors = _get_conversion_factors_from_different_references(
        distances_to_compute, df_xyz=df_xyz
    )
    anipose_io = _add_distances_in_cm_for_each_conversion_factor(anipose_io, conversion_factors, df_xyz=df_xyz)
    return anipose_io


def _set_computed_angles(angles_to_compute: Dict, anipose_io: Dict, df_xyz: pd.DataFrame) -> Dict:
    """
    Compute angles as defined in angles_to_compute and add to anipose_io.
    """
    anipose_io["computed_angles"] = _compute_angles(angles_to_compute, df_xyz=df_xyz)
    return anipose_io


def _compute_differences_between_triangulated_and_gt_distances(
        triangulated_distances: Dict, gt_distances: Dict
) -> List[Tuple[Any, Any, Any, Any]]:
    marker_ids_with_distance_error = []
    for marker_id_a in triangulated_distances.keys():
        for marker_id_b in triangulated_distances[marker_id_a].keys():
            if (marker_id_a in gt_distances.keys()) & (
                    marker_id_b in gt_distances[marker_id_a].keys()
            ):
                ground_truth = gt_distances[marker_id_a][marker_id_b]
                triangulated_distance = triangulated_distances[marker_id_a][marker_id_b]
                distance_error = abs(ground_truth - abs(triangulated_distance))
                percentage_error = distance_error / ground_truth
                marker_ids_with_distance_error.append(
                    (marker_id_a, marker_id_b, distance_error, percentage_error)
                )

    return marker_ids_with_distance_error


def _compute_differences_between_triangulated_and_gt_angles(
        gt_angles: Dict, anipose_io: Dict
) -> Dict[str, float]:
    """
    Compute the difference between the triangulated angles
    and the provided ground truth ones.

    Parameters
    ----------
    gt_angles: dict
        ground truth angles
    anipose_io: dict
        Containing information to validate calibration in comparison with
        ground truth.

    Returns
    -------
    marker_ids_with_angles_error: dict
        Markers with angle errors.
    """
    triangulates_angles = anipose_io["computed_angles"]
    marker_ids_with_angles_error = {}
    if triangulates_angles.keys() == gt_angles.keys():
        for key in triangulates_angles:
            wrapped_tri_angle = _wrap_angles_360(triangulates_angles[key])
            angle_error = abs(gt_angles[key]["value"] - wrapped_tri_angle)
            half_pi_corrected_angle = (
                angle_error if angle_error < 180 else angle_error - 180
            )
            marker_ids_with_angles_error[key] = half_pi_corrected_angle
    else:
        raise ValueError(
            "Please check the ground truth angles passed. The angles needed are:",
            ", ".join(str(key) for key in triangulates_angles),
            "\n But the angles in the passed ground truth are:",
            ", ".join(str(key) for key in gt_angles),
        )
    return marker_ids_with_angles_error


def _wrap_angles_360(angle: float) -> float:
    """
    Wrap negative angle on 360 space.

    Parameters
    ----------
    angle: float
        Input angle.

    Returns
    -------
    float:
        Angle if positive or 360+angle if negative.
    """
    return angle if angle > 0 else 360 + angle


def _compute_angles(angles_to_compute: Dict, df_xyz: pd.DataFrame) -> Dict:
    """
    Compute the angles.

    Parameters
    ----------
    angles_to_compute: dict
        Containing definition of angles.
    df_xyz: pd.DataFrame
        DataFrame of triangulated data.

    Returns
    -------
    dict
        dictionary of the angles computed
    """
    triangulated_angles = {}
    for angle, markers_dictionary in angles_to_compute.items():
        if len(markers_dictionary["marker"]) == 3:
            pt_a = _get_vector_from_label(
                label=markers_dictionary["marker"][0], df_xyz=df_xyz
            )
            pt_b = _get_vector_from_label(
                label=markers_dictionary["marker"][1], df_xyz=df_xyz
            )
            pt_c = _get_vector_from_label(
                label=markers_dictionary["marker"][2], df_xyz=df_xyz
            )
            triangulated_angles[angle] = _get_angle_between_three_points_at_PointA(
                PointA=pt_a, PointB=pt_b, PointC=pt_c
            )
        elif len(markers_dictionary["marker"]) == 5:
            pt_a = _get_vector_from_label(
                label=markers_dictionary["marker"][2], df_xyz=df_xyz
            )
            pt_b = _get_vector_from_label(
                label=markers_dictionary["marker"][3], df_xyz=df_xyz
            )
            pt_c = _get_vector_from_label(
                label=markers_dictionary["marker"][4], df_xyz=df_xyz
            )
            plane_coord = _get_coordinates_plane_equation_from_three_points(
                PointA=pt_a, PointB=pt_b, PointC=pt_c
            )
            N = _get_vector_product(A=plane_coord[0], B=plane_coord[2])

            pt_d = _get_vector_from_label(
                label=markers_dictionary["marker"][0], df_xyz=df_xyz
            )
            pt_e = _get_vector_from_label(
                label=markers_dictionary["marker"][1], df_xyz=df_xyz
            )
            triangulated_angles[angle] = _get_angle_between_two_points_and_plane(
                PointA=pt_d, PointB=pt_e, N=N
            )
        else:
            raise ValueError(
                "Invalid number (%d) of markers to compute the angle " + angle,
                (len(markers_dictionary["marker"])),
            )
    return triangulated_angles


def _convert_all_xyz_distances(anipose_io: Dict, conversion_factor: float, df_xyz: pd.DataFrame) -> Dict:
    marker_id_combinations = it.combinations(anipose_io["bodyparts"], 2)
    all_distances_in_cm = {}
    for marker_id_a, marker_id_b in marker_id_combinations:
        if marker_id_a not in all_distances_in_cm.keys():
            all_distances_in_cm[marker_id_a] = {}
        xyz_distance = get_xyz_distance_in_triangulation_space(marker_ids=(marker_id_a, marker_id_b),
                                                               df_xyz=df_xyz)
        all_distances_in_cm[marker_id_a][marker_id_b] = xyz_distance / conversion_factor
    return all_distances_in_cm


def _get_length_in_3d_space(PointA: np.array, PointB: np.array) -> float:
    length = math.sqrt(
        (PointA[0] - PointB[0]) ** 2
        + (PointA[1] - PointB[1]) ** 2
        + (PointA[2] - PointB[2]) ** 2
    )
    return length


def _get_angle_from_law_of_cosines(
        length_a: float, length_b: float, length_c: float
) -> float:
    cos_angle = (length_c ** 2 + length_b ** 2 - length_a ** 2) / (
            2 * length_b * length_c
    )
    return math.degrees(math.acos(cos_angle))


def _get_angle_between_three_points_at_PointA(
        PointA: np.array, PointB: np.array, PointC: np.array
) -> float:
    length_c = _get_length_in_3d_space(PointA, PointB)
    length_b = _get_length_in_3d_space(PointA, PointC)
    length_a = _get_length_in_3d_space(PointB, PointC)
    return _get_angle_from_law_of_cosines(length_a, length_b, length_c)


def _get_coordinates_plane_equation_from_three_points(
        PointA: np.array, PointB: np.array, PointC: np.array
) -> np.array:
    R1 = _get_vector_from_two_points(PointA, PointB)
    R2 = _get_vector_from_two_points(PointA, PointC)
    plane_equation_coordinates = np.asarray([PointA, R1, R2])
    return plane_equation_coordinates


def _get_vector_product(A: np.array, B: np.array) -> np.array:
    N = np.asarray(
        [
            A[1] * B[2] - A[2] * B[1],
            A[2] * B[0] - A[0] * B[2],
            A[0] * B[1] - A[1] * B[0],
        ]
    )
    return N


def _get_vector_from_two_points(
        PointA: np.array, PointB: np.array
) -> np.array:
    vector = np.asarray(
        [PointA[0] - PointB[0], PointA[1] - PointB[1], PointA[2] - PointB[2]]
    )
    return vector


def _get_vector_length(vector: np.array) -> float:
    length = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    return length


def _get_angle_between_plane_and_line(N: np.array, R: np.array) -> float:
    """
    Calculate angle between plane and line.

    Parameters
    ----------
    N: np.array
        normal vector of the plane
    R: np.array
        vector between two points

    Returns
    -------
    angle: float
        Angle between N and R in degrees.
    """
    cosphi = _get_vector_length(vector=_get_vector_product(A=N, B=R)) / (
            _get_vector_length(N) * _get_vector_length(R)
    )
    phi = math.degrees(math.acos(cosphi))
    angle = 90 - phi
    return angle


def _get_angle_between_two_points_and_plane(
        PointA: np.array, PointB: np.array, N: np.array
) -> float:
    R = _get_vector_from_two_points(PointA, PointB)
    return _get_angle_between_plane_and_line(N=N, R=R)


def _get_vector_from_label(label: str, df_xyz: pd.DataFrame) -> np.array:
    return np.asarray(
        [
            df_xyz[label + "_x"],
            df_xyz[label + "_y"],
            df_xyz[label + "_z"],
        ]
    )


def _get_distance_between_plane_and_point(
        N: np.array, PointOnPlane: np.array, DistantPoint: np.array
) -> float:
    a = N[0] * PointOnPlane[0] + N[1] * PointOnPlane[1] + N[2] * PointOnPlane[2]
    distance = abs(
        N[0] * DistantPoint[0] + N[1] * DistantPoint[1] + N[2] * DistantPoint[2] - a
    ) / math.sqrt(N[0] ** 2 + N[1] ** 2 + N[2] ** 2)
    return distance
