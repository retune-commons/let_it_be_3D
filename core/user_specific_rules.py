from .utils import convert_to_path


def user_specific_rules_on_videometadata(videometadata: "VideoMetadata"):
    if videometadata.filepath.suffix == ".AVI":
        videometadata.cam_id = "Top"
        if "00" in videometadata.filepath.name[-7:]:
            idx00=videometadata.filepath.name.index("00", -7)
            idx_number=videometadata.filepath.name[idx00+2]
            new_filepath = videometadata.filepath.with_name(
                videometadata.filepath.name.replace(
                    videometadata.filepath.name[idx00:idx00+3],
                    f"_00{idx_number}",
                )
            )

            videometadata.filepath.rename(new_filepath)
            videometadata.filepath = new_filepath
        
def user_specific_rules_on_triangulation_calibration_videos(videometadata: "VideoMetadata"):
    if videometadata.filepath.suffix == ".AVI":
        replacerstring = ""
        if "top" not in videometadata.filepath.name.lower():
            replacerstring = "_Top"
        if "00" in videometadata.filepath.name[-7:]:
            new_filepath = videometadata.filepath.with_name(
                videometadata.filepath.name.replace(
                    videometadata.filepath.name[
                        videometadata.filepath.name.index(
                            "00", -7
                        )-1 : videometadata.filepath.name.index("00", -7)+3
                    ],
                    replacerstring,
                )
            )

            videometadata.filepath.rename(new_filepath)
            videometadata.filepath = new_filepath
