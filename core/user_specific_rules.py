from .utils import convert_to_path

def user_specific_rules_on_videometadata(videometadata: "VideoMetadata"):
    if videometadata.filepath.suffix == '.AVI':
        videometadata.cam_id = "Top"
        if "00" in videometadata.filepath.name:
            new_filepath = videometadata.filepath.with_name(videometadata.filepath.name.replace(
                videometadata.filepath.name[
                    videometadata.filepath.name.index("00") : videometadata.filepath.name.index("00") + 3
                ],
                "",
            ))
            videometadata.filepath.rename(new_filepath)
            videometadata.filepath = new_filepath