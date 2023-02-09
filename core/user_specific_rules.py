from .utils import convert_to_path

def user_specific_rules_on_videometadata(videometadata: "VideoMetadata"):
    if videometadata.filepath.suffix == '.AVI':
        videometadata.cam_id = "Top"
        try:
            new_filepath = convert_to_path(videometadata.filepath.name.replace(
                videometadata.filepath.name[
                    videometadata.filepath.name.index("00") : videometadata.filepath.name.index("00") + 3
                ],
                "",
            ))
            videometadata.filepath.rename(new_filepath)
        except:
            pass