textures_mapping = {
    "Diffuse": 3,
    "Normal": 3,
    "Roughness": 1,
    "Specular": 3
}

texture_maps = list(textures_mapping.keys())

def validate_textures(textures):
    if len([x for x in textures if x not in textures_mapping.keys()]) > 0:
        raise Exception(
            f"Requested maps must be in: {list(textures_mapping.keys())}")