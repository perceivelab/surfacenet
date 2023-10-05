textures_mapping = {
    "diffuse": 3,
    "normal": 3,
    "roughness": 1,
    "specular": 3
}

texture_maps = list(textures_mapping.keys())

def validate_textures(textures):
    if len([x for x in textures if x not in textures_mapping.keys()]) > 0:
        raise Exception(
            f"Requested maps must be in: {list(textures_mapping.keys())}")