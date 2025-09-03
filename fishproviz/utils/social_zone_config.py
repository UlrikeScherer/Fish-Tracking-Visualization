import fishproviz.config as config
import os
import json
import codecs


def read_social_zone_data_from_json():
    """Reads the novel object zone data from the json-file and returns a dictionary with the
    data for each fish.
    """
    maze_dict = {}
    for file in os.listdir(config.OBJECT_ZONE_COORDS_PATH):
        if file.endswith(".json"):
            if not '_'.join([file.split('_')[0], "front"]) in maze_dict:
                maze_dict['_'.join([file.split('_')[0], "front"])] = {}
                maze_dict['_'.join([file.split('_')[0], "back"])] = {}
            # with open(os.path.join(config.OBJECT_ZONE_COORDS_PATH, file), "r") as f:
            with codecs.open(os.path.join(config.OBJECT_ZONE_COORDS_PATH, file), "r", encoding='utf-8',
                             errors='ignore') as f:
                file_data = json.load(f)
                for data in file_data:
                    try:
                        if not '_'.join(file.split('_')[1:3]).split('.')[0] in maze_dict['_'.join([file.split('_')[0], data['comment'].split('_')[0]])]:
                            maze_dict['_'.join([file.split('_')[0], data['comment'].split('_')[0]])]['_'.join(file.split('_')[1:3]).split('.')[0]] = {}
                        maze_dict['_'.join([file.split('_')[0], data['comment'].split('_')[0]])]['_'.join(file.split('_')[1:3]).split('.')[0]][data['comment'].split('_')[1]] = data

                    except json.decoder.JSONDecodeError as e:
                        print(f"Something wrong with filename {file}, raises json.decoder.JSONDecodeError")
                        raise e
    return maze_dict

