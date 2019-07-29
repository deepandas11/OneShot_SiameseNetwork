import os
import json
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--fold", default="train", type=str, help="Determines which fold to parse")
parser.add_argument("--save_flag", type=str)


def gen_dictionary(fold):
    data = dict()
    file_name = ''
    if fold == "train":
        base_path = "images_background"
        file_name = 'train_'
    else:
        base_path = "images_evaluation"
        file_name = 'test_'

    class_count = 0

    for alphabet in os.listdir(base_path):
        for character in os.listdir(os.path.join(base_path, alphabet)):
            for image in os.listdir(os.path.join(base_path, alphabet, character)):
                character_class = image.split('_')[0]
                file_path = os.path.join(alphabet, character, image)
                data[file_path] = character_class
            class_count += 1

    return data, file_name


def gen_json_file(data, file_name):
    save_file = file_name+'data.json'
    with open(save_file, 'w') as f:
        json.dump(data, f)
    print("JSON File Created!")


if __name__ == "__main__":
    args = parser.parse_args()
    data, file_name = gen_dictionary(args.fold)
    print("Save mode is : ", args.save_flag)
    if args.save_flag=='save':
        gen_json_file(data, file_name)
    else:
        print(data)
