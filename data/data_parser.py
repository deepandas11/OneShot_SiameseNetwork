import os
import json
import argparse

from random import sample 

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--fold", default="train", type=str, help="Determines which fold to parse")
parser.add_argument("--save_flag", type=str)


def gen_dictionary(base_path, split_flag):

    data = dict()

    for alphabet in os.listdir(base_path):
        data[alphabet] = dict()
        for character in os.listdir(os.path.join(base_path, alphabet)):
            character_images = os.path.join(base_path, alphabet, character)
            data[alphabet][character] = os.listdir(character_images)

    if split_flag:
        selected_alphabets = list(data.keys())
        train_alphabets = sample(selected_alphabets, int(0.8*len(selected_alphabets)))
        train_data = dict()

        for alphabet in train_alphabets:
            train_data[alphabet] = data[alphabet]
            data.pop(alphabet)

        return train_data, data 

    return data 


def gen_json_file(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f)
    print(f"{file_name} JSON File Created!")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.fold in ["train", "val"]:
        train_dict, val_dict = gen_dictionary('images_background', True)
        if args.save_flag=='save':
            gen_json_file(train_dict, "train_data.json")
            gen_json_file(val_dict, "val_data.json")
        else:
            print(train_dict.keys(), len(list(train_dict.keys())))
    else:
        eval_dict = gen_dictionary('images_evaluation', False)
        if args.save_flag=='save':
            gen_json_file(eval_dict, "eval_data.json")
        else:
            print(eval_dict.keys())

