import argparse

import json
import os

from tqdm import tqdm

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='', help='')

    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_opt()

    json_path = opt.path

    conf_thres = 0.02

    while True:
        print('checking conf_thres ', conf_thres)

        if os.path.exists('tmp.json'):
            os.remove('tmp.json')

        with open(json_path) as f:
            json_data = json.load(f)

            jdict = []

            for det in json_data:
                if det['score'] >= conf_thres:
                    jdict.append(det)

            with open('tmp.json', 'w') as f:
                json.dump(jdict, f)
        
        json_size = os.path.getsize('tmp.json') / (1000.0 * 1000.0)

        print('json_size ', json_size)

        if json_size < 20:
            break
        
        conf_thres = round(conf_thres + 0.001, 3)
