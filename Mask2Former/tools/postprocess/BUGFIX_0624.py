import json
import os

def process_json_files(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(subdir, file)
                with open(json_file_path, "r") as input_f:
                    json_info = json.load(input_f)

                new_dict = {}
                for shape_type in json_info:
                    list_a = json_info[shape_type]
                    sorted_list_a = sorted(list_a, key=lambda x: x["img_path"].split('/')[-2] + '_' + os.path.basename(x["img_path"]))

                    new_list = []
                    # Reordering img_path and seg_path according to sorted seg_path
                    for original, sorted_item in zip(list_a, sorted_list_a):
                        sorted_item['vis_path'] = original.get('vis_path', None)
                        sorted_item['image_score'] = original.get('image_score', None)
                        if sorted_item['vis_path']:
                            sorted_item['vis_path'] = "/mnt/ve_share/songyuhao/seg_cleanlab/res/" + "/".join(sorted_item['vis_path'].split("/")[-3:])
                        new_list.append(original)
                    
                    new_dict[shape_type] = new_list

                output_file_path = os.path.join(subdir, file.replace(".json", "_processed.json"))
                with open(output_file_path, 'w') as f:
                    json.dump(new_dict, f, indent=4)

# 示例调用
root_directory = "/mnt/ve_share/songyuhao/seg_cleanlab/res/test"
process_json_files(root_directory)
