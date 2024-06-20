from tqdm import tqdm

img_path_path = ["/mnt/ve_share/songyuhao/generation/data/lane_output_0/img_paths.txt", "/mnt/ve_share/songyuhao/generation/data/lane_output_1/img_paths.txt", "/mnt/ve_share/songyuhao/generation/data/lane_output_2/img_paths.txt" , "/mnt/ve_share/songyuhao/generation/data/lane_output_3/img_paths.txt", "/mnt/ve_share/songyuhao/generation/data/lane_output_4/img_paths.txt"]
new_img_path_path = ["/mnt/ve_share/songyuhao/generation/data/lane_output_0/new_img_paths.txt", "/mnt/ve_share/songyuhao/generation/data/lane_output_1/new_img_paths.txt", "/mnt/ve_share/songyuhao/generation/data/lane_output_2/new_img_paths.txt", "/mnt/ve_share/songyuhao/generation/data/lane_output_3/new_img_paths.txt", "/mnt/ve_share/songyuhao/generation/data/lane_output_4/new_img_paths.txt"]


def read_multi(multi_paths: list) -> list:
    result = []
    for path in tqdm(multi_paths):
        with open(path) as input_file:
            result += [_.strip() for _ in input_file]
    return result

def sort_by_name(lst: list) -> list:
    sorted_lst = sorted(lst, key=lambda _: _.split("/")[-1])
    return sorted_lst

if __name__ == "__main__":
    origin_paths = read_multi(img_path_path)
    origin_paths = [_[1:] for _ in origin_paths]
    sorted_origin = sort_by_name(origin_paths)
    
    with open("origin.txt", "w") as origin_file:
        for origin in tqdm(sorted_origin):
            origin_file.writelines(origin + "\n")
    
    new_paths = read_multi(new_img_path_path)
    sorted_new = sort_by_name(new_paths)
    
    with open("gen.txt", "w") as gen_file:
        for new in tqdm(sorted_new):
            gen_file.writelines(new + "\n")

    print(len(origin_paths))
    print(len(set(origin_paths)))
    print(sorted_origin[:10])
    print(len(sorted_new))
    print(len(set(sorted_new)))
    print(sorted_new[:10])
    