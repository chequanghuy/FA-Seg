classes=['background', 
        'aerop lane', 
        'bag', 
        'bed', 
        'bed clothes', 
        'bench', 
        'bicycle', 
        'bird', 
        'boat', 
        'book', 
        'bottle', 
        'building', 
        'bus', 
        'cabinet', 
        'car', 
        'cat', 
        'ceiling', 
        'chair', 
        'cloth', 
        'compu ter', 
        'cow', 
        'cup', 
        'curtain', 
        'dog', 
        'door', 
        'fence', 
        'floor', 
        'flower', 
        'food', 
        'grass', 
        'ground', 
        'horse', 
        'keyboard', 
        'light', 
        'motorbike', 
        'mountain', 
        'mouse', 
        'person', 
        'plate', 
        'platform', 
        'potted plant', 
        'road', 
        'rock', 
        'sheep', 
        'shelves', 
        'sidewalk', 
        'sign', 
        'sky', 
        'snow', 
        'sofa', 
        'table', 
        'track', 
        'train', 
        'tree', 
        'truck', 
        'tv monitor', 
        'wall', 
        'water', 
        'window', 
        'wood']



palette=[[0, 0, 0], 
        [180, 120, 120], 
        [6, 230, 230], 
        [80, 50, 50], 
        [4, 200, 3], 
        [120, 120, 80], 
        [140, 140, 140], 
        [204, 5, 255], 
        [230, 230, 230], 
        [4, 250, 7], 
        [224, 5, 255], 
        [235, 255, 7], 
        [150, 5, 61], 
        [120, 120, 70], 
        [8, 255, 51], 
        [255, 6, 82], 
        [143, 255, 140], 
        [204, 255, 4], 
        [255, 51, 7], 
        [204, 70, 3], 
        [0, 102, 200], 
        [61, 230, 250], 
        [255, 6, 51], 
        [11, 102, 255], 
        [255, 7, 71], 
        [255, 9, 224], 
        [9, 7, 230], 
        [220, 220, 220], 
        [255, 9, 92], 
        [112, 9, 255], 
        [8, 255, 214], 
        [7, 255, 224], 
        [255, 184, 6], 
        [10, 255, 71], 
        [255, 41, 10], 
        [7, 255, 255], 
        [224, 255, 8], 
        [102, 8, 255], 
        [255, 61, 6], 
        [255, 194, 7], 
        [255, 122, 8], 
        [0, 255, 20], 
        [255, 8, 41], 
        [255, 5, 153], 
        [6, 51, 255], 
        [235, 12, 255], 
        [160, 150, 20], 
        [0, 163, 255], 
        [140, 140, 140], 
        [250, 10, 15], 
        [20, 255, 0], 
        [31, 255, 0], 
        [255, 31, 0], 
        [255, 224, 0], 
        [153, 255, 0], 
        [0, 0, 255], 
        [255, 71, 0], 
        [0, 235, 255], 
        [0, 173, 255], 
        [31, 0, 255]]



palette = [value for color in palette for value in color]

def get_indices(tokenizer, classes, prompt, class_prompt):
    class_prompt_ids = tokenizer(class_prompt.split(";")[-1]).input_ids
    prompt_ids = tokenizer(prompt.split(";")[0]).input_ids
    words_prompt = [tok.replace("</w>","") for tok in tokenizer.convert_ids_to_tokens(prompt_ids)][1:-1]
    words = [tok.replace("</w>","") for tok in tokenizer.convert_ids_to_tokens(class_prompt_ids)][1:-1]
    # print(words.)
    curr_indices, curr_labels = [], [0]
    for i_cls, cls in enumerate(classes):
        if i_cls == 0:
            continue
        cls_lower = cls.lower()

        if len(cls_lower.split()) > 1:
            cls_words = cls_lower.split()
            for idx in range(len(words) - len(cls_words) + 1):
                if words[idx:idx + len(cls_words)] == cls_words:
                    curr_indices.append(list(range(idx, idx + len(cls_words))))
                    curr_labels.append(i_cls)
        else:
            for idx, word in enumerate(words):
                if word == cls_lower:
                    if idx != len(words)-1:
                        if (word == 'bed' and words[idx+1]=='clothes'):
                            continue
                    curr_indices.append(idx)
                    curr_labels.append(i_cls)
    return [curr_indices], [curr_labels]
