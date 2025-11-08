from nltk.tokenize import word_tokenize


classes = [
    "background",
    "aerop lane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv monitor",
]

palette = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128]
]
palette = [value for color in palette for value in color]






def get_indices(tokenizer, classes, prompt, class_prompt):
    class_prompt_ids = tokenizer(class_prompt.split(";")[-1]).input_ids
    prompt_ids = tokenizer(prompt.split(";")[0]).input_ids
    words_prompt = [tok.replace("</w>","") for tok in tokenizer.convert_ids_to_tokens(prompt_ids)][1:-1]
    words = [tok.replace("</w>","") for tok in tokenizer.convert_ids_to_tokens(class_prompt_ids)][1:-1]
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
                    curr_indices.append(idx)
                    curr_labels.append(i_cls)
    return [curr_indices], [curr_labels]




