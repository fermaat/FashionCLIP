from PIL import Image


def load_images(example, image_processor):
    """
    This function will add the anchor, pos and negative embeddings to each examples from the dataset, 
    in a pytorch tensor way. Please note all other columns are also kept
    """
    return {**{'anchor_image': image_processor.preprocess(Image.open(example['anchor']), return_tensors='pt'),
               'pos_image': image_processor.preprocess(Image.open(example['pos']), return_tensors='pt'),
               'neg_image': image_processor.preprocess(Image.open(example['neg']), return_tensors='pt')}, 
               **example}