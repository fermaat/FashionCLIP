import os
import random
import string
import warnings
import pandas as pd
import logging as log
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageChops
from datasets import Dataset

from src.utils.init_utils import init_logging


current_path = os.getcwd()
dir = current_path.split('/')[-1]
# proof of mac!
if dir in ['research', 'dataprep', 'src']:
    current_path = '/'.join(current_path.split('/')[:-1])
PROJECT_PATH = current_path
# where the raw images are stored
DATA_PATH = f'{PROJECT_PATH}data/raw_images/dataset_name'
# where the processed images will be stored
PROCESSED_DATA_PATH = f'{PROJECT_PATH}data/processed_images_uncropped_clip'
# where the text source is stored
TEXT_PATH = f'{PROJECT_PATH}data/count_text/crsto10.txt'
# where the final dataset will be placed
DATASET_SAVE_PATH = f'{PROJECT_PATH}data/datasets/uncropped_triplet_toy_semipos'


# padding (for not placing the text too close to the border)
PADDING = 100
# Maximum size for the font to fit in the image
MAX_FONT_SIZE = 100
# first font size to try
DEFAULT_FONT_SIZE = 100
# minimum font size (otherwise there will be no font insertion)
MIN_FONT_SIZE = 40
# minimum amount of pixel length for the text
MIN_FONT_PIXELS_IN_IMAGE = 200

# how many characters will the texts have (max and min)
MIN_LENGTH_TEXTS = 20
MAX_LENGHT_TEXTS = 50


# how many (percentage) of characters will be modified for semi-positives (maximum)
REPLACE_THRESHOLD = 0.25

# different fonts to be used (ubuntu/mac)
if PROJECT_PATH.split('/')[1] == 'home':
    ALL_FONTS = ['/usr/share/fonts/smc/Meera.ttf']
else: 
    ALL_FONTS = ['/Library/Fonts/Arial Unicode.ttf']
# different colours to be used
ALL_COLOURS = ['black', 'yellow', 'red']
# how many examples will be generated per image (times colours and fonts)
NUM_EXAMPLES_PER_IMAGE = 4
# amount of different raw images to generate examples
NUM_IMAGES = 5
# amount of different texts to sample from the text source
NUM_TEXTS = 5

###################################### Triplet Generation ######################################
# amount of triplets to be generated per example
NUM_TRIPLETS_PER_EXAMPLE = 10


def get_random_position(im, image_will_be_cropped=True):
    """
    Will get a random position (pair of pixels) for an image
    """
    # the processor will crop the image, thus, removing all info from the sides. Then, the 
    if image_will_be_cropped:
        what_the_crop_will_remove = int((im.width - im.height) / 2)
        x_min = 0+PADDING + what_the_crop_will_remove
        x_max = int(im.width - PADDING) - what_the_crop_will_remove
    else:
        x_min = 0+PADDING
        x_max = int(im.width - PADDING)
    x = np.random.randint(x_min, x_max)
    y = np.random.randint(0+PADDING, int(im.height - PADDING))
    # TODO: improve for more texts
    # while logo_intersection([x, x + percentage * logo.width, y, y+ percentage * logo.height], current_image_logos):
    #     x = np.random.randint(0+PADDING, int(im.width - (PADDING  + percentage * logo.width)))
    #     y = np.random.randint(0+PADDING, int(im.height - (PADDING  + percentage * logo.height)))
    return (x, y)


def text_doesnt_fit(font_size, text_position, image_size, padding=PADDING, image_will_be_cropped=True):
    """
    This function will chacek if the text bbox fits entirely on the screen
    """
    # TODO: perhaps we could check for no interesections within the generated bboxes wih more than one text
    bbox_x, bbox_y, bbox_width, bbox_height = text_position[0], text_position[1], font_size[0], font_size[1]
    img_width, img_height = image_size

    if image_will_be_cropped:
        what_the_crop_will_remove = int((img_width - img_height) / 2)
        min_x_left = (0 + padding) + what_the_crop_will_remove
        max_x_left = img_width - what_the_crop_will_remove
    else:
        min_x_left = (0 + padding)
        max_x_left = img_width

    if bbox_x >= min_x_left and bbox_y >= (0 + padding) and ( 
        bbox_x + bbox_width + padding) <= max_x_left and (
        bbox_y + bbox_height + padding) <= img_height:
        return False
    else:
        return True
    

def get_semi_positive_example():
    """
    This function is randomly True or false. 
    It will indicate whether if we generate a 
    semi-positive exampleor a pure positive one
    """
    return np.random.randint(0,1000) % 2


def randomize_text(text, replace_threshold):
    """
    If we are going to generate a semi-positive example, some of the characters will be replaced: a maximum fraction
    of replace_threshold. (e.g: on a text with 20 chars and a percentage of 0.25, up to 5 characters will be replaced)
    Every character replacement is randomized (50% prob) so that in the end, we get both the replaced text and the 
    fraction of characters that have actually been replaced.
    """
    k = int(replace_threshold*len(text))
    random_indexes = random.sample(range(len(text)), k=k)
    result = []
    rand = 0
    list_of_chars_to_be_added = string.ascii_uppercase + string.ascii_lowercase + ' '


    for i, char in enumerate(text):
        if i in random_indexes and np.random.randint(0,1000) % 2:
            rand_operation = np.random.randint(0,999) % 3
            # Levenshtein operations, depending on a random number:
            if rand_operation == 1:
                # insertion
                result.append(char)
                result.append(random.choice(list_of_chars_to_be_added))
            elif rand_operation == 2:
                # substitution
                result.append(random.choice(list_of_chars_to_be_added))
                #else,  deletion: no chars added
            rand += 1.0
        else:
            result.append(char)
    
    return "".join(result), rand / len(text)
    # return "".join(i if random.uniform(0,1) < replace_threshold else random.choice(string.ascii_uppercase + string.ascii_lowercase) for i in text)
    

def insert_text(file_name, text, position, colour, font_name, font_given_size, image_will_be_cropped):
    """
    this function will insert text into the image, with the given
    colour, font_name and font_given size (if possible).
    Text might get randomized so that we can get semi-positive examples
    """
    original_image = Image.open(f'{DATA_PATH}/{file_name}')
    raw_image = original_image.copy()
    draw = ImageDraw.Draw(raw_image)
    font_size = min(MAX_FONT_SIZE, font_given_size)
    font = ImageFont.truetype(font_name, font_size)

    if position is None:
        position= get_random_position(raw_image, image_will_be_cropped=image_will_be_cropped)
    original_text = text
    if get_semi_positive_example():
        text, semipos = randomize_text(text, REPLACE_THRESHOLD)
    else:
        semipos = 0
    # font is reduced every time it doesn't fit
    while text_doesnt_fit(font.getsize(text), position, raw_image.size, image_will_be_cropped=image_will_be_cropped): #[0] + position[0] > raw_image.size[0]
        font_size -= 1
        font = ImageFont.truetype(font_name, font_size)
        if font_size <= MIN_FONT_SIZE:
            break
    # text will only be inserted if its size is minimally readable
    if font.getsize(text)[0] > MIN_FONT_PIXELS_IN_IMAGE:

        # Draw the actual text
        draw.text(position, text, font=font, fill=colour)
        diff = ImageChops.difference(original_image, raw_image)
        if diff.getbbox():

            file_path = f'{PROCESSED_DATA_PATH}/{colour}-{str(position)}-{file_name}'
            raw_image.save(file_path)
            caption_input = pd.DataFrame({'file_path': file_path,
                    # The text that was supposed to be inserted in the image
                    'caption': original_text,
                    # the text that was finally inserted (after modifications)
                    'actual_caption': text,
                    'font': font_name,
                    'colour': colour,
                    # fraction of modified charaters in the text
                    'semipos': semipos,
                    'position': [position],
                    'font_given': font_size})
            return caption_input
        else: 
            return pd.DataFrame()
    else: 
        return pd.DataFrame()

def generate_examples_for_image(image_name, position, image_will_be_cropped, fonts_list=ALL_FONTS,
                                colour_list=ALL_COLOURS, texts_list=[], df_images=pd.DataFrame()):
    """
    Just a multiple for loop with
    - fonts
    - colours
    - texts
    to be inserted in each image
    """
    for font_name in fonts_list:
        for colour in colour_list:
            for text in texts_list:
                caption_info = insert_text(image_name, text=text, position=position, colour=colour, font_name=font_name, 
                                           font_given_size=DEFAULT_FONT_SIZE, image_will_be_cropped=image_will_be_cropped)
                df_images = pd.concat([df_images, caption_info], ignore_index=True)
    return df_images
                

def generate_examples(image_names_list, fonts_list, colour_list, texts_list, image_will_be_cropped, num_repeats=NUM_EXAMPLES_PER_IMAGE):
    """
    Will generate num_repeats examples per example (per font, per text, per colour)
    """
    df_images=pd.DataFrame()
    for image_name in image_names_list:
        for j in range(num_repeats):
            
            df_images = generate_examples_for_image(image_name, position=None, image_will_be_cropped=image_will_be_cropped,
                                        fonts_list=fonts_list, colour_list=colour_list, texts_list=texts_list, 
                                        df_images=df_images)

    return df_images



def generate_multi_triplets(df, num_examples):
    """
    This function will generate num_examples triplets per image on the df
    """
    group2df = dict(list(df.groupby('caption')))
    group2df    
    def aux(row):
        results = []
        anchor = row.file_path
        anchor_is_semipos = row.semipos
        caption = row.caption
        actual_caption = row.actual_caption
        ids = group2df[row.caption].file_path.to_list()
        ids.remove(row.file_path)

        groups = list(group2df.keys())
        groups.remove(row.caption)
        negatives = []
        
        for _ in range(num_examples):
            positive = random.choice(ids)
            ids.remove(positive)
            semipos = df[df.file_path == positive].semipos.values[0]

        
            neg_group = random.choice(groups)
            keep_looking = True
            while keep_looking:
                negative = random.choice(group2df[neg_group].file_path.to_list())
                keep_looking = negative in negatives
            negatives.append(negative)

            results.append((anchor, positive, negative, anchor_is_semipos, semipos, caption, actual_caption))

        return results # anchor, positive, negative

    return aux



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    init_logging()

    log.info('Starting images generation')
    with open(TEXT_PATH) as f:
        lines = f.readlines()
    cleaned_lines = [l.replace('/n', '').replace('\n', '') for l in lines if (
        l.replace('/n', '').replace('\n', '') != '') and MIN_LENGTH_TEXTS < len(l) < MAX_LENGHT_TEXTS]

    all_texts = cleaned_lines
    texts_list = np.random.choice(all_texts, NUM_TEXTS)
    log.info(f'Selected texts will be the following ones: {texts_list}')

    image_names_list = [f for _, _, files in os.walk(DATA_PATH) for f in files if f.endswith('.png')][:NUM_IMAGES]
    df = generate_examples(image_names_list, fonts_list=ALL_FONTS, colour_list=ALL_COLOURS, texts_list=texts_list, image_will_be_cropped=True)
    log.info('Finished image generation')
    log.info(f'Amount of images generated: {len(df)}')
    
    
    log.info('Starting Triplets generation')
    generated_triplets = []
    part_res = df.apply(generate_multi_triplets(df, NUM_TRIPLETS_PER_EXAMPLE), axis=1).to_list()

    res = [l for p in part_res for l in p]
    df_result = pd.DataFrame(res, columns=['anchor', 'pos', 'neg', 'anchor_is_semipos', 'semipos', 'caption', 'actual_caption'])
    df_result = df_result[df_result.anchor_is_semipos == 0]
    df_result.reset_index(drop=True, inplace=True)
    df_result.drop(['anchor_is_semipos'], axis=1, inplace=True)
    log.info('Finished Triplets generation')
    log.info(f'Amount of images generated: {len(df_result)}')
    log.info('Storing the dataset')
    
    dataset = Dataset.from_pandas(df_result)
    dataset.save_to_disk(DATASET_SAVE_PATH)

    log.info('Execution Finished')
