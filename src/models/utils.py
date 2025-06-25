from PIL import Image

def image_grid(imgs, rows, cols):
    """
    Will generate a grid of images to be displayed in a rows*cols grid
    """
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    # grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def display_triplet(dataset, ind):
    """
    Assuming dataset is a dataset of triplets, the function will display a grid of images
    with anchor, pos and negative in that order. It will also print the semipos coefficient
    """
    element = dataset[ind]
    images = [Image.open(element['anchor']), 
              Image.open(element['pos']), 
              Image.open(element['neg'])]
    print(element['semipos'])
    return image_grid(images, rows=1, cols=len(images))