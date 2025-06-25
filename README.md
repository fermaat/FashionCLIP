# FashionCLIP: Fine-tuning CLIP for semantic fashion image retrieval using advanced metric learning. (Codename: 🎸 The-Seeker 🎸)
This project demonstrates an advanced deep learning pipeline for fine-tuning a CLIP-based vision model on a custom-generated fashion dataset. The primary goal is to teach the model a nuanced understanding of semantic similarity between images, enabling it to function as a powerful engine for fashion image retrieval and thematic embedding extraction.

The core of this work lies in using a sophisticated Triplet Loss with Semi-Positives to train an image encoder. This method teaches the model to differentiate between subtle and significant variations in style, a critical task in the fashion domain.

## Key Features
- Advanced Metric Learning: Implements a custom TripletSemiPosMarginWithDistanceLoss to learn a rich embedding space. This loss function goes beyond the standard triplet loss by incorporating "semi-positive" examples, allowing for a more fine-grained understanding of similarity.
- Automated Data Generation: A robust data preparation pipeline (src/data_prep.py) programmatically generates a large-scale dataset of image triplets (Anchor, Positive, Negative) and semi-positives from a base set of images, providing full control over the training data's characteristics.
- CLIP-Based Architecture: Leverages the power of OpenAI's CLIP model by using its vision encoder as the base for our ImageEncoderNetwork, which is then fine-tuned on the specialized task.
- Structured & Reusable Code: The project is organized into clear and reusable components, separating data preparation, dataset handling, model definition, training loops, and loss functions into distinct modules.
- Experiment Tracking: Integrated with TensorBoard for visualizing high-dimensional embeddings and monitoring training progress, a crucial practice for deep learning research and development.

## Methodology
The project's methodology is based on Metric Learning, which aims to learn a function that maps inputs to an embedding space where "similar" inputs are close together and "dissimilar" inputs are far apart.

## 1. Data Generation

To learn a nuanced similarity metric, a specialized dataset is required. The data_prep.py script generates this dataset by:

- Taking a set of source images.
- Programmatically creating variations by adding text with different fonts, colors, and content.
- Constructing triplets for training:
- Anchor: A reference image.
- Positive: A highly similar image (e.g., the same base image with a different font color).
- Negative: A distinctly different image.
- Semi-Positive: An image that is similar to the anchor in some ways but different in others (e.g., same base image but with a completely different text caption).
## 2. Model Architecture

The model is a Siamese-like network. The same core image encoder processes the anchor, positive, and negative images to generate their respective embeddings. The encoder itself is built upon the pre-trained CLIPVisionModel, allowing us to leverage its powerful, general-purpose features as a starting point for fine-tuning.
```bash
      +------------------+
      |  Anchor Image    |
      +------------------+
              |
              V
      +------------------+
      |  Image Encoder   |  <--- (CLIP Vision Model)
      |  (Shared Weights)|
      +------------------+
              |
              V
      +------------------+
      | Anchor Embedding |
      +------------------+

      (Same process for Positive, Negative, and Semi-Positive images)
```
## 3. Training with Triplet Loss

The network is trained using a custom triplet loss function defined in losses.py. The standard triplet loss aims to minimize the distance between the anchor and positive while maximizing the distance between the anchor and negative. The formula is:
```math
L(a, p, n) = \max \left( d(E_a, E_p) - d(E_a, E_n) + \alpha,\, 0 \right)
```
Where:

$E_a$, $E_p$, $E_n$ are the embeddings for the anchor, positive, and negative.
$d$ is a distance function (e.g., Euclidean distance).
$\alpha$ is the margin.
This project enhances this by adding a semi-positive term, which helps the model learn finer-grained distinctions. The loss encourages the semi-positive to be further away than the positive but closer than the negative.

## Project Structure
The repository is organized into research notebooks and production-ready scripts:

- **/notebooks**: Contains the original research and experiments.
    - `01_generate_triplet_input.ipynb`: Initial exploration of data generation.
    - `02_triplet_training_new_metric.ipynb`: Research on improving the training metric/loss.
- **/src**: Contains the cleaned, modular, and reusable code.
    - `data_prep.py`: The main script for generating the triplet dataset.
    - `datasets.py`: Pytorch Dataset utilities for loading the data.
    - `models.py`: Defines the ImageEncoderNetwork and the main learning_loop.
    - `losses.py`: Contains the custom TripletSemiPosMarginWithDistanceLoss function.
    - `tensorboard_viz.py`: Tools for visualizing embeddings with TensorBoard.
    - `utils.py`: Helper functions for visualization and other tasks.

## How to Use

### 1. Setup

First, clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/fashion-clip.git
cd fashion-clip
pip install -r requirements.txt
```

### 2. Data Preparation

It might be more interesting to tryout the notebook examples, but you can always use the code directly from the python scripts. Place your source images in the data/raw_images directory (you may need to create this). Then, run the data preparation script to generate the triplets dataset.

```bash
python src/data_prep.py
```
This will populate the data/processed_images directory with the generated triplets and create a final dataset file.

### 3. Training

To start fine-tuning the model, run the training script. This project's training logic is encapsulated in the notebooks, which can be adapted into a train.py script.

(Example from 02_triplet_training_new_metric.ipynb)

```python
# (Inside a training script)
# 1. Load the dataset created by data_prep.py
# 2. Initialize the ImageEncoderNetwork from models.py
# 3. Initialize the TripletSemiPosMarginWithDistanceLoss from losses.py
# 4. Set up an optimizer (e.g., AdamW)
# 5. Call the learning_loop function from models.py

history = learning_loop(
    model=model,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    # ... other parameters
)
```
The best performing model weights will be saved to `best.pt`.

## Results and Evaluation

The model was trained for 30 epochs, showing a consistent decrease in both training and validation loss, which indicates that the model was successfully learning a meaningful embedding space. The training was monitored with early stopping, saving the best model based on the validation loss.

### Training Performance

The model's performance improved steadily throughout the training process. The validation loss dropped from **1.079** in the first epoch to **0.273** in the final epoch, demonstrating effective learning and generalization to unseen data.

| Epoch | Train Loss | Validation Loss |
| :---: | :--------: | :-------------: |
| 1     | 1.060      | 1.079           |
| 10    | 0.614      | 0.684           |
| 20    | 0.273      | 0.393           |
| 30    | 0.178      | **0.273** |

The final saved model (`best.pt`) represents the point of optimal performance before any potential overfitting.

### Qualitative Results: Semantic Search

Although it is not inclued on the repository, a nice model application can be tested (code will be pasted below):

The application allows for two types of queries:
1.  **Text-to-Image Search**: A user can input a text caption (e.g., "dog on the beach"), and the system retrieves the most visually similar images from a pre-computed database of embeddings.
2.  **Image-to-Image Search**: A user can upload an image, and the system finds the most similar images from the database based on visual content and style.

```python
def search_text(query, top_k=10):
    """" Search an image based on the text query.
    
    Args:
        query ([string]): query you want search for
        top_k (int, optional): Amount of images o return]. Defaults to 1.
    Returns:
        [list]: list of images with captions that are related to the query.
        [list]: list of images that are related to the query.
        [list]: list of captions with the images that are related to the query.
        [time]: start time of marking relevance of the images.
    """
    # logging.info(f"[SearchText]: Searching for {query} with top_k={top_k}...")
    
    # First, we encode the query.
    inputs = tokenizer([query],  padding=True, return_tensors="pt")
    with torch.no_grad():
        query_emb = model.get_text_features(**inputs)
        query_emb /= query_emb.norm(p=2, dim=-1, keepdim=True)
        

    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = semantic_search(query_emb, im_embeds, top_k=top_k)[0]

    image_caption = []
    images = []
    captions = []
    scores = []
    for hit in hits:
        # print(hit)
        # print(paths[hit['corpus_id']])
        object = Image.open(os.path.join(
            "photos/", paths[hit['corpus_id']]))
        caption = annotations[hit['corpus_id']]
        image_caption.append((object, caption))
        images.append(object)
        captions.append(caption)
        scores.append(hit['score'])

    curr_time = time.time()
    # logging.info(f"[SearchText]: Found {len(image_caption)} images at "
    #              f"{time.ctime(curr_time)}.")
    return image_caption, images, captions, scores, curr_time



def search_image(image_query, top_k=10):
    """" Search an image based on the text query.
    
    Args:
        image query ([image]): image query you want search for
        top_k (int, optional): Amount of images o return]. Defaults to 1.
    Returns:
        [list]: list of images with captions that are related to the query.
        [list]: list of images that are related to the query.
        [list]: list of captions with the images that are related to the query.
        [time]: start time of marking relevance of the images.
    """
    # logging.info(f"[SearchText]: Searching for {query} with top_k={top_k}...")
    
    # First, we encode the query.
    inputs = image_processor.preprocess(image_query, return_tensors='pt')
    with torch.no_grad():
        query_emb = model.get_image_features(**inputs)
        query_emb /= query_emb.norm(p=2, dim=-1, keepdim=True)
        

    # Then, we use the util.semantic_search function, which computes the cosine-similarity
    # between the query embedding and all image embeddings.
    # It then returns the top_k highest ranked images, which we output
    hits = semantic_search(query_emb, im_embeds, top_k=top_k)[0]

    image_caption = []
    images = []
    captions = []
    scores = []
    for hit in hits:
        # print(hit)
        # print(paths[hit['corpus_id']])
        object = Image.open(os.path.join(
            "photos/", paths[hit['corpus_id']]))
        caption = annotations[hit['corpus_id']]
        image_caption.append((object, caption))
        images.append(object)
        captions.append(caption)
        scores.append(hit['score'])

    curr_time = time.time()
    # logging.info(f"[SearchText]: Found {len(image_caption)} images at "
    #              f"{time.ctime(curr_time)}.")
    return image_caption, images, captions, scores, curr_time

BASEWIDTH = 800
def show_small(im, caption=None, base_width=BASEWIDTH):
    if caption is not None:
        print(caption)
    wpercent = (base_width/float(im.size[0]))
    hsize = int((float(im.size[1])*float(wpercent)))
    return im.resize((base_width,hsize), Image.Resampling.LANCZOS)



def search_image_from_path(path, top_k):
    print(path)
    image_query = Image.open(path)
    search_image(image_query, top_k0)


```

Please note the above code has been totalluy inspired in results from HF2 course, for instance, this one snippet [repo](https://github.com/marcelcastrobr/huggingface_course2/blob/main/SageMaker/clip_sagemaker_huggingface.ipynb)


📄 License

MIT License. See LICENSE for more details.

📬 Contact
For questions, collaborations, or feedback, feel free to reach out:

📧 Email: fermaat.vl@gmail.com
🧑‍💻 GitHub: [@fermaat](https://github.com/fermaat)
🌐 [Website](https://fermaat.github.io)
