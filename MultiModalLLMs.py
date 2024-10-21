# %% [markdown]
# <a href="https://colab.research.google.com/github/vijaygwu/SEAS8525/blob/main/Class_6_MultiModalLLMs.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
!pip install Pillow requests

# %%
import requests
from PIL import Image
from io import BytesIO

# %%
from urllib.request import urlopen




# %%
# URL of the image
url = 'https://cdn.pixabay.com/photo/2021/01/17/07/35/dog-5924174_1280.jpg'

# %%
# Send a GET request to the URL
response = requests.get(url)
image = Image.open(BytesIO(response.content))




# %%
image.show()

# %%
caption = "a puppy playing in the snow"

# %%
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

model_id = "openai/clip-vit-base-patch32"

# Load a tokenizer to preprocess the text
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)

# Load a processor to preprocess the images
processor = CLIPProcessor.from_pretrained(model_id)

# Main model for generating text and image embeddings
model = CLIPModel.from_pretrained(model_id)


# %%
inputs = tokenizer(caption, return_tensors="pt"); inputs

# %%
tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# %%
# Create a text embedding
text_embedding = model.get_text_features(**inputs)
text_embedding.shape

# %%
# Preprocess image
processed_image = processor(text=None, images=image, return_tensors='pt')['pixel_values']
processed_image.shape


# %%
import numpy as np
import matplotlib.pyplot as plt


# Prepare image for visualization
img = processed_image.squeeze(0).T
img = np.einsum('ijk->jik', img)

# Visualize preprocessed image
plt.imshow(img)
plt.axis('off')

# %%
# Create the image embedding
image_embedding = model.get_image_features(processed_image)
image_embedding.shape

# %%
# Calculate the probability of the text belonging to the image
text_probs = (100.0 * image_embedding @ text_embedding.T).softmax(dim=-1)
text_probs

# %%
# Normalize the embeddings
text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

# Calculate their similarity
text_embedding = text_embedding.detach().cpu().numpy()
image_embedding = image_embedding.detach().cpu().numpy()
score = np.dot(text_embedding, image_embedding.T)
score

# %%
!pip install -U sentence-transformers

# %%


from sentence_transformers import SentenceTransformer, util

# Load SBERT-compatible CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# Encode the images
image_embeddings = model.encode(image)

# Encode the captions
text_embeddings = model.encode(caption)

#Compute cosine similarities
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)
print(sim_matrix)

# %%
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

# Load processor and main model
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

# Send the model to GPU to speed up inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# %%
from urllib.request import urlopen
from PIL import Image

# Load a wide image
link = "https://images.unsplash.com/photo-1524602010760-6eb67b3c63a0?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2631&q=80"
image = Image.open(urlopen(link)).convert("RGB")
image

# %%
import numpy as np

np.array(image).shape

# %%
inputs = processor(image, return_tensors="pt").to(device, torch.float16)
inputs["pixel_values"].shape

# %%
processor.tokenizer

# %%
# Preprocess the text
text = "Her vocalization was remarkably melodic"
token_ids = processor(image, text=text, return_tensors="pt").to(device, torch.float16)["input_ids"][0]

# Convert input ids back to tokens
tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)

# %%
inputs = processor(image, return_tensors="pt").to(device, torch.float16)
inputs["pixel_values"].shape



# %%
processor.tokenizer

# %%
# Preprocess the text
text = "Her vocalization was remarkably melodic"
token_ids = processor(image, text=text, return_tensors="pt").to(device, torch.float16)["input_ids"][0]

# Convert input ids back to tokens
tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)

# %%
tokens = [token.replace("Ġ", "_") for token in tokens]
tokens

# %%
import requests
from PIL import Image
from io import BytesIO

# %%


 # URL of the image
url = 'https://cdn.pixabay.com/photo/2021/01/17/07/35/dog-5924174_1280.jpg'

# Send a GET request to the URL
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")


# Convert an image into inputs and preprocess it
inputs = processor(image, return_tensors="pt").to(device, torch.float16)
image

# %%
# Generate token ids using the full BLIP-2 model
generated_ids = model.generate(**inputs, max_new_tokens=20)

# Convert the token ids to text
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# %%
print(generated_text)

# %%
# Load rorschach image
url = "https://upload.wikimedia.org/wikipedia/commons/7/70/Rorschach_blot_01.jpg"
image = Image.open(urlopen(url)).convert("RGB")

# Generate caption
inputs = processor(image, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# %%
 print(generated_text)

# %%
# URL of the image
url = "https://upload.wikimedia.org/wikipedia/commons/7/70/Rorschach_blot_01.jpg"

# Send a GET request to the URL
response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")


inputs = processor(image, return_tensors="pt").to(device, torch.float16)

# %%
# Visual Question Answering
prompt = "Question: Write down what you see in this picture. Answer:"

# Process both the image and the prompt
inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

# Generate text
generated_ids = model.generate(**inputs, max_new_tokens=30)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# %%
print(generated_text)

# %%
from IPython.display import HTML, display
import ipywidgets as widgets

def text_eventhandler(*args):
  question = args[0]["new"]
  if question:
    args[0]["owner"].value = ""

    # Create prompt
    if not memory:
      prompt = " Question: " + question + " Answer:"
    else:
      template = "Question: {} Answer: {}."
      prompt = " ".join([template.format(memory[i][0], memory[i][1]) for i in range(len(memory))]) + " Question: " + question + " Answer:"

    # Generate text
    inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().split("Question")[0]

    # Update memory
    memory.append((question, generated_text))

    # Assign to output
    output.append_display_data(HTML("<b>USER:</b> " + question))
    output.append_display_data(HTML("<b>BLIP-2:</b> " + generated_text))
    output.append_display_data(HTML("<br>"))

# Prepare widgets
in_text = widgets.Text()
in_text.continuous_update = False
in_text.observe(text_eventhandler, "value")
output = widgets.Output()
memory = []

# Display chat box
display(
    widgets.VBox(
        children=[output, in_text],
        layout=widgets.Layout(display="inline-flex", flex_flow="column-reverse"),
    )
)


