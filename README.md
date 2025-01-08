# CODECRAFT_GA_1
TEXT GENERATION WITH GPT-2

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Check if CUDA (GPU) is available, otherwise use CPU
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer for the GPT-2 model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load the GPT-2 model and set the pad token ID to the EOS token ID to avoid warnings
model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)

# Encode the context that the generation is conditioned on
model_inputs = tokenizer('I enjoy walking with my cute dog', return_tensors='pt').to(torch_device)

# Generate 40 new tokens using greedy search
greedy_output = model.generate(**model_inputs, max_new_tokens=40)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# Activate beam search and early stopping
beam_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    num_beams=5,
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# Set no_repeat_ngram_size to 2 to avoid repeating n-grams of size 2
beam_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# Set return_num_sequences > 1 to generate multiple sequences
beam_outputs = model.generate(
    **model_inputs,
    max_new_tokens=40,
    num_beams=5,
    no_repeat_ngram_size=2,
    num_return_sequences=5,
    early_stopping=True
)

# Print multiple output sequences
print("Output:\n" + 100 * '-')
for i, beam_output in enumerate(beam_outputs):
    print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))

# Set seed to reproduce results. Feel free to change the seed to get different results
from transformers import set_seed
set_seed(42)

# Activate sampling and deactivate top_k by setting top_k sampling to 0
sample_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# Use temperature to decrease the sensitivity to low probability candidates
sample_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=0,
    temperature=0.6,
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# Set top_k to 50
sample_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=50
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# Set top_k to 50 and top_p to 0.92
sample_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_p=0.92,
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# Set top_k to 50, top_p to 0.95, and num_return_sequences to 3
sample_outputs = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3,
)

# Print multiple output sequences
print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
!pip install -q aitextgen

import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

from aitextgen import aitextgen
from aitextgen.colab import mount_gdrive, copy_file_from_gdrive
ImportError                               Traceback (most recent call last)
<ipython-input-2-7042b0d17636> in <cell line: 10>()
      8     )
      9 
---> 10 from aitextgen import aitextgen
     11 from aitextgen.colab import mount_gdrive, copy_file_from_gdrive
/usr/local/lib/python3.10/dist-packages/aitextgen/aitextgen.py in <module>
     12 import torch
     13 from pkg_resources import resource_filename
---> 14 from pytorch_lightning.plugins import DeepSpeedPlugin
     15 from tqdm.auto import trange
     16 from transformers import (

ImportError: cannot import name 'DeepSpeedPlugin' from 'pytorch_lightning.plugins' (/usr/local/lib/python3.10/dist-packages/pytorch_lightning/plugins/__init__.py)

---------------------------------------------------------------------------
NOTE: If your import is failing due to a missing package, you can
manually install dependencies using either !pip or !apt.

To view examples of installing some common dependencies, click the
"Open Examples" button below.
---------------------------------------------------------------------------
Requirement already satisfied: matplotlib-venn in /usr/local/lib/python3.10/dist-packages (1.1.1)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from matplotlib-venn) (3.8.0)
Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from matplotlib-venn) (1.26.4)
Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from matplotlib-venn) (1.13.1)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (1.3.1)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (4.55.3)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (1.4.7)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (24.2)
Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (11.0.0)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (3.2.0)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib->matplotlib-venn) (2.8.2)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib->matplotlib-venn) (1.17.0)
E: Package 'libfluidsynth1' has no installation candidate
# https://pypi.python.org/pypi/libarchive
!apt-get -qq install -y libarchive-dev && pip install -U libarchive
import libarchive
Selecting previously unselected package libarchive-dev:amd64.
(Reading database ... 123632 files and directories currently installed.)
Preparing to unpack .../libarchive-dev_3.6.0-1ubuntu1.3_amd64.deb ...
Unpacking libarchive-dev:amd64 (3.6.0-1ubuntu1.3) ...
Setting up libarchive-dev:amd64 (3.6.0-1ubuntu1.3) ...
Processing triggers for man-db (2.10.2-1) ...
Collecting libarchive
  Downloading libarchive-0.4.7.tar.gz (23 kB)
  Preparing metadata (setup.py) ... done
Collecting nose (from libarchive)
  Downloading nose-1.3.7-py3-none-any.whl.metadata (1.7 kB)
Downloading nose-1.3.7-py3-none-any.whl (154 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 154.7/154.7 kB 4.0 MB/s eta 0:00:00
Building wheels for collected packages: libarchive
  error: subprocess-exited-with-error
  
  × python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
  ╰─> See above for output.
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  Building wheel for libarchive (setup.py) ... error
  ERROR: Failed building wheel for libarchive
  Running setup.py clean for libarchive
Failed to build libarchive
ERROR: ERROR: Failed to build installable wheels for some pyproject.toml based projects (libarchive)
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
<ipython-input-3-5bee6841787f> in <cell line: 3>()
      1 # https://pypi.python.org/pypi/libarchive
      2 get_ipython().system('apt-get -qq install -y libarchive-dev && pip install -U libarchive')
----> 3 import libarchive

ModuleNotFoundError: No module named 'libarchive'

---------------------------------------------------------------------------
NOTE: If your import is failing due to a missing package, you can
manually install dependencies using either !pip or !apt.

To install libarchive, click the button below.
---------------------------------------------------------------------------
# https://pypi.python.org/pypi/libarchive
!apt-get -qq install -y libarchive-dev && pip install -U libarchive
import libarchive
# https://pypi.python.org/pypi/pydot
!apt-get -qq install -y graphviz && pip install pydot
import pydot
Requirement already satisfied: pydot in /usr/local/lib/python3.10/dist-packages (3.0.3)
Requirement already satisfied: pyparsing>=3.0.9 in /usr/local/lib/python3.10/dist-packages (from pydot) (3.2.0)
!pip install cartopy
import cartopy
Collecting cartopy
  Downloading Cartopy-0.24.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.9 kB)
Requirement already satisfied: numpy>=1.23 in /usr/local/lib/python3.10/dist-packages (from cartopy) (1.26.4)
Requirement already satisfied: matplotlib>=3.6 in /usr/local/lib/python3.10/dist-packages (from cartopy) (3.8.0)
Requirement already satisfied: shapely>=1.8 in /usr/local/lib/python3.10/dist-packages (from cartopy) (2.0.6)
Requirement already satisfied: packaging>=21 in /usr/local/lib/python3.10/dist-packages (from cartopy) (24.2)
Requirement already satisfied: pyshp>=2.3 in /usr/local/lib/python3.10/dist-packages (from cartopy) (2.3.1)
Requirement already satisfied: pyproj>=3.3.1 in /usr/local/lib/python3.10/dist-packages (from cartopy) (3.7.0)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6->cartopy) (1.3.1)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6->cartopy) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6->cartopy) (4.55.3)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6->cartopy) (1.4.7)
Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6->cartopy) (11.0.0)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6->cartopy) (3.2.0)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.6->cartopy) (2.8.2)
Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from pyproj>=3.3.1->cartopy) (2024.12.14)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib>=3.6->cartopy) (1.17.0)
Downloading Cartopy-0.24.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (11.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.7/11.7 MB 79.9 MB/s eta 0:00:00
Installing collected packages: cartopy
Successfully installed cartopy-0.24.1
Wed Jan  8 14:50:38 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000000:00:04.0 Off |                    0 |
| N/A   41C    P8               9W /  70W |      0MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
Loading GPT-2 or GPT Neo
If you're retraining a model on new text, you need to download and load the GPT-2 model into the GPU.

There are several sizes of GPT-2:

124M (default): the "small" model, 500MB on disk.
355M (default): the "medium" model, 1.5GB on disk.
774M (default): the "large" model, 3GB on disk.
You can also finetune a GPT Neo model instead, which is more suitable for longer texts and the base model has more recent data:

125M: Analogous to the GPT-2 124M model.
350M: Analogous to the GPT-2 355M model
The next cell downloads the model and saves it in the Colaboratory VM. If the model has already been downloaded, running this cell will reload it.
ai = aitextgen(tf_gpt2="124M", to_gpu=True)

# Comment out the above line and uncomment the below line to use GPT Neo instead.
# ai = aitextgen(model="EleutherAI/gpt-neo-125M", to_gpu=True)
NameError                                 Traceback (most recent call last)
<ipython-input-4-ff67d6c09688> in <cell line: 1>()
----> 1 ai = aitextgen(tf_gpt2="124M", to_gpu=True)
      2 
      3 # Comment out the above line and uncomment the below line to use GPT Neo instead.
      4 # ai = aitextgen(model="EleutherAI/gpt-neo-125M", to_gpu=True)

NameError: name 'aitextgen' is not defined
Mounting Google Drive
The best way to get input text to-be-trained into the Colaboratory VM, and to get the trained model out of Colaboratory, is to route it through Google Drive first.

Running this cell (which will only work in Colaboratory) will mount your personal Google Drive in the VM, which later cells can use to get data in/out. (it will ask for an auth code; that auth is not saved anywhere)
mount_gdrive()
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-5-8eab94bacc50> in <cell line: 1>()
----> 1 mount_gdrive()

NameError: name 'mount_gdrive' is not defined
Uploading a Text File to be Trained to Colaboratory
In the Colaboratory Notebook sidebar on the left of the screen, select Files. From there you can upload files:
file_name = "input.txt"
copy_file_from_gdrive(file_name)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-10-9a43db37b55f> in <cell line: 1>()
----> 1 copy_file_from_gdrive(file_name)

NameError: name 'copy_file_from_gdrive' is not defined
ai.train(file_name,
         line_by_line=False,
         from_cache=False,
         num_steps=3000,
         generate_every=1000,
         save_every=1000,
         save_gdrive=False,
         learning_rate=1e-3,
         fp16=False,
         batch_size=1, 
         )
         ---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-11-0a7ce79b21b3> in <cell line: 1>()
----> 1 ai.train(file_name,
      2          line_by_line=False,
      3          from_cache=False,
      4          num_steps=3000,
      5          generate_every=1000,

NameError: name 'ai' is not defined
from_folder = None

for file in ["pytorch_model.bin", "config.json"]:
  if from_folder:
    copy_file_from_gdrive(file, from_folder)
  else:
    copy_file_from_gdrive(file)
    ---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-12-b12050bb0bab> in <cell line: 3>()
      5     copy_file_from_gdrive(file, from_folder)
      6   else:
----> 7     copy_file_from_gdrive(file)

NameError: name 'copy_file_from_gdrive' is not defined
ai = aitextgen(model_folder=".", to_gpu=True)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-13-f08c4b21de6f> in <cell line: 1>()
----> 1 ai = aitextgen(model_folder=".", to_gpu=True)

NameError: name 'aitextgen' is not defined
ai = aitextgen(model_folder="trained_model", to_gpu=True)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-14-7b8c155d7988> in <cell line: 1>()
----> 1 ai = aitextgen(model_folder="trained_model", to_gpu=True)

NameError: name 'aitextgen' is not defined
ai.generate()
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-15-999192644495> in <cell line: 1>()
----> 1 ai.generate()

NameError: name 'ai' is not defined
ai.generate(n=5,
            batch_size=5,
            prompt="ROMEO:",
            max_length=256,
            temperature=1.0,
            top_p=0.9)
            ---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-17-79341f383eb5> in <cell line: 1>()
----> 1 ai.generate(n=5,
      2             batch_size=5,
      3             prompt="ROMEO:",
      4             max_length=256,
      5             temperature=1.0,

NameError: name 'ai' is not defined
num_files = 5

for _ in range(num_files):
  ai.generate_to_file(n=1000,
                     batch_size=50,
                     prompt="ROMEO:",
                     max_length=256,
                     temperature=1.0,
                     top_p=0.9)
                     ---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-18-b438da90e9cd> in <cell line: 3>()
      2 
      3 for _ in range(num_files):
----> 4   ai.generate_to_file(n=1000,
      5                      batch_size=50,
      6                      prompt="ROMEO:",

NameError: name 'ai' is not defined
