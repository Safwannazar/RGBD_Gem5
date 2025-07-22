# ProjectRGBDGem5

## Overview
Stereo RGB-D depth estimation and alignment system with deep learning comparison and visualization. Includes traditional stereo matching, MiDaS DL depth, anaglyph generation, and batch GIFs.


## How to Run (Windows)

1. Unzip the folder
2. Open terminal in folder

3. Create and activate virtual environment:
python -m venv RGBD
RGBD\Scripts\activate.bat

4. Install required packages:
pip install -r requirement.txt

5. Setting up the fps needed :
gif_fps = 6 (Increase more faster)

6. Run the main script:
python RGBD.py


## How to Run on Google Colab

1. Upload `ProjectRGBDGem5.zip` to your Google Drive
2. Use the following code in a Colab notebook to unzip and install:
```python
from google.colab import drive
drive.mount('/content/drive')

zip_path = '/content/drive/MyDrive/ProjectRGBDGem5.zip'
extract_path = '/content/ProjectRGBDGem5'

import zipfile, os
os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

%cd /content/ProjectRGBDGem5
!pip install -r requirement.txt  

3. Run the main script:
    !python RGBD.py