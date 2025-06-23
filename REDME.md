3D Model Generator
Overview
This project converts text prompts (e.g., "Model Text") or images (e.g., PNG files) into 3D models in OBJ format. The generated models are double-sided, vertically oriented, and can include textures (for images) or vertex colors (for text). The script supports visualization using Trimesh's built-in viewer and is designed to work in both local environments and Pyodide (with limitations in Pyodide). The output is an OBJ file that can be opened in 3D viewers like Blender or MeshLab.

Dependencies
Python 3.8+
Libraries (listed in requirements.txt):
numpy
matplotlib
Pillow
scipy
scikit-image
trimesh
Install the dependencies using:
bash
Copy
pip install -r requirements.txt
Usage
The script supports two input types:

Text Input: Converts a text string into a 3D model with carved text and colored surfaces (light blue to dark blue gradient with black text).
Image Input: Converts an image into a 3D model with texture mapping, using the image's alpha channel or grayscale for depth.
Running the Script
Save the script as model_generator.py.
