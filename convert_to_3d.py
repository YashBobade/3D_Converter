import trimesh
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from scipy import ndimage
from skimage import measure, filters, morphology
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
import tempfile
import shutil

app = Flask(__name__)
CORS(app) 

UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_proper_3d_text(text, height=15.0, depth=5.0, font_size=72):
    """Create a plane with carved text filled with black color, oriented correctly"""
    print(f"Creating 3D text for: '{text}'")
    
    padding = 20
    img_width = len(text) * font_size + padding * 2
    img_height = font_size * 2 + padding * 2
    
    image = Image.new('L', (img_width, img_height), color=255)
    draw = ImageDraw.Draw(image)
 
    try:
        font_paths = [
            '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
            '/usr/share/fonts/TTF/Arial.ttf',
            '/Library/Fonts/Arial.ttf',
            'C:\\Windows\\Fonts\\arial.ttf',
            None
        ]
        
        font = None
        for path in font_paths:
            try:
                if path:
                    font = ImageFont.truetype(path, font_size)
                    break
            except (OSError, IOError):
                continue
                
        if font is None:
            font = ImageFont.load_default()
            print("Using default font as Arial was not found")
    except Exception as e:
        print(f"Font error: {e}")
        font = ImageFont.load_default()
        print("Using default font")
    
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (img_width - text_width) / 2
    text_y = padding 

    draw.text((text_x, text_y), text, fill=0, font=font)
    
    text_array = 255 - np.array(image)
    text_array = text_array / 255.0
    
    text_mask = text_array > 0.1 
   
    plane_depth = depth * 2 
    volume = np.ones((img_height, img_width, int(plane_depth)), dtype=np.float32) 

    carve_depth = depth 
    for z in range(int(plane_depth)):
        if z < carve_depth:
            volume[:, :, z] = np.where(text_mask, 0.0, 1.0)
        else:
            volume[:, :, z] = 1.0
    
    volume = filters.gaussian(volume, sigma=(1, 1, 0.5))
    
    print("Generating mesh with marching cubes...")
    try:
        verts, faces, _, _ = measure.marching_cubes(volume, level=0.5, spacing=(1, 1, 1))
    except Exception as e:
        print(f"Marching cubes failed: {e}")
        return None
    
    plane_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    if len(plane_mesh.vertices) == 0 or len(plane_mesh.faces) == 0:
        print("Error: Plane mesh is empty!")
        return None

    plane_mesh.vertices -= plane_mesh.vertices.mean(axis=0)

    if np.ptp(plane_mesh.vertices[:, 1]) > 0:
        scale = height / np.ptp(plane_mesh.vertices[:, 1])
        plane_mesh.vertices *= scale
    else:
        print("Warning: Plane mesh has zero width, skipping scaling")

    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.radians(90), direction=[1, 0, 0])
    plane_mesh.apply_transform(rotation_matrix)

    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.radians(180), direction=[0, 0, 1])
    plane_mesh.apply_transform(rotation_matrix)

    vertex_colors = np.zeros((len(verts), 4), dtype=np.uint8)
    light_blue = np.array([173, 216, 230, 255], dtype=np.uint8)  
    dark_blue = np.array([70, 130, 180, 255], dtype=np.uint8)    
    black = np.array([0, 0, 0, 255], dtype=np.uint8)             

    z_coords = plane_mesh.vertices[:, 2]
    z_max = z_coords.max()  
    z_carve = z_max - carve_depth * scale  
   
    y_coords = plane_mesh.vertices[:, 1] 
    if np.ptp(y_coords) > 0:
        normalized_y = (y_coords - y_coords.min()) / np.ptp(y_coords) 
        for i in range(len(vertex_colors)):
            if z_coords[i] > z_max - 0.5: 
                fraction = 1.0 - normalized_y[i] 
                color_start = light_blue.astype(np.float32)
                color_end = dark_blue.astype(np.float32)
                interpolated_color = color_start + fraction * (color_end - color_start)
                vertex_colors[i] = interpolated_color.astype(np.uint8)
            elif z_coords[i] < z_carve + 0.5:  
                vertex_colors[i] = black
            else:
                vertex_colors[i] = black
    else:
        vertex_colors[:] = light_blue 
    
    plane_mesh.visual = trimesh.visual.ColorVisuals(mesh=plane_mesh, vertex_colors=vertex_colors)

    scene = trimesh.Scene([plane_mesh])
    return scene

def create_3d_image(image_path, height=15.0, depth=10.0):
    """Create 3D model from an image with preserved colors"""
    print(f"Creating 3D model from image: {image_path}")

    original_img = Image.open(image_path).convert('RGBA')
    width, height_img = original_img.size

    if 'A' in original_img.getbands():
        alpha = np.array(original_img.split()[-1]) / 255.0
    else:
        gray_img = original_img.convert('L')
        alpha = (np.array(gray_img) < 240).astype(float)

    gray_img = original_img.convert('L')
    gray_array = np.array(gray_img) / 255.0
    depth_map = filters.sobel(gray_array) * alpha
  
    volume = np.zeros((height_img, width, int(depth)), dtype=np.float32)
    for z in range(int(depth)):
        weight = np.clip(1.0 - (z / depth) + (depth_map * 0.5), 0, 1)
        volume[:, :, z] = alpha * weight
    
    volume = filters.gaussian(volume, sigma=1)
    
    print("Generating mesh with marching cubes...")
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            volume, 
            level=0.5, 
            spacing=(1, 1, 1),
            method='lewiner'
        )
    except Exception as e:
        print(f"Marching cubes failed: {e}")
        return None
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        print("Error: Image mesh is empty!")
        return None

    mesh.vertices[:, 0] = height_img - mesh.vertices[:, 0]
    mesh.vertices -= mesh.vertices.mean(axis=0)

    if np.ptp(mesh.vertices[:, 1]) > 0:
        scale = height / np.ptp(mesh.vertices[:, 1])
        mesh.vertices *= scale

    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.radians(90), direction=[1, 0, 0])
    mesh.apply_transform(rotation_matrix)


    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.radians(180), direction=[0, 0, 1])
    mesh.apply_transform(rotation_matrix)

    uv = np.zeros((len(verts), 2))
    uv[:, 0] = verts[:, 1] / width
    uv[:, 1] = 1.0 - (verts[:, 0] / height_img)

    return mesh, original_img, uv

def text_to_3d_model(text_prompt, output_file="text_model.obj"):
    """Generate a 3D model from a text prompt and export as OBJ"""
    print(f"Creating 3D model from text: '{text_prompt}'")
    scene = create_proper_3d_text(text_prompt)
    if scene is None:
        print("Failed to create 3D model")
        return None
    
    if output_file:
        if not output_file.lower().endswith('.obj'):
            output_file += '.obj'
        
        try:
            text_mesh = scene.geometry[list(scene.geometry.keys())[0]]
            text_mesh.export(output_file, file_type='obj')
            print(f"Saved 3D model to {output_file}")
        except Exception as e:
            print(f"Failed to save model: {e}")
            return None
    return scene

def image_to_3d_model(image_path, output_file="image_model.obj"):
    """Generate a 3D model from an image and export as OBJ with texture"""
    print(f"Creating 3D model from image: {image_path}")
    result = create_3d_image(image_path)
    if result is None:
        print("Failed to create 3D model")
        return None
    
    mesh, texture_image, uv = result
    
    if output_file:
        if not output_file.lower().endswith('.obj'):
            output_file += '.obj'
        
        try:
        
            output_dir = os.path.dirname(output_file)
            output_basename = os.path.splitext(os.path.basename(output_file))[0]
            texture_filename = f"{output_basename}.png"
            mtl_filename = f"{output_basename}.mtl"
          
            texture_path = os.path.join(output_dir, texture_filename)
            texture_image.save(texture_path)
            print(f"Saved texture to {texture_path}")
            
            mtl_content = f"""# MTL file for {output_basename}.obj
newmtl material0
Ka 1.000000 1.000000 1.000000
Kd 1.000000 1.000000 1.000000
Ks 0.000000 0.000000 0.000000
Tr 0.000000
illum 1
Ns 0.000000
map_Kd {texture_filename}
"""
            mtl_path = os.path.join(output_dir, mtl_filename)
            with open(mtl_path, 'w') as f:
                f.write(mtl_content)
            print(f"Created MTL file {mtl_path}")
        
            with open(output_file, 'w') as f:
                f.write(f"# OBJ file created by 3D Image Generator\n")
                f.write(f"mtllib {mtl_filename}\n")
             
                for vertex in mesh.vertices:
                    f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
              
                for normal in mesh.vertex_normals:
                    f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
            
                for uv_coord in uv:
                    f.write(f"vt {uv_coord[0]} {uv_coord[1]}\n")
          
                f.write(f"usemtl material0\n")
                for face in mesh.faces:
              
                    v1, v2, v3 = face + 1
                    f.write(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}\n")
            
            print(f"Saved 3D model to {output_file} with texture")
            return mesh, texture_path, mtl_path
        except Exception as e:
            print(f"Failed to save model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return mesh

@app.route('/')
def serve_index():
    """Serve the index.html file"""
    return send_file('index.html')

@app.route('/api/text_to_3d', methods=['POST'])
def text_to_3d():
    data = request.get_json()
    if not data or 'text' not in data or not data['text']:
        return jsonify({'error': 'Text input is required'}), 400
    
    text = data['text']
    output_file = data.get('output_file', 'text_model')
    output_file = secure_filename(output_file)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_file + '.obj')
    
    scene = text_to_3d_model(text, output_path)
    if scene is None:
        return jsonify({'error': 'Failed to generate 3D model'}), 500
    
    return jsonify({
        'message': f'Successfully generated 3D model: {output_file}.obj',
        'file_url': f'/download/{output_file}.obj'
    })

@app.route('/api/image_to_3d', methods=['POST'])
def image_to_3d():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400
    
    output_file = request.form.get('output_file', 'image_model')
    output_file = secure_filename(output_file)

    image_filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    file.save(image_path)
    
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_file + '.obj')
    
    result = image_to_3d_model(image_path, output_path)
    if result is None:
        return jsonify({'error': 'Failed to generate 3D model'}), 500
    
    mesh, texture_path, mtl_path = result
   
    response_data = {
        'message': f'Successfully generated 3D model: {output_file}.obj',
        'file_url': f'/download/{output_file}.obj',
        'texture_url': f'/download/{os.path.basename(texture_path)}',
        'mtl_url': f'/download/{os.path.basename(mtl_path)}'
    }

    try:
        os.remove(image_path)
    except:
        pass
    
    return jsonify(response_data)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)