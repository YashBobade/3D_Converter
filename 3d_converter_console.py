import trimesh
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from scipy import ndimage
from skimage import measure, filters, morphology

def create_proper_3d_text(text, height=15.0, depth=5.0, font_size=72):
    """Create a plane with carved text filled with black color, oriented vertically, double-sided"""
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
        verts, faces, normals, _ = measure.marching_cubes(volume, level=0.5, spacing=(1, 1, 1))
    except Exception as e:
        print(f"Marching cubes failed: {e}")
        return None

    plane_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    if len(plane_mesh.vertices) == 0 or len(plane_mesh.faces) == 0:
        print("Error: Plane mesh is empty!")
        return None


    new_verts = np.zeros_like(verts)
    new_verts[:, 0] = verts[:, 1] 
    new_verts[:, 1] = verts[:, 2]  
    new_verts[:, 2] = img_height - verts[:, 0] 

    new_normals = np.zeros_like(normals)
    new_normals[:, 0] = normals[:, 1]  
    new_normals[:, 1] = normals[:, 2]
    new_normals[:, 2] = normals[:, 0]

    plane_mesh.vertices = new_verts
    plane_mesh.vertex_normals = new_normals

    plane_mesh.vertices -= plane_mesh.vertices.mean(axis=0)

    if np.ptp(plane_mesh.vertices[:, 2]) > 0:
        scale = height / np.ptp(plane_mesh.vertices[:, 2])
        plane_mesh.vertices *= scale
        scaled_normals = plane_mesh.vertex_normals.copy() * scale
        plane_mesh.vertex_normals = scaled_normals
    else:
        print("Warning: Plane mesh has zero height, skipping scaling")

    back_verts = plane_mesh.vertices.copy()
    back_verts[:, 1] = -back_verts[:, 1] 
    back_normals = -plane_mesh.vertex_normals 
    back_faces = plane_mesh.faces[:, [0, 2, 1]] 

    combined_verts = np.vstack([plane_mesh.vertices, back_verts])
    combined_faces = np.vstack([
        plane_mesh.faces,
        back_faces + len(plane_mesh.vertices)
    ])
    combined_normals = np.vstack([plane_mesh.vertex_normals, back_normals])

    final_mesh = trimesh.Trimesh(
        vertices=combined_verts,
        faces=combined_faces,
        vertex_normals=combined_normals
    )

    vertex_colors = np.zeros((len(combined_verts), 4), dtype=np.uint8)
    light_blue = np.array([173, 216, 230, 255], dtype=np.uint8)  
    dark_blue = np.array([70, 130, 180, 255], dtype=np.uint8)    
    black = np.array([0, 0, 0, 255], dtype=np.uint8)             

    x_coords = final_mesh.vertices[:, 0]
    z_coords = final_mesh.vertices[:, 2]
    z_max = z_coords.max()  
    z_carve = z_max - carve_depth * scale  
   
    if np.ptp(x_coords) > 0:
        normalized_x = (x_coords - x_coords.min()) / np.ptp(x_coords) 
        for i in range(len(vertex_colors)):
            if z_coords[i] > z_max - 0.5: 
                fraction = 1.0 - normalized_x[i] 
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
    
    final_mesh.visual = trimesh.visual.ColorVisuals(mesh=final_mesh, vertex_colors=vertex_colors)

    scene = trimesh.Scene([final_mesh])
    return scene

def create_3d_image(image_path, height=15.0, depth=10.0):
    """Create 3D model from an image with correct vertical orientation and double-sided texture"""
    print(f"Creating 3D model from image: {image_path}")

    original_img = Image.open(image_path).convert('RGBA')
    width, height_img = original_img.size

    if 'A' in original_img.getbands():
        alpha = np.array(original_img.split()[-1]) / 255.0
    else:
        gray_img = original_img.convert('L')
        alpha = (np.array(gray_img) < 240).astype(float)
    
    img_array = np.array(original_img)[:, :, :3]
    img_array = img_array / 255.0

    gray_img = original_img.convert('L')
    gray_array = np.array(gray_img) / 255.0
    depth_map = filters.sobel(gray_array) * alpha

    volume_depth = int(depth)
    volume = np.zeros((height_img, width, volume_depth), dtype=np.float32)
    for z in range(volume_depth):
      
        if z == 0:
            volume[:, :, z] = alpha
        else:
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
 
    new_verts = np.zeros_like(verts)
    new_verts[:, 0] = verts[:, 1]
    new_verts[:, 1] = verts[:, 2] 
    new_verts[:, 2] = height_img - verts[:, 0]
  
    new_normals = np.zeros_like(normals)
    new_normals[:, 0] = normals[:, 1] 
    new_normals[:, 1] = normals[:, 2] 
    new_normals[:, 2] = normals[:, 0]  
    
    mesh.vertices = new_verts
    mesh.vertex_normals = new_normals
 
    mesh.vertices -= mesh.vertices.mean(axis=0)

    if np.ptp(mesh.vertices[:, 2]) > 0:
        scale = height / np.ptp(mesh.vertices[:, 2])
        mesh.vertices *= scale
        scaled_normals = mesh.vertex_normals.copy() * scale
        mesh.vertex_normals = scaled_normals
 
    uv = np.zeros((len(verts), 2))
    uv[:, 0] = verts[:, 1] / width
    uv[:, 1] = 1.0 - (verts[:, 0] / height_img)  
    

    texture = trimesh.visual.texture.TextureVisuals(
        uv=uv,
        image=original_img
    )
    mesh.visual = texture
   
    back_verts = mesh.vertices.copy()
    back_verts[:, 1] = -back_verts[:, 1]  
    back_normals = -mesh.vertex_normals  
    back_faces = mesh.faces[:, [0, 2, 1]]  
    back_uv = uv.copy() 

    combined_verts = np.vstack([mesh.vertices, back_verts])
    combined_faces = np.vstack([
        mesh.faces,
        back_faces + len(mesh.vertices)  
    ])
    combined_normals = np.vstack([mesh.vertex_normals, back_normals])
    combined_uv = np.vstack([uv, back_uv])
   
    final_mesh = trimesh.Trimesh(
        vertices=combined_verts,
        faces=combined_faces,
        vertex_normals=combined_normals
    )

    final_mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=combined_uv,
        image=original_img
    )
    
    return final_mesh

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
    return scene

def image_to_3d_model(image_path, output_file="image_model.obj"):
    """Generate a 3D model from an image and export as OBJ"""
    print(f"Creating 3D model from image: {image_path}")
    mesh = create_3d_image(image_path)
    if mesh is None:
        print("Failed to create 3D model")
        return None
    
    if output_file:
        if not output_file.lower().endswith('.obj'):
            output_file += '.obj'
        
        try:
            texture_image = mesh.visual.material.image
            texture_path = os.path.splitext(output_file)[0] + '.png'
            texture_image.save(texture_path)
            
            material = trimesh.visual.material.SimpleMaterial(image=texture_image)
            mesh.visual.material = material
            
            mesh.export(output_file, file_type='obj', include_texture=True)
            print(f"Saved 3D model to {output_file} with texture at {texture_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
    return mesh

def show_3d_model(mesh):
    """Display the 3D model or scene directly"""
    if mesh is None:
        print("No mesh to display")
        return
    
    print("Displaying 3D model...")
    try:
        mesh.show()
    except Exception as e:
        print(f"Could not open 3D viewer: {e}")
        print("The 3D model file was created successfully.")
        print("You can open it with a 3D viewer like Blender or MeshLab.")

def main():
    """Main function to create 3D text or image"""
    print("=" * 50)
    print("3D MODEL GENERATOR")
    print("Convert text or image to 3D models")
    print("=" * 50)
    
    choice = input("Choose input type (1 for text, 2 for image, default 1): ").strip()
    if choice == '2':
        image_path = input("Enter image path (e.g., car_image.png): ").strip()
        if not os.path.exists(image_path):
            print("Image file not found!")
            return
        output_file = input("Enter output file path (default: image_model): ").strip() or "image_model"
        mesh = image_to_3d_model(image_path, output_file)
        show_3d_model(mesh)
    else:
        text_input = input("Enter text to convert to 3D (default: Model Text): ").strip() or "Model Text"
        output_file = input("Enter output file path (default: text_model): ").strip() or "text_model"
        scene = text_to_3d_model(text_input, output_file)
        show_3d_model(scene)

if __name__ == "__main__":
    main()