document.addEventListener('DOMContentLoaded', function() {
    console.log('Document loaded, initializing forms');
    const textForm = document.getElementById('textForm');
    const imageForm = document.getElementById('imageForm');
    const textResult = document.getElementById('textResult');
    const imageResult = document.getElementById('imageResult');
    const modelViewer = document.getElementById('modelViewer');
    const closeBtn = document.querySelector('.close');
    const viewerContainer = document.getElementById('viewerContainer');
    const downloadObj = document.getElementById('downloadObj');
    const downloadMtl = document.getElementById('downloadMtl');
    const downloadTexture = document.getElementById('downloadTexture');
    
    closeBtn.addEventListener('click', function() {
        console.log('Closing modal');
        modelViewer.style.display = 'none';
        viewerContainer.innerHTML = '';
    });
    
    window.addEventListener('click', function(event) {
        if (event.target === modelViewer) {
            console.log('Closing modal via background click');
            modelViewer.style.display = 'none';
            viewerContainer.innerHTML = '';
        }
    });
    
    textForm.addEventListener('submit', function(e) {
        e.preventDefault();
        console.log('Text form submitted');
        
        const text = document.getElementById('textInput').value;
        const outputFile = document.getElementById('textOutput').value || 'text_model';
        console.log('Text input:', text, 'Output file:', outputFile);
        
        textResult.innerHTML = 'Processing... Please wait.';
        textResult.className = 'result show';
        
        fetch('/api/text_to_3d', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                output_file: outputFile
            })
        })
        .then(response => {
            console.log('Text form response status:', response.status);
            return response.json();
        })
        .then(data => {
            console.log('Text form response data:', data);
            if (data.error) {
                textResult.className = 'result show error';
                textResult.innerHTML = `Error: ${data.error}`;
            } else {
                textResult.className = 'result show success';
                textResult.innerHTML = `
                    <div class="success-message">${data.message}</div>
                    <div class="model-preview">
                        <img src="${data.preview_url}?t=${Date.now()}" alt="3D Model Preview" id="textPreviewImg">
                    </div>
                    <div class="download-section">
                        <a href="${data.file_url}?dl=1" class="download-btn" download="${outputFile}.obj">Download OBJ</a>
                        <button class="download-btn view-btn" id="textViewBtn">View 3D Model</button>
                    </div>
                `;
                
                document.getElementById('textViewBtn').addEventListener('click', function() {
                    console.log('Opening model viewer for text model');
                    openModelViewer(data.file_url, null, null, outputFile);
                });
                
                document.getElementById('textPreviewImg').addEventListener('click', function() {
                    console.log('Opening model viewer via preview image click');
                    openModelViewer(data.file_url, null, null, outputFile);
                });
            }
        })
        .catch(error => {
            console.error('Text form error:', error);
            textResult.className = 'result show error';
            textResult.innerHTML = `Error: ${error.message}`;
        });
    });
    
    imageForm.addEventListener('submit', function(e) {
        e.preventDefault();
        console.log('Image form submitted');
        
        const formData = new FormData(imageForm);
        const outputFile = document.getElementById('imageOutput').value || 'image_model';
        console.log('Output file:', outputFile);
        
        imageResult.innerHTML = 'Processing... Please wait.';
        imageResult.className = 'result show';
        
        fetch('/api/image_to_3d', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log('Image form response status:', response.status);
            return response.json();
        })
        .then(data => {
            console.log('Image form response data:', data);
            if (data.error) {
                imageResult.className = 'result show error';
                imageResult.innerHTML = `Error: ${data.error}`;
            } else {
                imageResult.className = 'result show success';
                imageResult.innerHTML = `
                    <div class="success-message">${data.message}</div>
                    <div class="model-preview">
                        <img src="${data.preview_url}?t=${Date.now()}" alt="3D Model Preview" id="imagePreviewImg">
                    </div>
                    <div class="download-section">
                        <a href="${data.file_url}?dl=1" class="download-btn" download="${outputFile}.obj">Download OBJ</a>
                        <a href="${data.mtl_url}?dl=1" class="download-btn" download="${outputFile}.mtl">Download MTL</a>
                        <a href="${data.texture_url}?dl=1" class="download-btn" download="${outputFile}.png">Download Texture</a>
                        <button class="download-btn view-btn" id="imageViewBtn">View 3D Model</button>
                    </div>
                `;
                
                document.getElementById('imageViewBtn').addEventListener('click', function() {
                    console.log('Opening model viewer for image model');
                    openModelViewer(data.file_url, data.mtl_url, data.texture_url, outputFile);
                });
                
                document.getElementById('imagePreviewImg').addEventListener('click', function() {
                    console.log('Opening model viewer via preview image click');
                    openModelViewer(data.file_url, data.mtl_url, data.texture_url, outputFile);
                });
            }
        })
        .catch(error => {
            console.error('Image form error:', error);
            imageResult.className = 'result show error';
            imageResult.innerHTML = `Error: ${error.message}`;
        });
    });
    
    function openModelViewer(objUrl, mtlUrl, textureUrl, filename) {
        console.log('Opening model viewer', { objUrl, mtlUrl, textureUrl, filename });
        modelViewer.style.display = 'block';
        
        downloadObj.href = `${objUrl}?dl=1`;
        downloadObj.download = `${filename}.obj`;
        downloadObj.style.display = 'block';
        
        if (mtlUrl) {
            downloadMtl.href = `${mtlUrl}?dl=1`;
            downloadMtl.download = `${filename}.mtl`;
            downloadMtl.style.display = 'block';
        } else {
            downloadMtl.style.display = 'none';
        }
        
        if (textureUrl) {
            downloadTexture.href = `${textureUrl}?dl=1`;
            downloadTexture.download = `${filename}.png`;
            downloadTexture.style.display = 'block';
        } else {
            downloadTexture.style.display = 'none';
        }
        
        viewerContainer.innerHTML = '';
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);
        
        const camera = new THREE.PerspectiveCamera(75, viewerContainer.clientWidth / viewerContainer.clientHeight, 0.1, 1000);
        camera.position.z = 30;
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(viewerContainer.clientWidth, viewerContainer.clientHeight);
        viewerContainer.appendChild(renderer.domElement);
        
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);
        
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight2.position.set(-1, -1, -1);
        scene.add(directionalLight2);
        
        if (mtlUrl) {
            console.log('Loading OBJ with MTL');
            const mtlLoader = new THREE.MTLLoader();
            mtlLoader.crossOrigin = 'anonymous';
            mtlLoader.load(mtlUrl + '?t=' + Date.now(), function(materials) {
                materials.preload();
                
                const objLoader = new THREE.OBJLoader();
                objLoader.setMaterials(materials);
                objLoader.load(objUrl + '?t=' + Date.now(), function(object) {
                    const box = new THREE.Box3().setFromObject(object);
                    const center = box.getCenter(new THREE.Vector3());
                    object.position.sub(center);
                    
                    const size = box.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const scale = 20 / maxDim;
                    object.scale.set(scale, scale, scale);
                    
                    scene.add(object);
                    setupControls(camera, renderer.domElement, object);
                }, undefined, function(error) {
                    console.error('Error loading OBJ:', error);
                    viewerContainer.innerHTML = '<div class="error">Error loading 3D model</div>';
                });
            }, undefined, function(error) {
                console.error('Error loading MTL:', error);
                loadObjOnly(objUrl, scene, camera, renderer);
            });
        } else {
            console.log('Loading OBJ only');
            loadObjOnly(objUrl, scene, camera, renderer);
        }
        
        window.addEventListener('resize', function() {
            camera.aspect = viewerContainer.clientWidth / viewerContainer.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(viewerContainer.clientWidth, viewerContainer.clientHeight);
        });
    }
    
    function loadObjOnly(objUrl, scene, camera, renderer) {
        const objLoader = new THREE.OBJLoader();
        objLoader.load(objUrl + '?t=' + Date.now(), function(object) {
            const box = new THREE.Box3().setFromObject(object);
            const center = box.getCenter(new THREE.Vector3());
            object.position.sub(center);
            
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 20 / maxDim;
            object.scale.set(scale, scale, scale);
            
            object.traverse(function(child) {
                if (child instanceof THREE.Mesh) {
                    child.material = new THREE.MeshPhongMaterial({
                        color: 0x3498db,
                        specular: 0x111111,
                        shininess: 30
                    });
                }
            });
            
            scene.add(object);
            setupControls(camera, renderer.domElement, object);
        }, undefined, function(error) {
            console.error('Error loading OBJ:', error);
            viewerContainer.innerHTML = '<div class="error">Error loading 3D model</div>';
        });
    }
    
    function setupControls(camera, domElement, object) {
        let rotationSpeed = 0.01;
        let autoRotate = true;
        
        function animate() {
            requestAnimationFrame(animate);
            
            if (autoRotate && object) {
                object.rotation.y += rotationSpeed;
            }
            
            renderer.render(scene, camera);
        }
        
        animate();
        
        domElement.addEventListener('mousedown', function() {
            autoRotate = false;
        });
        
        domElement.addEventListener('touchstart', function() {
            autoRotate = false;
        });
        
        let isDragging = false;
        let previousMousePosition = { x: 0, y: 0 };
        
        domElement.addEventListener('mousedown', function(e) {
            isDragging = true;
            previousMousePosition = {
                x: e.clientX,
                y: e.clientY
            };
        });
        
        document.addEventListener('mouseup', function() {
            isDragging = false;
        });
        
        document.addEventListener('mousemove', function(e) {
            if (isDragging && object) {
                const deltaX = e.clientX - previousMousePosition.x;
                const deltaY = e.clientY - previousMousePosition.y;
                
                object.rotation.y += deltaX * 0.01;
                object.rotation.x += deltaY * 0.01;
                
                previousMousePosition = {
                    x: e.clientX,
                    y: e.clientY
                };
            }
        });
    }
});
