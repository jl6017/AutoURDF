import {
    WebGLRenderer,
    PerspectiveCamera,
    Scene,
    Mesh,
    PlaneGeometry,
    ShadowMaterial,
    DirectionalLight,
    PCFSoftShadowMap,
    sRGBEncoding,
    Color,
    AmbientLight,
    Box3,
    LoadingManager,
    MeshPhongMaterial,
    Vector3,
    GridHelper,
    MathUtils,
    Raycaster,
    Vector2,
    Quaternion,
    Matrix4,
    DoubleSide,
    CylinderGeometry,
    Group,
    ArrowHelper,
    CanvasTexture,
    RepeatWrapping
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import URDFLoader from 'URDFLoader';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';

class URDFViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = new Scene();
        this.scene.background = new Color(0xffffff);

        // Initialize camera
        this.camera = new PerspectiveCamera(75, this.container.clientWidth / this.container.clientHeight, 0.1, 1000);
        this.camera.position.set(10, 10, 10);
        this.camera.lookAt(0, 0, 0);

        // Initialize renderer
        this.renderer = new WebGLRenderer({ antialias: true });
        this.renderer.outputEncoding = sRGBEncoding;
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = PCFSoftShadowMap;
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        // Setup lights
        const directionalLight = new DirectionalLight(0xffffff, 1.0);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.setScalar(1024);
        directionalLight.position.set(5, 30, 5);
        this.scene.add(directionalLight);

        const ambientLight = new AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);

        // Setup PyBullet style ground
        // Create a proper PyBullet-style checkerboard ground
        const groundSize = 30; // Large ground plane
        const groundGeometry = new PlaneGeometry(groundSize, groundSize, 1, 1);
        
        // PyBullet's exact colors
        const darkColor = new Color(0.18, 0.31, 0.31); // Dark teal-ish
        const lightColor = new Color(0.7, 0.7, 0.7);   // Light gray
        
        // Create a checkerboard pattern in the material
        const groundMaterial = new MeshPhongMaterial({
            vertexColors: false,
            shininess: 0
        });
        
        // Set UV coordinates for proper texture mapping
        groundGeometry.computeBoundingBox();
        
        // Create checkerboard texture programmatically
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const context = canvas.getContext('2d');
        
        // Draw the checkerboard pattern
        const squareSize = 64; // Size of each checkerboard square
        for (let x = 0; x < canvas.width; x += squareSize) {
            for (let y = 0; y < canvas.height; y += squareSize) {
                // Determine color based on position to create checkerboard
                const isEvenRow = Math.floor(x / squareSize) % 2 === 0;
                const isEvenCol = Math.floor(y / squareSize) % 2 === 0;
                
                if (isEvenRow === isEvenCol) {
                    context.fillStyle = '#2a5698'; // Lighter, more vibrant blue
                } else {
                    context.fillStyle = '#b3b3b3'; // Light gray
                }
                
                context.fillRect(x, y, squareSize, squareSize);
            }
        }
        
        // Create texture from canvas
        const texture = new CanvasTexture(canvas);
        texture.wrapS = RepeatWrapping;
        texture.wrapT = RepeatWrapping;
        texture.repeat.set(5, 5); // Adjust repetition to match PyBullet scale
        
        // Apply texture to material
        groundMaterial.map = texture;
        
        const ground = new Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = 0;
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Setup controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.minDistance = 0.5;
        this.controls.target.y = 1;
        this.controls.update();

        // Setup for joint manipulation
        this.raycaster = new Raycaster();
        this.mouse = new Vector2();
        this.selectedJoint = null;
        this.activeController = null;
        this.manipulationActive = false;
        this.jointValues = {};
        this.joints = {};
        this.jointControllers = new Group();
        this.scene.add(this.jointControllers);
        this.ignoreLimits = false;
        this.showVisuals = true;
        this.highlightColor = "#ff8800";
        this.controlsVisible = false;
        
        // Add tooltip element
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'joint-tooltip';
        this.tooltip.style.display = 'none';
        this.tooltip.style.position = 'absolute';
        this.tooltip.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
        this.tooltip.style.color = 'white';
        this.tooltip.style.padding = '5px 10px';
        this.tooltip.style.borderRadius = '4px';
        this.tooltip.style.fontSize = '14px';
        this.tooltip.style.pointerEvents = 'none';
        this.tooltip.style.zIndex = '1000';
        this.container.appendChild(this.tooltip);
        this.hoveredJoint = null;

        // Add event listeners for interaction
        this.container.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.container.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.container.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.container.addEventListener('mouseleave', this.onMouseUp.bind(this));
        this.container.addEventListener('mousemove', this.onHover.bind(this));

        // Bind methods
        this.animate = this.animate.bind(this);
        this.onResize = this.onResize.bind(this);
        
        // Add event listeners
        window.addEventListener('resize', this.onResize);
        
        // Create the joint control panel
        this.createControlPanel();
        
        // Start animation loop
        this.animate();
    }

    createControlPanel() {
        // Create control panel container
        const panel = document.createElement('div');
        panel.id = 'joint-control-panel';
        panel.style.position = 'absolute';
        panel.style.top = '10px';
        panel.style.right = '10px';
        panel.style.width = '250px';
        panel.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
        panel.style.padding = '10px';
        panel.style.borderRadius = '5px';
        panel.style.maxHeight = '80vh';
        panel.style.overflowY = 'auto';
        panel.style.boxShadow = '0 2px 5px rgba(0,0,0,0.1)';
        panel.style.border = '1px solid #e0e0e0';
        
        // Add panel header with controls
        const header = document.createElement('div');
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.marginBottom = '10px';
        
        const title = document.createElement('h3');
        title.textContent = 'Joint Controls';
        title.style.margin = '0';
        title.style.color = '#333';
        
        const toggleBtn = document.createElement('button');
        toggleBtn.textContent = 'Hide';
        toggleBtn.style.cursor = 'pointer';
        toggleBtn.addEventListener('click', () => {
            this.sliderContainer.style.display = this.sliderContainer.style.display === 'none' ? 'block' : 'none';
            toggleBtn.textContent = this.sliderContainer.style.display === 'none' ? 'Show' : 'Hide';
        });
        
        header.appendChild(title);
        header.appendChild(toggleBtn);
        panel.appendChild(header);
        
        // Container for joint sliders
        this.sliderContainer = document.createElement('div');
        panel.appendChild(this.sliderContainer);
        
        // Reset button
        const resetBtn = document.createElement('button');
        resetBtn.textContent = 'Reset All Joints';
        resetBtn.style.width = '100%';
        resetBtn.style.marginTop = '10px';
        resetBtn.style.padding = '5px';
        resetBtn.addEventListener('click', () => {
            this.resetAllJoints();
        });
        panel.appendChild(resetBtn);
        
        this.container.appendChild(panel);
    }

    resetAllJoints() {
        Object.keys(this.joints).forEach(jointName => {
            this.setJointValue(jointName, 0);
        });
        this.updateSliders();
    }

    loadURDF(urdfPath) {
        console.log('Starting to load URDF from path:', urdfPath);
        if (this.robot) {
            console.log('Removing existing robot from scene');
            this.scene.remove(this.robot);
            
            // Clear existing joint controls
            this.joints = {};
            this.jointValues = {};
            this.sliderContainer.innerHTML = '';
            
            // Remove 3D controllers
            while (this.jointControllers.children.length) {
                this.jointControllers.remove(this.jointControllers.children[0]);
            }
        }

        const manager = new LoadingManager();
        const loader = new URDFLoader(manager);

        // Configure the loader
        loader.loadMeshCb = (path, manager, done) => {
            console.log('Loading mesh from path:', path);
            // Extract the robot type from the URDF path
            let robotType = urdfPath.split('/')[4]; // Updated index to account for /AutoURDF/ in path
            
            // Handle the path differently based on the mesh path structure
            let relativePath;
            if (path.includes('mesh/')) {
                relativePath = path.split('mesh/')[1];
            } else {
                relativePath = path.split('/').pop();
            }
            
            const absolutePath = window.location.origin + '/AutoURDF/static/urdf/' + robotType + '/urdf/mesh/' + relativePath;
            console.log('Constructed absolute path:', absolutePath);
            
            const stlLoader = new STLLoader(manager);
            stlLoader.load(
                absolutePath,
                (geometry) => {
                    console.log('STL geometry loaded successfully');
                    const material = new MeshPhongMaterial({
                        color: 0x888888,
                        shininess: 30
                    });
                    const mesh = new Mesh(geometry, material);
                    mesh.castShadow = true;
                    mesh.receiveShadow = true;
                    done(mesh);
                },
                undefined,
                (error) => {
                    console.error('Error loading STL:', absolutePath, error);
                }
            );
        };

        // Extract the robot type for the packages path
        const robotType = urdfPath.split('/')[4]; // Updated index to account for /AutoURDF/ in path
        loader.packages = {
            "": window.location.origin + "/AutoURDF/static/urdf/" + robotType + "/urdf/"
        };

        // Load the URDF
        loader.load(urdfPath, result => {
            this.robot = result;
            
            // Process joints
            this.processJoints();
        });

        // Wait until all geometry has loaded to add the model to the scene
        manager.onLoad = () => {
            console.log('All geometries loaded, processing robot...');
            
            // Apply rotations to the robot
            this.robot.rotation.x = Math.PI / 2;
            // Flip the z-axis by applying a 180-degree rotation around the y-axis
            this.robot.rotation.y = Math.PI;
            
            this.robot.traverse(c => {
                c.castShadow = true;
            });

            // Update matrices
            this.robot.updateMatrixWorld(true);

            // Center the robot
            const bb = new Box3();
            bb.setFromObject(this.robot);
            this.robot.position.y -= bb.min.y;

            // Add to scene
            this.scene.add(this.robot);
            console.log('Robot added to scene successfully');

            // Update camera and controls
            const center = bb.getCenter(new Vector3());
            const size = bb.getSize(new Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            this.camera.position.set(maxDim, maxDim, maxDim);
            this.controls.target.copy(center);
            this.controls.update();
            
            // Create joint controllers
            this.createJointControllers();
        };
    }

    processJoints() {
        if (!this.robot) return;
        
        // Process the robot joints
        this.robot.traverse(child => {
            // Check if this is a joint object with a jointType property
            if (child.isURDFJoint && child.jointType !== 'fixed') {
                const jointName = child.name;
                console.log(`Processing joint: ${jointName}, type: ${child.jointType}`);
                
                // Extract axis data - URDFLoader may store it in different formats
                let axis = [0, 0, 1]; // Default Z-axis
                if (child.axis) {
                    if (Array.isArray(child.axis)) {
                        axis = child.axis;
                    } else if (typeof child.axis === 'object' && child.axis.x !== undefined) {
                        axis = [child.axis.x, child.axis.y, child.axis.z];
                    }
                }
                
                // Get limits from the joint object
                let lowerLimit = -Math.PI;
                let upperLimit = Math.PI;
                
                if (child.limit) {
                    lowerLimit = child.limit.lower !== undefined ? child.limit.lower : -Math.PI;
                    upperLimit = child.limit.upper !== undefined ? child.limit.upper : Math.PI;
                }
                
                // Add to joints dictionary
                this.joints[jointName] = {
                    object: child,
                    type: child.jointType,
                    axis: axis,
                    value: 0,
                    lowerLimit: lowerLimit,
                    upperLimit: upperLimit
                };
                
                // Set initial value
                this.jointValues[jointName] = 0;
                
                // Create slider for this joint
                this.createJointSlider(jointName);
            }
        });
        
        console.log('Processed joints:', Object.keys(this.joints));
    }

    createJointSlider(jointName) {
        const joint = this.joints[jointName];
        if (!joint) return;
        
        const container = document.createElement('div');
        container.style.marginBottom = '10px';
        
        const label = document.createElement('label');
        label.textContent = jointName;
        label.style.display = 'block';
        label.style.marginBottom = '2px';
        
        const controlRow = document.createElement('div');
        controlRow.style.display = 'flex';
        controlRow.style.alignItems = 'center';
        
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = joint.lowerLimit !== undefined ? joint.lowerLimit : (joint.type === 'revolute' ? -Math.PI : -2);
        slider.max = joint.upperLimit !== undefined ? joint.upperLimit : (joint.type === 'revolute' ? Math.PI : 2);
        slider.step = joint.type === 'revolute' ? '0.01' : '0.001';
        slider.value = 0;
        slider.style.flex = '1';
        slider.dataset.joint = jointName;
        
        const valueDisplay = document.createElement('span');
        valueDisplay.textContent = '0';
        valueDisplay.style.marginLeft = '10px';
        valueDisplay.style.minWidth = '50px';
        valueDisplay.style.textAlign = 'right';
        
        slider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.setJointValue(jointName, value);
            valueDisplay.textContent = value.toFixed(2);
        });
        
        controlRow.appendChild(slider);
        controlRow.appendChild(valueDisplay);
        
        container.appendChild(label);
        container.appendChild(controlRow);
        
        this.sliderContainer.appendChild(container);
    }

    createJointControllers() {
        // Remove existing controllers
        while (this.jointControllers.children.length) {
            this.jointControllers.remove(this.jointControllers.children[0]);
        }
        
        // Do not create new controllers since they are disabled
        return;
    }

    getJointValue(jointName) {
        return this.jointValues[jointName] || 0;
    }

    setJointValue(jointName, value) {
        const joint = this.joints[jointName];
        if (!joint) return;
        
        // Always check joint limits
        if (joint.lowerLimit !== undefined && joint.upperLimit !== undefined) {
            value = Math.max(joint.lowerLimit, Math.min(joint.upperLimit, value));
        }
        
        // Store the value
        this.jointValues[jointName] = value;
        joint.value = value;
        
        // Apply to the URDF robot
        if (joint.object) {
            if (joint.type === 'revolute' || joint.type === 'continuous') {
                // Use the robot's built-in joint value setter instead of direct property assignment
                joint.object.setJointValue(value);
            } else if (joint.type === 'prismatic') {
                // Use the robot's built-in joint value setter instead of direct property assignment
                joint.object.setJointValue(value);
            }
        }
        
        // Update the corresponding slider if exists
        const slider = this.sliderContainer.querySelector(`input[data-joint="${jointName}"]`);
        if (slider && parseFloat(slider.value) !== value) {
            slider.value = value;
            const valueDisplay = slider.nextElementSibling;
            if (valueDisplay) {
                valueDisplay.textContent = value.toFixed(2);
            }
        }
    }

    updateSliders() {
        Object.entries(this.jointValues).forEach(([jointName, value]) => {
            const slider = this.sliderContainer.querySelector(`input[data-joint="${jointName}"]`);
            if (slider) {
                slider.value = value;
                const valueDisplay = slider.nextElementSibling;
                if (valueDisplay) {
                    valueDisplay.textContent = value.toFixed(2);
                }
            }
        });
    }

    onMouseDown(event) {
        // Since controllers are disabled, we don't need to handle controller interaction
        return;
    }

    onMouseMove(event) {
        // Since controllers are disabled, we don't need to handle controller interaction
        return;
    }

    onMouseUp(event) {
        // Since controllers are disabled, we don't need to handle controller interaction
        return;
    }

    dispatchEvent(type, detail) {
        const event = new CustomEvent(type, { detail: detail });
        this.container.dispatchEvent(event);
    }

    onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(this.animate);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    // Add the onHover method to handle tooltip display
    onHover(event) {
        // Since controllers are disabled, we don't need to handle controller hovering
        return;
    }

    get angles() {
        const angles = {};
        Object.entries(this.joints).forEach(([name, joint]) => {
            angles[name] = this.getJointValue(name);
        });
        return angles;
    }
}

let viewer = null;  // Keep track of viewer instance

function initializeButtons() {
    const wx200Button = document.getElementById('load-wx200');
    // const wx200rButton = document.getElementById('load-wx200r');
    const ur5Button = document.getElementById('load-ur5');
    const pandaButton = document.getElementById('load-panda');
    const pxButton = document.getElementById('load-pxs');
    // const soloButton = document.getElementById('load-solo');

    if (!wx200Button || !ur5Button || !pandaButton) {
        console.warn('Buttons not found, will retry in 100ms');
        setTimeout(initializeButtons, 100);
        return;
    }

    wx200Button.addEventListener('click', () => {
        console.log('Loading WX200 URDF...');
        viewer.loadURDF('/AutoURDF/static/urdf/wx200_5dof/urdf/robot.urdf');
    });

    // wx200rButton.addEventListener('click', () => {
    //     console.log('Loading WX200 Real-world URDF...');
    //     viewer.loadURDF('/AutoURDF/static/urdf/wx200_real_5dof/urdf/robot.urdf');
    // });

    ur5Button.addEventListener('click', () => {
        console.log('Loading UR5e URDF...');
        viewer.loadURDF('/AutoURDF/static/urdf/UR5e_5dof/urdf/robot.urdf');
    });

    pandaButton.addEventListener('click', () => {
        console.log('Loading Panda URDF...');
        viewer.loadURDF('/AutoURDF/static/urdf/Pandas_6dof/urdf/robot.urdf');
    });

    pxButton.addEventListener('click', () => {
        console.log('Loading PhantomX URDF...');
        viewer.loadURDF('/AutoURDF/static/urdf/pxs_18dof/urdf/robot.urdf');
    });

    // soloButton.addEventListener('click', () => {
    //     console.log('Loading Solo URDF...');
    //     viewer.loadURDF('/AutoURDF/static/urdf/solo_8dof/urdf/robot.urdf');
    // });
    
}

function initializeViewer() {
    if (viewer) {
        console.log('Viewer already initialized, skipping...');
        return;
    }

    try {
        console.log('Initializing URDF viewer...');
        const container = document.getElementById('urdf-container');
        if (!container) {
            console.warn('Container not found, will retry in 100ms');
            setTimeout(initializeViewer, 100);
            return;
        }
        
        viewer = new URDFViewer('urdf-container');
        console.log('URDF viewer initialized successfully');

        // Initialize buttons after viewer is ready
        initializeButtons();
        
        // Auto-load WX200 robot
        console.log('Auto-loading WX200 robot...');
        setTimeout(function() {
            viewer.loadURDF('/AutoURDF/static/urdf/wx200_5dof/urdf/robot.urdf');
        }, 300);
    } catch (error) {
        console.error('Error initializing URDF viewer:', error);
    }
}

// Try to initialize as soon as possible if DOM is already loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeViewer);
} else {
    initializeViewer();
} 