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
    MathUtils
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import URDFLoader from 'URDFLoader';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';

class URDFViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = new Scene();
        this.scene.background = new Color(0x263238);

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

        const ambientLight = new AmbientLight(0xffffff, 0.2);
        this.scene.add(ambientLight);

        // Setup ground
        const ground = new Mesh(new PlaneGeometry(), new ShadowMaterial({ opacity: 0.25 }));
        ground.rotation.x = -Math.PI / 2;
        ground.scale.setScalar(30);
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Add grid helper
        const grid = new GridHelper(10, 10);
        this.scene.add(grid);

        // Setup controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.minDistance = 4;
        this.controls.target.y = 1;
        this.controls.update();

        // Bind methods
        this.animate = this.animate.bind(this);
        this.onResize = this.onResize.bind(this);
        
        // Add event listeners
        window.addEventListener('resize', this.onResize);
        
        // Start animation loop
        this.animate();
    }

    loadURDF(urdfPath) {
        console.log('Starting to load URDF from path:', urdfPath);
        if (this.robot) {
            console.log('Removing existing robot from scene');
            this.scene.remove(this.robot);
        }

        const manager = new LoadingManager();
        const loader = new URDFLoader(manager);

        // Configure the loader
        loader.loadMeshCb = (path, manager, done) => {
            console.log('Loading mesh from path:', path);
            const relativePath = path.split('mesh/')[1];
            const absolutePath = window.location.origin + '/static/urdf/wx200_5dof/urdf/mesh/' + relativePath;
            console.log('Constructed absolute path:', absolutePath);
            
            const stlLoader = new STLLoader(manager);
            stlLoader.load(
                absolutePath,
                (geometry) => {
                    console.log('STL geometry loaded successfully:', absolutePath);
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

        loader.packages = {
            "": window.location.origin + "/static/urdf/wx200_5dof/urdf/"
        };

        // Load the URDF
        loader.load(urdfPath, result => {
            this.robot = result;
        });

        // Wait until all geometry has loaded to add the model to the scene
        manager.onLoad = () => {
            console.log('All geometries loaded, processing robot...');
            
            this.robot.rotation.x = Math.PI / 2;
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
            this.camera.position.set(maxDim * 2, maxDim * 2, maxDim * 2);
            this.controls.target.copy(center);
            this.controls.update();
        };
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
}

let viewer = null;  // Keep track of viewer instance

function initializeButtons() {
    const wx200Button = document.getElementById('load-wx200');
    const ur5Button = document.getElementById('load-ur5');
    const pandaButton = document.getElementById('load-panda');

    if (!wx200Button || !ur5Button || !pandaButton) {
        console.warn('Buttons not found, will retry in 100ms');
        setTimeout(initializeButtons, 100);
        return;
    }

    wx200Button.addEventListener('click', () => {
        console.log('Loading WX200 URDF...');
        viewer.loadURDF('/static/urdf/wx200_5dof/urdf/robot.urdf');
    });

    ur5Button.addEventListener('click', () => {
        console.log('Loading UR5e URDF...');
        viewer.loadURDF('/static/urdf/ur5e_5dof/urdf/ur5e_5_20_seg/4_deg_20_cams.urdf');
    });

    pandaButton.addEventListener('click', () => {
        console.log('Loading Panda URDF...');
        viewer.loadURDF('/static/urdf/panda_6dof/urdf/panda_6_20_seg/4_deg_20_cams.urdf');
    });
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