import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import URDFLoader from "https://cdn.jsdelivr.net/npm/urdf-loader@0.12.6/src/URDFLoader.js";
import { GUI } from "https://cdn.jsdelivr.net/npm/lil-gui@0.19/dist/lil-gui.esm.js";

const config = window.SO101_VISUALIZER || {};
const containerId = config.containerId || "so101-viz-container";
const container = document.getElementById(containerId);
if (!container) throw new Error("SO101 container not found: " + containerId);

const WIDTH = container.clientWidth || 800;
const HEIGHT = 400;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);

const camera = new THREE.PerspectiveCamera(50, WIDTH / HEIGHT, 0.001, 10);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(WIDTH, HEIGHT);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const target = new THREE.Vector3(0, 0, 0);
const INIT_ELEVATION = Math.PI / 3;
const INIT_RADIUS = 1.0;
let azimuth = 0.5;
let elevation = INIT_ELEVATION;
let radius = INIT_RADIUS;
let drag = false;
let prevX = 0;
let prevY = 0;

function updateCamera() {
  camera.position.set(
    target.x + radius * Math.sin(elevation) * Math.cos(azimuth),
    target.y + radius * Math.cos(elevation),
    target.z + radius * Math.sin(elevation) * Math.sin(azimuth)
  );
  camera.lookAt(target);
}

renderer.domElement.addEventListener("pointerdown", (e) => {
  drag = true;
  prevX = e.clientX;
  prevY = e.clientY;
});
renderer.domElement.addEventListener("pointerup", () => { drag = false; });
renderer.domElement.addEventListener("pointerleave", () => { drag = false; });
renderer.domElement.addEventListener("pointermove", (e) => {
  if (!drag) return;
  azimuth -= (e.clientX - prevX) * 0.005;
  elevation = Math.max(0.1, Math.min(Math.PI - 0.1, elevation + (e.clientY - prevY) * 0.005));
  prevX = e.clientX;
  prevY = e.clientY;
});

const ambient = new THREE.AmbientLight(0x404040, 10);
scene.add(ambient);
const light = new THREE.DirectionalLight(0xffffff, 5);
light.position.set(0.3, 0.3, 0.3);
scene.add(light);

const manager = new THREE.LoadingManager();
const loader = new URDFLoader(manager);

const urdfText = config.urdfText;
const meshDataUrls = config.meshDataUrls || {};

if (urdfText && Object.keys(meshDataUrls).length > 0) {
  loader.loadMeshCb = function (path, mgr, done) {
    const url = meshDataUrls[path];
    if (!url) {
      done(null, new Error("No data URL for " + path));
      return;
    }
    const stlLoader = new STLLoader(mgr);
    stlLoader.load(
      url,
      (geom) => done(new THREE.Mesh(geom, new THREE.MeshPhongMaterial())),
      undefined,
      (err) => done(null, err)
    );
  };
  loader.workingPath = "";
  const robot = loader.parse(urdfText, "");
  addRobotToScene(robot);
} else {
  loader.workingPath = config.workingPath || "../assets/so101/";
  const urdfUrl = config.urdfUrl || "../assets/so101/robot.urdf";
  loader.load(
    urdfUrl,
    (robot) => addRobotToScene(robot),
    undefined,
    (err) => console.error("URDF load error", err)
  );
}

function addRobotToScene(robot) {
  const wrap = new THREE.Group();
  wrap.rotation.x = -Math.PI / 2;
  wrap.add(robot);
  scene.add(wrap);

  let solvableRegionGroup = null;
  const guiParams = { solvableRegion: true };
  if (config.solvableRegionObjUrl) {
    const objLoader = new OBJLoader();
    objLoader.load(
      config.solvableRegionObjUrl,
      (group) => {
        const mat = new THREE.MeshPhongMaterial({
          color: 0x4488cc,
          transparent: true,
          opacity: 0.35,
          depthWrite: false,
          side: THREE.DoubleSide,
        });
        group.traverse((child) => {
          if (child.isMesh) child.material = mat;
        });
        solvableRegionGroup = group;
        wrap.add(group);
        if (guiParams.solvableRegion !== undefined) group.visible = guiParams.solvableRegion;
      },
      undefined,
      (err) => console.error("Solvable region OBJ load error", err)
    );
  }

  const box = new THREE.Box3().setFromObject(wrap);
  const center = box.getCenter(new THREE.Vector3());
  target.copy(center);
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  radius = INIT_RADIUS;
  elevation = INIT_ELEVATION;
  updateCamera();

  const jointOrder = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
  ];

  container.style.position = "relative";
  const gui = new GUI({ container });
  gui.domElement.style.position = "absolute";
  gui.domElement.style.top = "8px";
  gui.domElement.style.right = "8px";
  const jointParams = {};

  if (config.solvableRegionObjUrl) {
    gui.add(guiParams, "solvableRegion").name("Solvable region").onChange((v) => {
      if (solvableRegionGroup) solvableRegionGroup.visible = v;
    });
  }

  for (const jname of jointOrder) {
    const joint = robot.joints[jname];
    if (!joint || joint.jointType === "fixed") continue;
    const lim = joint.limit;
    const min = lim && typeof lim.lower === "number" ? lim.lower : -Math.PI;
    const max = lim && typeof lim.upper === "number" ? lim.upper : Math.PI;
    const value = typeof joint.angle === "number" ? joint.angle : 0;
    jointParams[jname] = value;
    gui.add(jointParams, jname, min, max).onChange(
      (function (name) {
        return function (v) {
          robot.setJointValue(name, v);
        };
      })(jname)
    );
  }

  const gripperLink =
    (robot.links && robot.links.gripper_frame_link) ||
    robot.getObjectByName("gripper_frame_link") ||
    (() => {
      let found = null;
      robot.traverse((obj) => {
        if (obj.name === "gripper_frame_link") found = obj;
      });
      return found;
    })();
  let statusEl = null;
  let gripperZArrow = null;
  const arrowWorldZ = new THREE.Vector3();
  const worldDown = new THREE.Vector3(0, -1, 0);
  const DOWNTURNED_THRESHOLD = 0.95;

  if (gripperLink) {
    const arrowLength = 0.05;
    const arrowColor = 0x00ffff;
    gripperZArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(0, 0, 0),
      arrowLength,
      arrowColor
    );
    gripperLink.add(gripperZArrow);
    statusEl = document.createElement("div");
    statusEl.style.fontSize = "12px";
    statusEl.style.padding = "4px 0";
    gui.domElement.appendChild(statusEl);
  }

  function animate() {
    requestAnimationFrame(animate);
    updateCamera();
    if (gripperZArrow && statusEl) {
      const elements = gripperZArrow.cone.matrixWorld.elements;
      arrowWorldZ.set(elements[4], elements[5], elements[6]).normalize();
      const dot = arrowWorldZ.dot(worldDown);
      if (dot >= DOWNTURNED_THRESHOLD) {
        statusEl.textContent = "downturned";
        statusEl.style.color = "#2ecc71";
      } else {
        statusEl.textContent = "not downturned";
        statusEl.style.color = "#e74c3c";
      }
    }
    renderer.render(scene, camera);
  }
  animate();
}
