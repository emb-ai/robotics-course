import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { GUI } from "https://cdn.jsdelivr.net/npm/lil-gui@0.19/dist/lil-gui.esm.js";

const config = window.BEADS_VISUALIZER || {};
const containerId = config.containerId || "beads-viz-container";
const endpoints = config.endpoints || [[0, 0, 0], [0, 0, 1], [0.08, 0, 2], [0.15, 0.05, 3], [0.2, 0.1, 4]];
const beadRadius = config.bead_radius != null ? config.bead_radius : 1.0;

const container = document.getElementById(containerId);
if (!container) throw new Error("Beads container not found: " + containerId);

const WIDTH = container.clientWidth || 800;
const HEIGHT = 400;

const MAT = new THREE.MeshLambertMaterial({ color: 0xb4c8dc, side: THREE.DoubleSide });
const ANGLE = Math.PI / 3;
const CYLINDER_RADIUS_FACTOR = 0.2

function ballSocketSegment(p0, p1, radius) {
  const dx = p1[0] - p0[0], dy = p1[1] - p0[1], dz = p1[2] - p0[2];
  const length = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-10;
  const ax = dx / length, ay = dy / length, az = dz / length;
  const axis = new THREE.Vector3(ax, ay, az);
  const group = new THREE.Group();
  const up = new THREE.Vector3(0, -1, 0);

  const thetaStart = 0;
  const thetaLength = Math.PI - ANGLE - radius * CYLINDER_RADIUS_FACTOR;
  const outerHemisphere = new THREE.SphereGeometry(radius, 32, 16, 0, 2 * Math.PI, thetaStart, thetaLength);
  const innerHemisphere = new THREE.SphereGeometry(radius * 0.9, 32, 16, 0, 2 * Math.PI, thetaStart, thetaLength);
  const socketOuter = new THREE.Mesh(outerHemisphere, MAT);
  const socketInner = new THREE.Mesh(innerHemisphere, MAT);
  socketOuter.position.set(p1[0], p1[1], p1[2]);
  socketInner.position.set(p1[0], p1[1], p1[2]);
  socketOuter.quaternion.setFromUnitVectors(up, axis);
  socketInner.quaternion.setFromUnitVectors(up, axis);
  group.add(socketOuter);
  group.add(socketInner);

  const rimDist = radius * Math.cos(ANGLE + radius * CYLINDER_RADIUS_FACTOR);
  const outerR = radius * Math.sin(ANGLE + radius * CYLINDER_RADIUS_FACTOR);
  const innerR = outerR * 0.9;
  const shape = new THREE.Shape();
  shape.absarc(0, 0, outerR, 0, 2 * Math.PI, false);
  const hole = new THREE.Path();
  hole.absarc(0, 0, innerR, 0, 2 * Math.PI, true);
  shape.holes.push(hole);
  const annulusGeom = new THREE.ShapeGeometry(shape);
  const annulusMesh = new THREE.Mesh(annulusGeom, MAT);
  annulusMesh.position.set(p1[0] + rimDist * ax, p1[1] + rimDist * ay, p1[2] + rimDist * az);
  annulusMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis);
  group.add(annulusMesh);

  const cylinderHeight = Math.max(length - radius * 0.9, 1e-6);
  const cylGeom = new THREE.CylinderGeometry(radius * CYLINDER_RADIUS_FACTOR, radius * CYLINDER_RADIUS_FACTOR, cylinderHeight, 16);
  const cylMesh = new THREE.Mesh(cylGeom, new THREE.MeshLambertMaterial({ color: 0xb4c8dc }));
  const mid = [(p0[0] + p1[0] - radius / length * dx) / 2, (p0[1] + p1[1] - radius / length * dy) / 2, (p0[2] + p1[2] - radius / length * dz) / 2];
  cylMesh.position.set(mid[0], mid[1], mid[2]);
  cylMesh.quaternion.setFromUnitVectors(up, axis);
  group.add(cylMesh);

  const ballGeom = new THREE.SphereGeometry(radius * 0.8, 24, 16);
  const ballMesh = new THREE.Mesh(ballGeom, new THREE.MeshLambertMaterial({ color: 0xb4c8dc }));
  ballMesh.position.set(p0[0], p0[1], p0[2]);
  group.add(ballMesh);

  return group;
}

function boundingSphereWireframe(pts) {
  const n = pts.length;
  const cx = pts.reduce((s, p) => s + p[0], 0) / n;
  const cy = pts.reduce((s, p) => s + p[1], 0) / n;
  const cz = pts.reduce((s, p) => s + p[2], 0) / n;
  let r = 0;
  for (const p of pts) {
    const d = Math.hypot(p[0] - cx, p[1] - cy, p[2] - cz);
    if (d > r) r = d;
  }
  r = Math.max(r, 1e-6);
  const sphere = new THREE.SphereGeometry(r + 1.0, 24, 16);
  const edges = new THREE.EdgesGeometry(sphere);
  const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x808080 }));
  line.position.set(cx, cy, cz);
  return line;
}

function collisionSpheresGroup(pts, radius) {
  const group = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({ color: 0x4080c0 });
  const geom = new THREE.SphereGeometry(radius, 24, 16);
  for (const p of pts) {
    const mesh = new THREE.Mesh(geom, mat.clone());
    mesh.position.set(p[0], p[1], p[2]);
    group.add(mesh);
  }
  return group;
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);
scene.add(new THREE.AmbientLight(0x404040));
const light = new THREE.DirectionalLight(0xffffff, 0.8);
light.position.set(2, 3, 4);
scene.add(light);

for (let i = 0; i < endpoints.length - 1; i++) {
  scene.add(ballSocketSegment(endpoints[i], endpoints[i + 1], beadRadius));
}

const boundingSphereMesh = boundingSphereWireframe(endpoints);
const collisionGroup = collisionSpheresGroup(endpoints, beadRadius);
scene.add(boundingSphereMesh);
scene.add(collisionGroup);

const guiParams = { boundingSphere: true, collisionShape: false };
container.style.position = "relative";
const gui = new GUI({ container });
gui.domElement.style.position = "absolute";
gui.domElement.style.top = "8px";
gui.domElement.style.right = "8px";
gui.add(guiParams, "boundingSphere").onChange((v) => {
  boundingSphereMesh.visible = v;
});
gui.add(guiParams, "collisionShape").onChange((v) => {
  collisionGroup.visible = v;
});
boundingSphereMesh.visible = guiParams.boundingSphere;
collisionGroup.visible = guiParams.collisionShape;

const xs = endpoints.map((p) => p[0]);
const ys = endpoints.map((p) => p[1]);
const zs = endpoints.map((p) => p[2]);
const min = new THREE.Vector3(Math.min(...xs), Math.min(...ys), Math.min(...zs));
const max = new THREE.Vector3(Math.max(...xs), Math.max(...ys), Math.max(...zs));
const center = new THREE.Vector3().addVectors(min, max).multiplyScalar(0.5);
const size = new THREE.Vector3().subVectors(max, min);
const maxDim = Math.max(size.x, size.y, size.z) || 1;
const margin = 2;
const distance = Math.max(maxDim * 1.5, margin);

const camera = new THREE.PerspectiveCamera(50, WIDTH / HEIGHT, 0.01, 1000);
camera.position.set(center.x + distance * 0.6, center.y + distance * 0.6, center.z + distance * 0.6);
camera.lookAt(center);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(WIDTH, HEIGHT);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const target = center.clone();
const delta = camera.position.clone().sub(target);
let radius = delta.length();
let elevation = Math.acos(Math.max(-1, Math.min(1, delta.y / radius)));
let azimuth = Math.atan2(delta.z, delta.x);
let drag = false;
let prevX = 0, prevY = 0;

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

function animate() {
  requestAnimationFrame(animate);
  updateCamera();
  renderer.render(scene, camera);
}
animate();
