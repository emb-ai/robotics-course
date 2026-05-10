/**
 * Unicycle robot viewer and control using the official MuJoCo WASM bindings + Three.js.
 * Right-hand pad sets open-loop pelvis motor torques (scaled to each actuator ctrlrange).
 * <kbd>W</kbd>/<kbd>S</kbd> or <kbd>↑</kbd>/<kbd>↓</kbd> add wheel torque. No PD tracking and
 * no LQR balance assist.
 * MuJoCo: official single-threaded `@mujoco/mujoco` package loaded from the viewer import map.
 * Plain static serving is enough, for example `python3 -m http.server`.
 */
import * as THREE from "three";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.184.0/examples/jsm/controls/OrbitControls.js";
import loadMujoco from "@mujoco/mujoco";

const ASSETS_URL = new URL("../../../assets/", import.meta.url).href;
const SIMPLIFIED_MJCF_URL = new URL("unicycle_simplified.xml", ASSETS_URL).href;
const FULL_MJCF_URL = new URL("unicycle_with_humanoid.xml", ASSETS_URL).href;
// Meshes are referenced only by the full humanoid MJCF; the simplified model is primitives only.
const MESH_FILES = ["unicycle/seat.obj", "unicycle/shaft.obj", "unicycle/wheel_crank_mesh.obj"];

// Simplified-model body names whose rendered geoms should hide when the humanoid ragdoll is shown.
const HIDE_ON_RAGDOLL = new Set(["com", "seat", "wheel"]);
// Joints whose simplified qpos/qvel are pinned onto the visual humanoid each frame.
// After re-rooting both models at the wheel, these three joints exist with identical names,
// axes, and angle conventions in both models, so the pin is a direct qpos/qvel copy.
const PINNED_JOINTS = ["pelvis_y", "pelvis_x", "wheel"];
// Joints on the visual model that must stay at 0 so that the full humanoid's upper body
// (torso, waist_lower, pelvis) remains rigidly stacked above the pelvis hinge, exactly as
// in the simplified model — the simplified rider has no spine DOFs between the pelvis
// hinges and the wheel, so letting the abdomen triplet ragdoll would offset the visual
// upper body from the simplified one.
const ZERO_PINNED_JOINTS = ["abdomen_z", "abdomen_y", "abdomen_x"];
const UI_CONTROL_SELECTOR = "input,button,select,textarea,label,a[href],[contenteditable='true']";
const UNICYCLE_KEY_CODES = new Set([
  "KeyW",
  "KeyS",
  "ArrowUp",
  "ArrowDown",
  "Space",
  "KeyR",
  "KeyH",
  "Escape",
]);

/** MuJoCo 3.x WASM bindings may use BigInt for sizes and ids; avoid mixing with Number in `for` bounds and `*`. */
function mjNum(x) {
  return typeof x === "bigint" ? Number(x) : x;
}

// Use mujoco.mjtGeom.*.value for geom type; do not hardcode (enum order: PLANE, HFIELD, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, ...).

// MuJoCo is Z-up; Three.js is Y-up. Map (mj_x, mj_y, mj_z) -> (mj_x, mj_z, -mj_y).
function getMujocoPos(data, bodyId, out) {
  const mx = mjNum(data.xpos[bodyId * 3]);
  const my = mjNum(data.xpos[bodyId * 3 + 1]);
  const mz = mjNum(data.xpos[bodyId * 3 + 2]);
  out.set(mx, mz, -my);
  return out;
}

// Quat: MuJoCo (w,x,y,z) -> Three.js (x,y,z,w) with Z-up to Y-up swizzle per zalo/mujoco_wasm.
function getMujocoQuat(data, bodyId, out) {
  const w = mjNum(data.xquat[bodyId * 4]);
  const x = mjNum(data.xquat[bodyId * 4 + 1]);
  const y = mjNum(data.xquat[bodyId * 4 + 2]);
  const z = mjNum(data.xquat[bodyId * 4 + 3]);
  out.set(-x, -z, y, -w);
  return out;
}

// Swizzle a mesh-local Z-up vertex/normal (mx,my,mz) into Three.js Y-up (mx, mz, -my).
// This matches the body/geom pose swizzle used in getMujocoPos/getMujocoQuat, and
// preserves winding order (proper rotation about +x).
function buildMeshGeometry(model, geomId) {
  const meshId = mjNum(model.geom_dataid[geomId]);
  if (meshId < 0) return new THREE.SphereGeometry(0.05, 8, 8);
  const vertadr = mjNum(model.mesh_vertadr[meshId]);
  const vertnum = mjNum(model.mesh_vertnum[meshId]);
  const faceadr = mjNum(model.mesh_faceadr[meshId]);
  const facenum = mjNum(model.mesh_facenum[meshId]);

  const positions = new Float32Array(vertnum * 3);
  for (let i = 0; i < vertnum; i++) {
    const src = (vertadr + i) * 3;
    const mx = model.mesh_vert[src];
    const my = model.mesh_vert[src + 1];
    const mz = model.mesh_vert[src + 2];
    positions[i * 3] = mx;
    positions[i * 3 + 1] = mz;
    positions[i * 3 + 2] = -my;
  }

  const indices = new Uint32Array(facenum * 3);
  for (let i = 0; i < facenum; i++) {
    indices[i * 3] = model.mesh_face[(faceadr + i) * 3];
    indices[i * 3 + 1] = model.mesh_face[(faceadr + i) * 3 + 1];
    indices[i * 3 + 2] = model.mesh_face[(faceadr + i) * 3 + 2];
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setIndex(new THREE.BufferAttribute(indices, 1));
  if (model.mesh_normal && model.mesh_normal.length >= (vertadr + vertnum) * 3) {
    const normals = new Float32Array(vertnum * 3);
    for (let i = 0; i < vertnum; i++) {
      const src = (vertadr + i) * 3;
      const nx = model.mesh_normal[src];
      const ny = model.mesh_normal[src + 1];
      const nz = model.mesh_normal[src + 2];
      normals[i * 3] = nx;
      normals[i * 3 + 1] = nz;
      normals[i * 3 + 2] = -ny;
    }
    geom.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  } else {
    geom.computeVertexNormals();
  }
  return geom;
}

// Geometry creation aligned with zalo/mujoco_wasm: plane 100×100, cylinder Y-up (no extra rotation).
function createGeometry(mujoco, model, geomId) {
  const type = mjNum(model.geom_type[geomId]);
  const size = [
    mjNum(model.geom_size[geomId * 3]),
    mjNum(model.geom_size[geomId * 3 + 1]),
    mjNum(model.geom_size[geomId * 3 + 2]),
  ];
  const G = mujoco.mjtGeom || {};
  const plane = G.mjGEOM_PLANE?.value ?? 0;
  const sphere = G.mjGEOM_SPHERE?.value ?? 2;
  const capsule = G.mjGEOM_CAPSULE?.value ?? 3;
  const ellipsoid = G.mjGEOM_ELLIPSOID?.value ?? 4;
  const cylinder = G.mjGEOM_CYLINDER?.value ?? 5;
  const box = G.mjGEOM_BOX?.value ?? 6;
  const mesh = G.mjGEOM_MESH?.value ?? 7;
  let geometry;
  if (type === plane) {
    geometry = new THREE.PlaneGeometry(100, 100);
    geometry.rotateX(-Math.PI / 2);
  } else if (type === sphere) {
    geometry = new THREE.SphereGeometry(size[0], 24, 24);
  } else if (type === cylinder) {
    geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 24);
  } else if (type === capsule) {
    // MuJoCo capsule: size=[radius, half_length_of_cylindrical_section], axis = local +Z.
    // Three.js CapsuleGeometry(radius, length, capSegments, radialSegments) builds along +Y
    // and `length` is the length of the cylindrical middle section. After our Z-up→Y-up
    // quat swizzle (getMujocoQuat), a MuJoCo local +Z axis maps to Three's local +Y, so no
    // extra rotation is needed.
    const r = size[0];
    const len = Math.max(1e-6, size[1] * 2);
    geometry = THREE.CapsuleGeometry
      ? new THREE.CapsuleGeometry(r, len, 8, 24)
      : new THREE.CylinderGeometry(r, r, len, 24);
  } else if (type === box) {
    geometry = new THREE.BoxGeometry(size[0] * 2, size[2] * 2, size[1] * 2);
  } else if (type === ellipsoid) {
    geometry = new THREE.SphereGeometry(1.0, 24, 24);
    geometry.scale(size[0], size[2], size[1]);
  } else if (type === mesh) {
    geometry = buildMeshGeometry(model, geomId);
  } else {
    geometry = new THREE.SphereGeometry(0.1, 8, 8);
  }
  return geometry;
}

// Geom pose: same position/quat swizzle as zalo/mujoco_wasm (getPosition / getQuaternion).
function createMeshForGeom(mujoco, model, geomId, material, isPlane) {
  const geometry = createGeometry(mujoco, model, geomId);
  const mesh = new THREE.Mesh(geometry, material);
  const gpos = model.geom_pos;
  if (gpos && geomId * 3 + 2 < gpos.length) {
    const mx = mjNum(gpos[geomId * 3]);
    const my = mjNum(gpos[geomId * 3 + 1]);
    const mz = mjNum(gpos[geomId * 3 + 2]);
    mesh.position.set(mx, mz, -my);
  }
  const gquat = model.geom_quat;
  if (gquat && geomId * 4 + 3 < gquat.length && !isPlane) {
    const w = mjNum(gquat[geomId * 4]);
    const x = mjNum(gquat[geomId * 4 + 1]);
    const y = mjNum(gquat[geomId * 4 + 2]);
    const z = mjNum(gquat[geomId * 4 + 3]);
    mesh.quaternion.set(-x, -z, y, -w);
  }
  mesh.castShadow = !isPlane;
  mesh.receiveShadow = true;
  return mesh;
}

// ---------------- MuJoCo material/texture extraction -----------------------------------------
// Follows zalo/mujoco_wasm's mujocoUtils.js: per-geom appearance comes from
// `geom_matid` → `mat_{rgba,texid,texrepeat,specular,reflectance,shininess}`, and per-texture
// pixel data is read directly from `tex_data` (expanded to RGBA for Three.DataTexture).
//
// MuJoCo texture types (model.tex_type):   0 = 2D, 1 = cube, 2 = skybox.
// Texture roles (model.mat_texid is nmat × mjNTEXROLE): we use role 1 = mjTEXROLE_RGB.

const MJ_TEX_2D = 0;
const MJ_TEX_CUBE = 1;
const MJ_TEX_SKYBOX = 2;
const MJ_NTEXROLE = 10;
const MJ_TEXROLE_RGB = 1;

// Build a THREE.DataTexture for every texture in the model by slicing model.tex_data.
// For cube/skybox types the data is a vertical strip of 6 width×width faces stacked in MuJoCo's
// face order (right, left, up, down, front, back). We return the full strip as a single
// DataTexture (suitable for .map on primitives with standard UVs, matching zalo's approach);
// a separate helper splits the strip into a proper CubeTexture for the skybox.
function buildDataTextures(model) {
  const out = [];
  const ntex = mjNum(model.ntex);
  for (let t = 0; t < ntex; t++) {
    const width = mjNum(model.tex_width[t]);
    const height = mjNum(model.tex_height[t]);
    const channels = mjNum(model.tex_nchannel[t]);
    const adr = mjNum(model.tex_adr[t]);
    const rgba = new Uint8Array(width * height * 4);
    for (let p = 0; p < width * height; p++) {
      const src = adr + p * channels;
      rgba[p * 4] = model.tex_data[src];
      rgba[p * 4 + 1] = channels > 1 ? model.tex_data[src + 1] : rgba[p * 4];
      rgba[p * 4 + 2] = channels > 2 ? model.tex_data[src + 2] : rgba[p * 4];
      rgba[p * 4 + 3] = channels > 3 ? model.tex_data[src + 3] : 255;
    }
    const tex = new THREE.DataTexture(rgba, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    out.push({ tex, width, height, channels, adr, type: model.tex_type[t] });
  }
  return out;
}

// Split a cube/skybox strip (width × 6*width) into a THREE.CubeTexture.
// MuJoCo face order in the strip: [+X, -X, +Z, -Z, +Y, -Y].
// Three.js CubeTexture order:     [+X, -X, +Y, -Y, +Z, -Z].
// We additionally remap for the MuJoCo-Z-up → Three-Y-up scene convention used everywhere else:
// Three  +Y  ← MuJoCo +Z (up)  : strip index 2
// Three  -Y  ← MuJoCo -Z (down): strip index 3
// Three  +Z  ← MuJoCo -Y (back in Three = -Y in MuJoCo, per (mx, mz, -my)): strip index 5
// Three  -Z  ← MuJoCo +Y       : strip index 4
function buildCubeFromStrip(model, texIndex) {
  const w = mjNum(model.tex_width[texIndex]);
  const h = mjNum(model.tex_height[texIndex]);
  if (h !== 6 * w) return null;
  const channels = mjNum(model.tex_nchannel[texIndex]);
  const adr = mjNum(model.tex_adr[texIndex]);
  const faces = [];
  for (let f = 0; f < 6; f++) {
    const cvs = document.createElement("canvas");
    cvs.width = w;
    cvs.height = w;
    const ctx = cvs.getContext("2d");
    const img = ctx.createImageData(w, w);
    for (let y = 0; y < w; y++) {
      for (let x = 0; x < w; x++) {
        const src = adr + ((f * w + y) * w + x) * channels;
        const dst = (y * w + x) * 4;
        img.data[dst] = model.tex_data[src];
        img.data[dst + 1] = channels > 1 ? model.tex_data[src + 1] : img.data[dst];
        img.data[dst + 2] = channels > 2 ? model.tex_data[src + 2] : img.data[dst];
        img.data[dst + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    faces.push(cvs);
  }
  const order = [0, 1, 2, 3, 5, 4];
  const cube = new THREE.CubeTexture(order.map((i) => faces[i]));
  cube.colorSpace = THREE.SRGBColorSpace;
  cube.needsUpdate = true;
  return cube;
}

async function loadMujocoModule() {
  try {
    const m = await loadMujoco();
    if (m?.MjModel && m?.MjData && typeof m.mj_forward === "function") return m;
    throw new Error("MuJoCo module did not expose MjModel/MjData/mj_forward");
  } catch (e) {
    const msg = e?.message ?? String(e);
    throw new Error(
      `Failed to load MuJoCo from the @mujoco/mujoco CDN package. ${msg}. ` +
        "Ensure the viewer import map points to @mujoco/mujoco@3.7.0 and the page is served over HTTP."
    );
  }
}

async function stageAssets(mujoco) {
  const fetchText = async (url) => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}: HTTP ${res.status}`);
    return res.text();
  };
  const fetchBytes = async (url) => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ${url}: HTTP ${res.status}`);
    return new Uint8Array(await res.arrayBuffer());
  };

  const [rawSimpText, rawFullText, ...meshBuffers] = await Promise.all([
    fetchText(SIMPLIFIED_MJCF_URL),
    fetchText(FULL_MJCF_URL),
    ...MESH_FILES.map((name) => fetchBytes(new URL(name, ASSETS_URL).href)),
  ]);
  const disableCompilerThreads = (xml) =>
    xml.replace("<compiler angle=\"radian\" />", "<compiler angle=\"radian\" usethread=\"false\" />");
  const simpText = disableCompilerThreads(rawSimpText);
  const fullText = disableCompilerThreads(rawFullText);

  const vfs = new mujoco.MjVFS();
  MESH_FILES.forEach((name, i) => vfs.addBuffer(name, meshBuffers[i]));
  return { simpText, fullText, vfs };
}

function loadXmlModel(mujoco, xmlText, vfs) {
  if (typeof mujoco.MjModel.from_xml_string === "function") {
    return vfs ? mujoco.MjModel.from_xml_string(xmlText, vfs) : mujoco.MjModel.from_xml_string(xmlText);
  }
  throw new Error("This MuJoCo build does not expose MjModel.from_xml_string");
}

export async function startUnicycleViewer(config = {}) {
  const mujoco = await loadMujocoModule();
  const staged = await stageAssets(mujoco);

  // Authoritative physics: simplified model.
  const model = loadXmlModel(mujoco, staged.simpText, staged.vfs);
  const data = new mujoco.MjData(model);
  mujoco.mj_forward(model, data);

  // Visual-only: full humanoid. Disable all contacts so the ragdoll doesn't collide with anything.
  const model_v = loadXmlModel(mujoco, staged.fullText, staged.vfs);
  const ngeomV = mjNum(model_v.ngeom);
  for (let g = 0; g < ngeomV; g++) {
    model_v.geom_contype[g] = 0;
    model_v.geom_conaffinity[g] = 0;
  }
  const data_v = new mujoco.MjData(model_v);
  mujoco.mj_forward(model_v, data_v);

  // Extract GPU textures directly from the MuJoCo model's procedurally-generated pixel data.
  // Both MJCFs declare the same asset set, so we only build them from the full model; either
  // source would produce identical data.
  const modelTextures = buildDataTextures(model_v);
  // Find the skybox texture (if any) for the scene background.
  let skyboxCube = null;
  for (let t = 0; t < modelTextures.length; t++) {
    if (mjNum(modelTextures[t].type) === MJ_TEX_SKYBOX) {
      skyboxCube = buildCubeFromStrip(model_v, t);
      break;
    }
  }

  // Resolve enum ids for mj_name2id; fall back to canonical constants if the wasm build
  // doesn't expose mjtObj. Canonical: mjOBJ_BODY=1, mjOBJ_JOINT=3, mjOBJ_ACTUATOR=19.
  const OBJ = mujoco.mjtObj || {};
  const OBJ_BODY = OBJ.mjOBJ_BODY?.value ?? 1;
  const OBJ_JOINT = OBJ.mjOBJ_JOINT?.value ?? 3;
  const OBJ_GEOM = OBJ.mjOBJ_GEOM?.value ?? 5;
  const OBJ_ACTUATOR = OBJ.mjOBJ_ACTUATOR?.value ?? 19;
  const jointAddr = (m, name) => {
    const id = mjNum(mujoco.mj_name2id(m, OBJ_JOINT, name));
    if (id < 0) throw new Error(`joint not found: ${name}`);
    return { id, qposadr: mjNum(m.jnt_qposadr[id]), dofadr: mjNum(m.jnt_dofadr[id]) };
  };
  const actId = (m, name) => {
    const id = mjNum(mujoco.mj_name2id(m, OBJ_ACTUATOR, name));
    if (id < 0) throw new Error(`actuator not found: ${name}`);
    return id;
  };
  const bodyId = (m, name) => {
    const id = mjNum(mujoco.mj_name2id(m, OBJ_BODY, name));
    if (id < 0) throw new Error(`body not found: ${name}`);
    return id;
  };
  const geomId = (m, name) => {
    const id = mjNum(mujoco.mj_name2id(m, OBJ_GEOM, name));
    if (id < 0) throw new Error(`geom not found: ${name}`);
    return id;
  };

  const ACT = {
    pelvis_y: actId(model, "pelvis_y"),
    pelvis_x: actId(model, "pelvis_x"),
    wheel: actId(model, "wheel"),
  };
  const JOINT = {
    pelvis_y: jointAddr(model, "pelvis_y"),
    pelvis_x: jointAddr(model, "pelvis_x"),
    wheel: jointAddr(model, "wheel"),
  };

  // qpos/qvel addresses for each pinned joint in both models.
  const pinMap = PINNED_JOINTS.map((name) => ({
    name,
    s: jointAddr(model, name),
    v: jointAddr(model_v, name),
  }));
  // Visual-model joints clamped to zero every frame (no counterpart in simplified).
  const zeroPinAddrs = ZERO_PINNED_JOINTS.map((name) => jointAddr(model_v, name));

  const PELVIS_Y_LO = mjNum(model.actuator_ctrlrange[ACT.pelvis_y * 2]);
  const PELVIS_Y_HI = mjNum(model.actuator_ctrlrange[ACT.pelvis_y * 2 + 1]);
  const PELVIS_X_LO = mjNum(model.actuator_ctrlrange[ACT.pelvis_x * 2]);
  const PELVIS_X_HI = mjNum(model.actuator_ctrlrange[ACT.pelvis_x * 2 + 1]);
  const WHEEL_LO = mjNum(model.actuator_ctrlrange[ACT.wheel * 2]);
  const WHEEL_HI = mjNum(model.actuator_ctrlrange[ACT.wheel * 2 + 1]);
  const WHEEL_TORQUE_MAX = Math.max(Math.abs(WHEEL_LO), Math.abs(WHEEL_HI));
  const PELVIS_Y_TAU = Math.max(Math.abs(PELVIS_Y_LO), Math.abs(PELVIS_Y_HI));
  const PELVIS_X_TAU = Math.max(Math.abs(PELVIS_X_LO), Math.abs(PELVIS_X_HI));
  const SIM_TIMESTEP = mjNum(model.opt.timestep);

  function clamp(x, lo, hi) {
    return Math.max(lo, Math.min(hi, x));
  }

  mujoco.mj_resetData(model, data);
  mujoco.mj_forward(model, data);
  mujoco.mj_resetData(model_v, data_v);
  mujoco.mj_forward(model_v, data_v);


  // Both the simplified and the full humanoid MJCFs are rooted at the wheel (free joint),
  // so the first 7 qpos values and the first 6 qvel values describe the wheel's world
  // pose/twist with identical semantics in both models. No offset computation is needed.

  const containerId = config.containerId ?? "unicycle-container";
  const container = document.getElementById(containerId) || document.body;
  const width = container.clientWidth || window.innerWidth;
  const height = container.clientHeight || window.innerHeight;

  const scene = new THREE.Scene();
  // Background: MuJoCo-declared skybox if present, otherwise a soft sky gradient color.
  const fallbackBg = new THREE.Color(0.55, 0.70, 0.85);
  if (skyboxCube) {
    scene.background = skyboxCube;
    // Use the skybox as an environment map too — cheap IBL for the physical materials,
    // which makes the rubbery/plastic parts of the unicycle pick up sky tones instead of
    // looking flat. Mirrors what zalo/mujoco_wasm does when a skybox is present.
    scene.environment = skyboxCube;
  } else {
    scene.background = fallbackBg;
  }
  // Distance fog blends props with the sky in the far field, matching the zalo demo.
  scene.fog = new THREE.Fog(skyboxCube ? new THREE.Color(0.55, 0.70, 0.85) : fallbackBg, 12, 40);

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.01, 100);
  camera.position.set(-1.8, 1.8, 1.8);
  scene.add(camera);

  // Three-light rig (hemisphere bounce + key directional sun + filler spotlight).
  // Intensities are tuned for modern three.js (post-r165, no useLegacyLights) where
  // light units are physically scaled; MuJoCo-demo-style visuals need roughly π× the
  // pre-r155 values we had before.
  const hemi = new THREE.HemisphereLight(0xbcd7ff, 0x3a2e1d, 0.85);
  scene.add(hemi);
  const ambient = new THREE.AmbientLight(0xffffff, 0.25);
  scene.add(ambient);
  const dirLight = new THREE.DirectionalLight(0xfff2d8, 2.8);
  dirLight.position.set(4, 6, 4);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.width = 2048;
  dirLight.shadow.mapSize.height = 2048;
  dirLight.shadow.camera.near = 0.1;
  dirLight.shadow.camera.far = 25;
  dirLight.shadow.camera.left = -8;
  dirLight.shadow.camera.right = 8;
  dirLight.shadow.camera.top = 8;
  dirLight.shadow.camera.bottom = -8;
  dirLight.shadow.bias = -0.0004;
  dirLight.shadow.normalBias = 0.02;
  dirLight.shadow.radius = 3;
  scene.add(dirLight);
  const spot = new THREE.SpotLight(0xffffff, 7.0, 14, 1.1, 0.55, 1.0);
  spot.position.set(0, 3.5, 3);
  spot.target.position.set(0, 0.6, 0);
  spot.castShadow = true;
  spot.shadow.mapSize.width = 1024;
  spot.shadow.mapSize.height = 1024;
  spot.shadow.camera.near = 0.2;
  spot.shadow.camera.far = 14;
  spot.shadow.bias = -0.0004;
  spot.shadow.radius = 3;
  scene.add(spot);
  scene.add(spot.target);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  // ACES filmic gives the scene a punchier, more cinematic roll-off without crushing
  // highlights; exposure ~1.1 compensates for the filmic dip on neutral mid-tones.
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  container.appendChild(renderer.domElement);
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute("aria-label", "Unicycle control surface");
  renderer.domElement.style.outline = "none";
  let controlsActive = false;
  let notebookKeyboardPreviouslyEnabled = null;

  function isUiControl(target) {
    return target instanceof Element && Boolean(target.closest(UI_CONTROL_SELECTOR));
  }

  function notebookKeyboardManager() {
    return window.Jupyter?.notebook?.keyboard_manager ?? window.IPython?.notebook?.keyboard_manager ?? null;
  }

  function disableNotebookKeyboard() {
    const manager = notebookKeyboardManager();
    if (!manager || notebookKeyboardPreviouslyEnabled !== null) return;
    notebookKeyboardPreviouslyEnabled = manager.enabled ?? true;
    manager.disable?.();
  }

  function restoreNotebookKeyboard() {
    const manager = notebookKeyboardManager();
    if (!manager || notebookKeyboardPreviouslyEnabled === null) return;
    if (notebookKeyboardPreviouslyEnabled) manager.enable?.();
    notebookKeyboardPreviouslyEnabled = null;
  }

  function activateControls() {
    controlsActive = true;
    renderer.domElement.focus({ preventScroll: true });
    disableNotebookKeyboard();
  }

  function releaseControls() {
    controlsActive = false;
    clearKeys();
    restoreNotebookKeyboard();
  }

  document.addEventListener("pointerdown", (e) => {
    if (!container.contains(e.target) || isUiControl(e.target)) {
      releaseControls();
      return;
    }
    activateControls();
  }, { capture: true });

  // ---------------- Target beacon (green disk + vertical beam + point light) -----------------
  // Goal: give the rider a visible nearby waypoint. The ground disk + additive ring + halo
  // reproduce the classic "landing pad" look; the vertical beam uses a custom shader that
  // fades alpha quadratically with height and pulses, so it reads as a sky-shot volumetric
  // column under additive blending without needing post-processing bloom.
  const TARGET_COLOR = new THREE.Color(0x3dff8c);
  const targetGroup = new THREE.Group();
  targetGroup.name = "unicycle-target";
  scene.add(targetGroup);

  const groundDisk = new THREE.Mesh(
    new THREE.CircleGeometry(0.32, 64),
    new THREE.MeshStandardMaterial({
      color: 0x052915,
      emissive: TARGET_COLOR,
      emissiveIntensity: 1.8,
      roughness: 0.6,
      metalness: 0.0,
      transparent: true,
      opacity: 0.92,
      depthWrite: false,
    }),
  );
  groundDisk.rotation.x = -Math.PI / 2;
  groundDisk.position.y = 0.005;
  targetGroup.add(groundDisk);

  const glowRing = new THREE.Mesh(
    new THREE.RingGeometry(0.33, 0.48, 64),
    new THREE.MeshBasicMaterial({
      color: TARGET_COLOR,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      toneMapped: false,
    }),
  );
  glowRing.rotation.x = -Math.PI / 2;
  glowRing.position.y = 0.007;
  targetGroup.add(glowRing);

  const halo = new THREE.Mesh(
    new THREE.CircleGeometry(0.95, 64),
    new THREE.MeshBasicMaterial({
      color: TARGET_COLOR,
      transparent: true,
      opacity: 0.14,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      toneMapped: false,
    }),
  );
  halo.rotation.x = -Math.PI / 2;
  halo.position.y = 0.004;
  targetGroup.add(halo);

  // Custom shader for the beam: height-fading alpha (uv.y goes 0 at the base → 1 at the top
  // on a standard THREE.CylinderGeometry side), plus a slow pulse. toneMapped is disabled so
  // the additive contribution survives the ACES roll-off and stays luminous.
  const beamUniforms = { uTime: { value: 0 }, uColor: { value: TARGET_COLOR } };
  const beamMaterial = new THREE.ShaderMaterial({
    uniforms: beamUniforms,
    vertexShader: [
      "varying float vH;",
      "void main() {",
      "  vH = uv.y;",
      "  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);",
      "}",
    ].join("\n"),
    fragmentShader: [
      "uniform float uTime;",
      "uniform vec3 uColor;",
      "varying float vH;",
      "void main() {",
      "  float alpha = pow(1.0 - vH, 2.2);",
      "  float pulse = 0.80 + 0.20 * sin(uTime * 3.5);",
      "  vec3 col = uColor * (1.6 + pulse * 0.5);",
      "  gl_FragColor = vec4(col, alpha * pulse * 0.85);",
      "}",
    ].join("\n"),
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    toneMapped: false,
  });
  const BEAM_H = 2.6;
  const beamCore = new THREE.Mesh(
    new THREE.CylinderGeometry(0.07, 0.07, BEAM_H, 24, 1, true),
    beamMaterial,
  );
  beamCore.position.y = BEAM_H / 2;
  targetGroup.add(beamCore);
  const beamOuter = new THREE.Mesh(
    new THREE.CylinderGeometry(0.24, 0.30, BEAM_H, 24, 1, true),
    beamMaterial,
  );
  beamOuter.position.y = BEAM_H / 2;
  targetGroup.add(beamOuter);

  // Local point light at the disk: softly lights nearby props (floor, unicycle wheel)
  // with the target color so the beacon visibly bleeds onto the scene.
  const beaconLight = new THREE.PointLight(TARGET_COLOR, 2.4, 3.5, 2.0);
  beaconLight.position.set(0, 0.25, 0);
  targetGroup.add(beaconLight);

  // Spawn the target in a ring around the origin. 1.5 m keeps it clear of the starting
  // pose of the unicycle, 3.5 m keeps it on-camera without zooming out.
  const TARGET_MIN_R = 1.5;
  const TARGET_MAX_R = 3.5;
  const TARGET_REACH_R = 0.45;
  function respawnTarget() {
    const theta = Math.random() * Math.PI * 2;
    const r = TARGET_MIN_R + Math.random() * (TARGET_MAX_R - TARGET_MIN_R);
    // Three.js coords: ground plane is XZ, Y is up.
    targetGroup.position.set(Math.cos(theta) * r, 0, Math.sin(theta) * r);
  }
  respawnTarget();

  // Wheel body id (simplified model) for the reached-target check each frame.
  const S_WHEEL_BID = bodyId(model, "wheel");

  const keys = { wheelFwd: false, wheelBwd: false };
  const WHEEL_TORQUE = WHEEL_TORQUE_MAX;
  let padNormX = 0;
  let padNormY = 0;
  let padActive = false;

  function applyControls() {
    // Pad → open-loop torques (same stick axes as before; no angle PD / no LQR).
    const padY = -padNormY;
    const uy = padY * PELVIS_Y_TAU;
    const ux = padNormX * PELVIS_X_TAU;
    const uWheel = (keys.wheelFwd ? WHEEL_TORQUE : 0) - (keys.wheelBwd ? WHEEL_TORQUE : 0);
    data.ctrl[ACT.pelvis_y] = clamp(uy, PELVIS_Y_LO, PELVIS_Y_HI);
    data.ctrl[ACT.pelvis_x] = clamp(ux, PELVIS_X_LO, PELVIS_X_HI);
    data.ctrl[ACT.wheel] = clamp(uWheel, WHEEL_LO, WHEEL_HI);
  }

  renderer.domElement.style.touchAction = "none";

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.6, 0);
  controls.enableDamping = true;
  controls.enableRotate = true;
  controls.enablePan = true;
  controls.enableZoom = true;
  controls.minDistance = 0.5;
  controls.maxDistance = 15;
  // Standard mappings: left = orbit, middle = dolly, right = pan; one finger = orbit, two = pinch+pan.
  controls.mouseButtons = {
    LEFT: THREE.MOUSE.ROTATE,
    MIDDLE: THREE.MOUSE.DOLLY,
    RIGHT: THREE.MOUSE.PAN,
  };
  controls.touches = {
    ONE: THREE.TOUCH?.ROTATE ?? 0,
    TWO: THREE.TOUCH?.DOLLY_PAN ?? 2,
  };

  const uiPanel = document.createElement("div");
  uiPanel.id = "unicycle-ui";
  uiPanel.style.cssText = [
    "position:absolute",
    "top:12px",
    "left:12px",
    "display:flex",
    "flex-direction:column",
    "gap:8px",
    "width:min(360px, calc(100% - 24px))",
    "max-height:calc(100% - 24px)",
    "padding:12px 14px",
    "border:1px solid rgba(255,255,255,0.14)",
    "border-radius:12px",
    "background:rgba(10,16,24,0.66)",
    "box-shadow:0 12px 28px rgba(0,0,0,0.28)",
    "backdrop-filter:blur(10px)",
    "color:#fff",
    "font:12px/1.5 sans-serif",
    "overflow-y:auto",
    "pointer-events:none",
    "text-shadow:0 1px 2px rgba(0,0,0,0.55)",
  ].join(";");
  uiPanel.innerHTML = [
    "<div style='font:600 14px/1.4 sans-serif'>Unicycle MuJoCo</div>",
    "<div style='color:rgba(255,255,255,0.78)'>Drag to orbit, right-drag to pan, wheel to zoom.</div>",
    "<div id='unicycle-manual-copy'>Pad: open-loop pelvis torques (no balance assist). <kbd>W</kbd>/<kbd>S</kbd> or <kbd>↑</kbd>/<kbd>↓</kbd>: wheel.</div>",
    "<div><span style='color:#3dff8c;font-weight:600'>Goal:</span> reach the green beacon.</div>",
    "<div><kbd>Space</kbd> pause · <kbd>R</kbd> reset</div>",
    "<label style='display:flex;align-items:flex-start;gap:8px;pointer-events:auto;cursor:pointer'>",
    "  <input id='unicycle-ragdoll' type='checkbox' style='margin-top:2px'>",
    "  <span>Show Humanoid (<kbd>H</kbd>)</span>",
    "</label>",
  ].join("");
  for (const kbd of uiPanel.querySelectorAll("kbd")) {
    kbd.style.cssText = [
      "display:inline-block",
      "padding:1px 5px",
      "border:1px solid rgba(255,255,255,0.18)",
      "border-radius:5px",
      "background:rgba(255,255,255,0.08)",
      "font:11px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
      "color:#fff",
      "text-shadow:none",
    ].join(";");
  }
  container.appendChild(uiPanel);

  function resetSimulation() {
    mujoco.mj_resetData(model, data);
    mujoco.mj_forward(model, data);
    mujoco.mj_resetData(model_v, data_v);
    mujoco.mj_forward(model_v, data_v);
    if (ragdollEnabled) syncVisualFromSimplified();
    respawnTarget();
  }

  function setRagdollEnabled(enabled) {
    ragdollEnabled = !!enabled;
    const cb = document.getElementById("unicycle-ragdoll");
    if (cb) cb.checked = ragdollEnabled;
    applyRagdollVisibility();
    if (ragdollEnabled) syncVisualFromSimplified();
  }
  const ragdollCheckbox = uiPanel.querySelector("#unicycle-ragdoll");
  if (ragdollCheckbox) {
    ragdollCheckbox.addEventListener("change", (e) => setRagdollEnabled(e.target.checked));
  }

  // Ensure the pad can be positioned relative to the container.
  const containerPos = getComputedStyle(container).position;
  if (containerPos === "static" || !containerPos) {
    container.style.position = "relative";
  }

  const PAD_SIZE = 180;
  const pad = document.createElement("div");
  pad.id = "unicycle-pad";
  pad.style.cssText = [
    "position:absolute",
    "right:12px",
    "bottom:12px",
    `width:${PAD_SIZE}px`,
    `height:${PAD_SIZE}px`,
    "background:rgba(0,0,0,0.35)",
    "border:1px solid rgba(255,255,255,0.55)",
    "border-radius:6px",
    "box-shadow:0 1px 4px rgba(0,0,0,0.5)",
    "touch-action:none",
    "cursor:crosshair",
    "user-select:none",
  ].join(";");

  const crosshair = document.createElement("div");
  crosshair.style.cssText = [
    "position:absolute",
    "left:0",
    "top:50%",
    "width:100%",
    "height:0",
    "border-top:1px dashed rgba(255,255,255,0.35)",
    "pointer-events:none",
  ].join(";");
  pad.appendChild(crosshair);
  const crosshairV = document.createElement("div");
  crosshairV.style.cssText = [
    "position:absolute",
    "top:0",
    "left:50%",
    "width:0",
    "height:100%",
    "border-left:1px dashed rgba(255,255,255,0.35)",
    "pointer-events:none",
  ].join(";");
  pad.appendChild(crosshairV);

  const centerDot = document.createElement("div");
  const CENTER_DOT = 8;
  centerDot.style.cssText = [
    "position:absolute",
    `left:${(PAD_SIZE - CENTER_DOT) / 2}px`,
    `top:${(PAD_SIZE - CENTER_DOT) / 2}px`,
    `width:${CENTER_DOT}px`,
    `height:${CENTER_DOT}px`,
    "border:1px solid rgba(255,255,255,0.8)",
    "border-radius:50%",
    "pointer-events:none",
  ].join(";");
  pad.appendChild(centerDot);

  const PAD_LABEL_CSS =
    "position:absolute;font:10px/1 sans-serif;color:rgba(255,255,255,0.85);text-shadow:0 1px 2px #000;pointer-events:none;";
  const labels = [
    { text: "tilt fwd", css: "left:50%;top:4px;transform:translateX(-50%);" },
    { text: "tilt back", css: "left:50%;bottom:4px;transform:translateX(-50%);" },
    { text: "roll ←", css: "left:4px;top:50%;transform:translateY(-50%);" },
    { text: "roll →", css: "right:4px;top:50%;transform:translateY(-50%);" },
  ];
  for (const l of labels) {
    const el = document.createElement("div");
    el.textContent = l.text;
    el.style.cssText = PAD_LABEL_CSS + l.css;
    pad.appendChild(el);
  }

  const KNOB = 18;
  const knob = document.createElement("div");
  knob.style.cssText = [
    "position:absolute",
    `width:${KNOB}px`,
    `height:${KNOB}px`,
    `left:${(PAD_SIZE - KNOB) / 2}px`,
    `top:${(PAD_SIZE - KNOB) / 2}px`,
    "border-radius:50%",
    "background:rgba(120,200,255,0.9)",
    "border:1px solid rgba(255,255,255,0.9)",
    "box-shadow:0 0 6px rgba(120,200,255,0.6)",
    "pointer-events:none",
    "transition:background 0.1s",
  ].join(";");
  pad.appendChild(knob);

  container.appendChild(pad);

  function updatePadFromEvent(e) {
    const rect = pad.getBoundingClientRect();
    const half = rect.width / 2;
    const cx = rect.left + half;
    const cy = rect.top + rect.height / 2;
    let nx = (e.clientX - cx) / half;
    let ny = (cy - e.clientY) / half;
    if (nx > 1) nx = 1;
    else if (nx < -1) nx = -1;
    if (ny > 1) ny = 1;
    else if (ny < -1) ny = -1;
    padNormX = nx;
    padNormY = ny;
    const kx = (PAD_SIZE - KNOB) / 2 + nx * (PAD_SIZE - KNOB) / 2;
    const ky = (PAD_SIZE - KNOB) / 2 - ny * (PAD_SIZE - KNOB) / 2;
    knob.style.left = `${kx}px`;
    knob.style.top = `${ky}px`;
  }

  function resetPad() {
    padActive = false;
    padNormX = 0;
    padNormY = 0;
    knob.style.left = `${(PAD_SIZE - KNOB) / 2}px`;
    knob.style.top = `${(PAD_SIZE - KNOB) / 2}px`;
    knob.style.background = "rgba(120,200,255,0.9)";
  }

  pad.addEventListener("pointerdown", (e) => {
    if (e.button !== undefined && e.button !== 0) return;
    padActive = true;
    knob.style.background = "rgba(255,190,90,0.95)";
    try {
      pad.setPointerCapture(e.pointerId);
    } catch (_) {
      /* ignore */
    }
    updatePadFromEvent(e);
    e.preventDefault();
  });
  pad.addEventListener("pointermove", (e) => {
    if (!padActive) return;
    updatePadFromEvent(e);
    e.preventDefault();
  });
  const endPad = (e) => {
    if (!padActive) return;
    try {
      pad.releasePointerCapture(e.pointerId);
    } catch (_) {
      /* ignore */
    }
    resetPad();
    e.preventDefault();
  };
  pad.addEventListener("pointerup", endPad);
  pad.addEventListener("pointercancel", endPad);
  pad.addEventListener("pointerleave", (e) => {
    if (padActive && !pad.hasPointerCapture?.(e.pointerId)) resetPad();
  });

  const tmpPos = new THREE.Vector3();
  const tmpQuat = new THREE.Quaternion();
  const G = mujoco.mjtGeom || {};
  const planeType = G.mjGEOM_PLANE?.value ?? 0;

  // Build a Three.js Group per body by collecting renderable geoms (group < 3, skipping floor
  // for the visual model to avoid z-fighting with the simplified floor). Material/texture
  // lookup follows zalo/mujoco_wasm: per-geom `mat_rgba`, `mat_texid[mat*10 + mjTEXROLE_RGB]`,
  // `mat_texrepeat`, `mat_{specular,reflectance,shininess}`.
  function buildBodyGroups(m, { skipBodyNames = new Set() } = {}) {
    const groups = {};
    const ngeomM = mjNum(m.ngeom);
    for (let g = 0; g < ngeomM; g++) {
      if (m.geom_group[g] >= 3) continue;
      const bid = mjNum(m.geom_bodyid[g]);
      const bname = mujoco.mj_id2name(m, OBJ_BODY, bid) ?? "";
      const type = mjNum(m.geom_type[g]);
      const isPlane = type === planeType;
      if (m === model_v && isPlane) continue;
      if (skipBodyNames.has(bname)) continue;
      if (!(bid in groups)) {
        groups[bid] = new THREE.Group();
        scene.add(groups[bid]);
      }

      let color = [
        mjNum(m.geom_rgba[g * 4]),
        mjNum(m.geom_rgba[g * 4 + 1]),
        mjNum(m.geom_rgba[g * 4 + 2]),
        mjNum(m.geom_rgba[g * 4 + 3]),
      ];
      let map = null;
      let specular, reflectance, shininess;
      const matId = m.geom_matid ? mjNum(m.geom_matid[g]) : -1;
      if (matId !== -1) {
        color = [
          mjNum(m.mat_rgba[matId * 4]),
          mjNum(m.mat_rgba[matId * 4 + 1]),
          mjNum(m.mat_rgba[matId * 4 + 2]),
          mjNum(m.mat_rgba[matId * 4 + 3]),
        ];
        specular = mjNum(m.mat_specular[matId]);
        reflectance = mjNum(m.mat_reflectance[matId]);
        shininess = mjNum(m.mat_shininess[matId]);
        const texId = mjNum(m.mat_texid[matId * MJ_NTEXROLE + MJ_TEXROLE_RGB]);
        if (texId >= 0 && texId < modelTextures.length) {
          const entry = modelTextures[texId];
          // Skybox textures belong on the scene background, not on geoms.
          if (mjNum(entry.type) !== MJ_TEX_SKYBOX) {
            map = entry.tex.clone();
            map.needsUpdate = true;
            map.wrapS = THREE.RepeatWrapping;
            map.wrapT = THREE.RepeatWrapping;
            // Plane geoms (floors) ignore material texrepeat=1,1 and tile by world scale;
            // zalo/mujoco_wasm applies a hard-coded 50× repeat for that case. Do the same.
            if (isPlane) {
              map.repeat.set(50, 50);
            } else {
              map.repeat.set(
                mjNum(m.mat_texrepeat[matId * 2]) || 1,
                mjNum(m.mat_texrepeat[matId * 2 + 1]) || 1
              );
            }
          }
        }
      }

      const matParams = {
        color: new THREE.Color(color[0], color[1], color[2]),
        transparent: color[3] < 1,
        opacity: color[3],
        map,
      };
      if (matId !== -1) {
        matParams.roughness = 1.0 - shininess;
        matParams.metalness = 0.1;
        matParams.reflectivity = reflectance;
        matParams.specularIntensity = specular;
      } else {
        matParams.roughness = 0.55;
        matParams.metalness = 0.1;
      }
      const mat = new THREE.MeshPhysicalMaterial(matParams);
      const mesh = createMeshForGeom(mujoco, m, g, mat, isPlane);
      groups[bid].add(mesh);
    }
    return groups;
  }

  const bodies = buildBodyGroups(model);
  const bodies_v = buildBodyGroups(model_v);

  // Map simplified body-name → id for the ragdoll-toggle hide list.
  const simplifiedHideIds = new Set();
  for (const name of HIDE_ON_RAGDOLL) {
    const id = mjNum(mujoco.mj_name2id(model, OBJ_BODY, name));
    if (id >= 0) simplifiedHideIds.add(id);
  }

  // Toggle state: when true, render the visual humanoid and hide overlapping simplified bodies.
  let ragdollEnabled = false;
  function applyRagdollVisibility() {
    for (const [bidStr, grp] of Object.entries(bodies)) {
      const bid = Number(bidStr);
      grp.visible = !ragdollEnabled || !simplifiedHideIds.has(bid);
    }
    for (const grp of Object.values(bodies_v)) {
      grp.visible = ragdollEnabled;
    }
  }
  // Humanoid visible and checkbox on by default (matches <kbd>H</kbd> behavior).
  setRagdollEnabled(true);

  // One-way sync: simplified -> visual.
  //   * Free joint (wheel pose + twist): direct copy of qpos[0..6] / qvel[0..5].
  //     Both models have the wheel body as root, so these 7/6 values have identical
  //     meaning (position + quaternion / linear + angular velocity of the wheel in world).
  //   * Pelvis hinges + wheel revolute: pin angle and velocity by matching joint name;
  //     this drives the visual humanoid's upper body tilt from the simplified COM tilt
  //     and keeps the visual wheel spin in lock-step with the simplified wheel.
  //   * Abdomen triplet (z/y/x): held at zero so the humanoid's upper body stays rigid
  //     above the pelvis hinge, matching the simplified rigid com segment.
  // All other visual DOFs (legs, arms, pedals) evolve under gravity, joint stiffness,
  // and the foot<->pedal welds / hand<->head connect equalities each visual step.
  function writeVisualPins() {
    const qs = data.qpos;
    const vs = data.qvel;
    data_v.qpos[0] = qs[0]; data_v.qpos[1] = qs[1]; data_v.qpos[2] = qs[2];
    data_v.qpos[3] = qs[3]; data_v.qpos[4] = qs[4]; data_v.qpos[5] = qs[5]; data_v.qpos[6] = qs[6];
    data_v.qvel[0] = vs[0]; data_v.qvel[1] = vs[1]; data_v.qvel[2] = vs[2];
    data_v.qvel[3] = vs[3]; data_v.qvel[4] = vs[4]; data_v.qvel[5] = vs[5];
    for (const p of pinMap) {
      data_v.qpos[p.v.qposadr] = data.qpos[p.s.qposadr];
      data_v.qvel[p.v.dofadr] = data.qvel[p.s.dofadr];
    }
    for (const a of zeroPinAddrs) {
      data_v.qpos[a.qposadr] = 0;
      data_v.qvel[a.dofadr] = 0;
    }
  }
  function syncVisualFromSimplified() {
    writeVisualPins();
    for (let i = 0; i < mjNum(model_v.nu); i++) data_v.ctrl[i] = 0;
    mujoco.mj_step(model_v, data_v);
    // Re-pin after step so pinned DOFs don't drift due to solver interactions.
    writeVisualPins();
    mujoco.mj_forward(model_v, data_v);
  }

  let paused = false;
  let disposed = false;
  let lastTime = performance.now();

  function clearKeys() {
    keys.wheelFwd = false;
    keys.wheelBwd = false;
  }

  function claimKeyboardEvent(e) {
    if (!UNICYCLE_KEY_CODES.has(e.code)) return false;
    if (!controlsActive) return false;
    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation?.();
    return true;
  }

  function onKeyDown(e) {
    if (!claimKeyboardEvent(e)) return;
    switch (e.code) {
      case "KeyW":
        keys.wheelFwd = true;
        break;
      case "KeyS":
        keys.wheelBwd = true;
        break;
      case "ArrowUp":
        keys.wheelFwd = true;
        break;
      case "ArrowDown":
        keys.wheelBwd = true;
        break;
      case "Space": paused = !paused; return;
      case "KeyR":
        resetSimulation();
        return;
      case "KeyH":
        setRagdollEnabled(!ragdollEnabled);
        return;
      case "Escape":
        releaseControls();
        renderer.domElement.blur();
        return;
      default: return;
    }
  }

  function onKeyUp(e) {
    if (!claimKeyboardEvent(e)) return;
    switch (e.code) {
      case "KeyW":
        keys.wheelFwd = false;
        break;
      case "KeyS":
        keys.wheelBwd = false;
        break;
      case "ArrowUp":
        keys.wheelFwd = false;
        break;
      case "ArrowDown":
        keys.wheelBwd = false;
        break;
      default: return;
    }
  }

  window.addEventListener("keydown", onKeyDown, { capture: true });
  window.addEventListener("keyup", onKeyUp, { capture: true });
  renderer.domElement.addEventListener("blur", releaseControls);
  window.addEventListener("blur", releaseControls);

  function animate() {
    if (disposed) return;
    requestAnimationFrame(animate);
    const now = performance.now();
    const dt = (now - lastTime) / 1000;
    lastTime = now;

    if (!paused) {
      applyControls();
      const timestep = SIM_TIMESTEP;
      let steps = Math.max(1, Math.floor(dt / timestep));
      steps = Math.min(steps, 5);
      for (let i = 0; i < steps; i++) {
        mujoco.mj_step(model, data);
        if (ragdollEnabled) syncVisualFromSimplified();
      }
    }

    // Advance the beam shader time and gently rotate the ring for a parallax cue.
    beamUniforms.uTime.value = now / 1000;
    glowRing.rotation.z += dt * 0.6;

    // Reached target? Check distance in the world XZ plane between the wheel and the
    // beacon using Three coords. xpos is MuJoCo (x, y, z) with z up, and we map to
    // Three.js as (x, z, -y) elsewhere — so the ground-plane coords are (mx, -my).
    const wx = data.xpos[S_WHEEL_BID * 3];
    const wy = data.xpos[S_WHEEL_BID * 3 + 1];
    const tx = targetGroup.position.x;
    const tz = targetGroup.position.z;
    const dx = wx - tx;
    const dz = -wy - tz;
    if (dx * dx + dz * dz < TARGET_REACH_R * TARGET_REACH_R) {
      respawnTarget();
    }

    for (let b = 0; b < mjNum(model.nbody); b++) {
      if (bodies[b]) {
        getMujocoPos(data, b, tmpPos);
        getMujocoQuat(data, b, tmpQuat);
        bodies[b].position.copy(tmpPos);
        bodies[b].quaternion.copy(tmpQuat);
        bodies[b].updateMatrixWorld(true);
      }
    }
    if (ragdollEnabled) {
      for (let b = 0; b < mjNum(model_v.nbody); b++) {
        if (bodies_v[b]) {
          getMujocoPos(data_v, b, tmpPos);
          getMujocoQuat(data_v, b, tmpQuat);
          bodies_v[b].position.copy(tmpPos);
          bodies_v[b].quaternion.copy(tmpQuat);
          bodies_v[b].updateMatrixWorld(true);
        }
      }
    }

    controls.update();
    renderer.render(scene, camera);
  }

  window.unicyclePause = () => { paused = !paused; };
  window.unicycleReset = () => {
    resetSimulation();
  };
  window.unicycleToggleRagdoll = () => setRagdollEnabled(!ragdollEnabled);
  window.unicycleRespawnTarget = respawnTarget;

  animate();

  window.addEventListener("resize", () => {
    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  window.addEventListener("beforeunload", () => {
    try {
      disposed = true;
      if (typeof data_v?.delete === "function") data_v.delete();
      if (typeof model_v?.delete === "function") model_v.delete();
      if (typeof data?.delete === "function") data.delete();
      if (typeof model?.delete === "function") model.delete();
      if (typeof staged.vfs?.delete === "function") staged.vfs.delete();
    } catch (_) {
      /* ignore */
    }
    controls.dispose();
    renderer.dispose();
  });
}
