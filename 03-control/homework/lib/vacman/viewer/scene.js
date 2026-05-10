/**
 * Three.js renderer, arena floor, path walls, charging-station voxel stack.
 */
import * as THREE from "https://esm.sh/three@0.165.0";
import { BASE_VISUAL_R, DUST_N, WALL_HALF_T } from "./constants.js";
import { dist } from "./math2d.js";

function makeGoalGlowTexture(size = 128) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;

  const ctx = canvas.getContext("2d");
  const c = size / 2;
  const r = size / 2;
  const gradient = ctx.createRadialGradient(c, c, 0, c, c, r);
  gradient.addColorStop(0.00, "rgba(61,255,140,0.75)");
  gradient.addColorStop(0.38, "rgba(61,255,140,0.30)");
  gradient.addColorStop(0.72, "rgba(61,255,140,0.10)");
  gradient.addColorStop(1.00, "rgba(61,255,140,0.00)");

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(c, c, r, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(190,255,220,0.85)";
  ctx.lineWidth = Math.max(2, size * 0.025);
  ctx.beginPath();
  ctx.arc(c, c, size * 0.34, 0, Math.PI * 2);
  ctx.stroke();

  return canvas;
}

export function createArenaScene(container, { halfX, halfZ, ARENA_X, ARENA_Z, wallRects }) {
  const W = container.clientWidth || window.innerWidth;
  const H = container.clientHeight || window.innerHeight;
  const arenaScale = Math.max(ARENA_X, ARENA_Z);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  const cameraState = {
    zoom: 1,
    minZoom: 0.72,
    maxZoom: 1.65,
    baseHeight: Math.max(18, arenaScale * 2.25),
    followDistance: Math.max(4.5, arenaScale * 0.55),
    lateralOffset: Math.max(1.2, arenaScale * 0.14),
    lookAhead: Math.max(1.5, arenaScale * 0.18),
    lookLift: Math.max(0.45, arenaScale * 0.03),
  };

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1117);
  scene.fog = new THREE.FogExp2(0x0d1117, Math.min(0.008, 0.075 / arenaScale));

  const worldGroup = new THREE.Group();
  scene.add(worldGroup);

  const camera = new THREE.PerspectiveCamera(42, W / H, 0.1, 500);
  camera.up.set(0, 0, 1);
  camera.position.set(0, cameraState.baseHeight, cameraState.followDistance);

  scene.add(new THREE.AmbientLight(0xffffff, 1.18));
  scene.add(new THREE.HemisphereLight(0xf2f8ff, 0x4f5f86, 1.15));
  const sun = new THREE.DirectionalLight(0xfff6e6, 2.1);
  sun.position.set(-10, 42, -8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const sc = Math.max(halfX, halfZ) + 8;
  Object.assign(sun.shadow.camera, { left: -sc, right: sc, top: sc, bottom: -sc, near: 1, far: 140 });
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xc7ddff, 1.25);
  fill.position.set(12, 24, 14);
  scene.add(fill);
  const topFill = new THREE.DirectionalLight(0xe8f0ff, 0.78);
  topFill.position.set(0, 34, 0);
  scene.add(topFill);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(ARENA_X, ARENA_Z),
    new THREE.MeshStandardMaterial({ color: 0x3f4969, roughness: 0.78, metalness: 0.0 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  worldGroup.add(floor);

  const gridMax = Math.max(ARENA_X, ARENA_Z);
  const gridDiv = Math.max(4, Math.round(gridMax / 2));
  const gridHelper = new THREE.GridHelper(gridMax, gridDiv, 0x4c5a82, 0x27304b);
  gridHelper.scale.set(ARENA_X / gridMax, 1, ARENA_Z / gridMax);
  worldGroup.add(gridHelper);

  const wallMat = new THREE.MeshStandardMaterial({ color: 0x7380bb, roughness: 0.58 });
  const wH = 1.5;
  for (const rect of wallRects) {
    const p0 = { x: (rect[0].x + rect[1].x) / 2, z: (rect[0].z + rect[1].z) / 2 };
    const p1 = { x: (rect[2].x + rect[3].x) / 2, z: (rect[2].z + rect[3].z) / 2 };
    const segLen = Math.max(dist(p0, p1) + WALL_HALF_T * 2, WALL_HALF_T * 2);
    const w = new THREE.Mesh(new THREE.BoxGeometry(WALL_HALF_T * 2, wH, segLen), wallMat);
    w.position.set((p0.x + p1.x) / 2, wH / 2, (p0.z + p1.z) / 2);
    w.rotation.y = Math.atan2(p1.x - p0.x, p1.z - p0.z);
    w.castShadow = true;
    w.receiveShadow = true;
    worldGroup.add(w);
  }

  const cellSizeX = ARENA_X / DUST_N;
  const cellSizeZ = ARENA_Z / DUST_N;
  const cellSizeMin = Math.min(cellSizeX, cellSizeZ);

  const baseGroup = new THREE.Group();
  worldGroup.add(baseGroup);

  const goalGlowTex = new THREE.CanvasTexture(makeGoalGlowTexture());
  goalGlowTex.magFilter = THREE.LinearFilter;
  goalGlowTex.minFilter = THREE.LinearFilter;
  const goalGlowMaterial = new THREE.MeshBasicMaterial({
    map: goalGlowTex,
    transparent: true,
    opacity: 0,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
  });
  const goalGlow = new THREE.Mesh(
    new THREE.PlaneGeometry(BASE_VISUAL_R * 2.35, BASE_VISUAL_R * 2.35),
    goalGlowMaterial,
  );
  goalGlow.rotation.x = -Math.PI / 2;
  goalGlow.position.y = 0.035;
  goalGlow.renderOrder = 2;
  goalGlow.visible = false;
  baseGroup.add(goalGlow);

  const voxelMat = new THREE.MeshStandardMaterial({
    color: 0x00e676,
    emissive: 0x00c853,
    emissiveIntensity: 0.05,
    roughness: 0.35,
  });
  const beaconVoxelMat = new THREE.MeshStandardMaterial({
    color: 0x69f0ae,
    emissive: 0x00e676,
    emissiveIntensity: 0.08,
    roughness: 0.3,
  });

  const beaconVoxels = [];
  function addVoxel(localX, localZ, y, h, mat) {
    const m = new THREE.Mesh(
      new THREE.BoxGeometry(cellSizeX * 0.98, h, cellSizeZ * 0.98),
      mat,
    );
    m.position.set(localX, y + h / 2, localZ);
    m.castShadow = true;
    m.receiveShadow = true;
    baseGroup.add(m);
    if (mat === beaconVoxelMat) beaconVoxels.push(m);
    return m;
  }

  renderer.domElement.addEventListener("wheel", (event) => {
    event.preventDefault();
    const scale = event.deltaY > 0 ? 1.08 : 0.92;
    cameraState.zoom = THREE.MathUtils.clamp(
      cameraState.zoom * scale,
      cameraState.minZoom,
      cameraState.maxZoom,
    );
  }, { passive: false });

  return {
    renderer,
    scene,
    worldGroup,
    camera,
    cameraState,
    baseGroup,
    goalGlow,
    goalGlowMaterial,
    beaconVoxels,
    beaconVoxelMat,
    cellSizeX,
    cellSizeZ,
    cellSizeMin,
    voxelMat,
    addVoxel,
  };
}

export function layoutBaseStation(baseGroup, basePos, halfX, halfZ, { cellSizeX, cellSizeZ, cellSizeMin, addVoxel, voxelMat, beaconVoxelMat }) {
  baseGroup.position.set(basePos.x, 0, basePos.z);

  const gx0 = Math.floor((-halfX - basePos.x) / cellSizeX);
  const gx1 = Math.ceil((halfX - basePos.x) / cellSizeX);
  const gz0 = Math.floor((-halfZ - basePos.z) / cellSizeZ);
  const gz1 = Math.ceil((halfZ - basePos.z) / cellSizeZ);
  for (let gi = gx0; gi <= gx1; gi++) {
    for (let gj = gz0; gj <= gz1; gj++) {
      const lx = gi * cellSizeX + cellSizeX / 2;
      const lz = gj * cellSizeZ + cellSizeZ / 2;
      const wx = basePos.x + lx;
      const wz = basePos.z + lz;
      if (Math.hypot(wx - basePos.x, wz - basePos.z) > BASE_VISUAL_R - cellSizeMin * 0.25) continue;
      addVoxel(lx, lz, 0, 0.18, voxelMat);
    }
  }

  let poleY = 0.2;
  while (poleY < 2.15) {
    addVoxel(0, 0, poleY, cellSizeMin, voxelMat);
    poleY += cellSizeMin;
  }

  const beaconPattern = [
    [-1, 0],
    [0, 0],
    [1, 0],
    [0, 1],
    [0, -1],
    [-1, 1],
    [1, 1],
    [-1, -1],
    [1, -1],
  ];
  const beaconBaseY = poleY;
  for (const [dx, dz] of beaconPattern) {
    addVoxel(dx * cellSizeX, dz * cellSizeZ, beaconBaseY, cellSizeMin * 0.95, beaconVoxelMat);
  }
}

export function attachResize(container, camera, renderer) {
  window.addEventListener("resize", () => {
    const w = container.clientWidth || window.innerWidth;
    const h = container.clientHeight || window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
}
