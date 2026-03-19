import * as THREE from "https://esm.sh/three";
import { GLTFLoader } from "https://esm.sh/three/examples/jsm/loaders/GLTFLoader.js";
import { Line2 } from "https://esm.sh/three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "https://esm.sh/three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "https://esm.sh/three/examples/jsm/lines/LineMaterial.js";
import { CSS2DRenderer, CSS2DObject } from "https://esm.sh/three/examples/jsm/renderers/CSS2DRenderer.js";

const config = window.QUIDDICH_VISUALIZER || {};
const containerId = config.containerId || "quiddich-viz-container";
const glbUrl = config.glbUrl || "assets/quiddich.glb";

const PRESET_COUNT = 6;
const GIZMO_RADIUS = 0.8;
const TANGENT_CIRCLE_RADIUS = 1.2;
const ARROW_LEN = 0.9;
const CROSS_SIZE = 1.5;
const PARALLAX_AMOUNT = 0.35;
const PARALLAX_LERP = 0.06;

function getWorldPosition(obj, out = new THREE.Vector3()) {
  if (!obj) return null;
  obj.getWorldPosition(out);
  return out;
}

function findByName(root, name) {
  let found = null;
  root.traverse((o) => {
    if (o.name === name) found = o;
  });
  return found;
}

function circlePoints(radius, plane = "xz", segments = 64) {
  const points = [];
  for (let i = 0; i <= segments; i++) {
    const t = (i / segments) * Math.PI * 2;
    if (plane === "yz") {
      points.push(new THREE.Vector3(0, radius * Math.cos(t), radius * Math.sin(t)));
    } else if (plane === "xy") {
      points.push(new THREE.Vector3(radius * Math.cos(t), radius * Math.sin(t), 0));
    } else {
      points.push(new THREE.Vector3(radius * Math.cos(t), 0, radius * Math.sin(t)));
    }
  }
  return points;
}

function halfArcPoints(radius, plane = "xz", segments = 32) {
  const points = [];
  for (let i = 0; i <= segments; i++) {
    const t = ((i / segments) + 1.05) * Math.PI;
    if (plane === "yz") {
      points.push(new THREE.Vector3(0, radius * Math.cos(t), radius * Math.sin(t)));
    } else if (plane === "xy") {
      points.push(new THREE.Vector3(radius * Math.cos(t), radius * Math.sin(t), 0));
    } else {
      points.push(new THREE.Vector3(radius * Math.cos(t), 0, radius * Math.sin(t)));
    }
  }
  return points;
}

function radiusVectorEnd(radius, plane) {
  if (plane === "yz") return new THREE.Vector3(0, radius, 0);
  return new THREE.Vector3(radius, 0, 0);
}

function makeGizmo(origin) {
  const group = new THREE.Group();
  group.position.copy(origin);
  const zero = new THREE.Vector3(0, 0, 0);

  const arrowX = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), zero, ARROW_LEN, 0x2ecc71);
  const arrowY = new THREE.ArrowHelper(new THREE.Vector3(0, -1, 0), zero, ARROW_LEN, 0xe74c3c);
  const arrowZ = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), zero, ARROW_LEN, 0x3498db);
  group.add(arrowX);
  group.add(arrowY);
  group.add(arrowZ);

  const headingGeom = new THREE.BufferGeometry().setFromPoints(circlePoints(GIZMO_RADIUS, "xy"));
  const headingLine = new THREE.Line(headingGeom, new THREE.LineBasicMaterial({ color: 0xf1c40f }));
  group.add(headingLine);

  const pitchGeom = new THREE.BufferGeometry().setFromPoints(circlePoints(GIZMO_RADIUS, "yz"));
  const pitchLine = new THREE.Line(pitchGeom, new THREE.LineBasicMaterial({ color: 0x9b59b6 }));
  group.add(pitchLine);

  return group;
}

function makeLine(start, end, color = 0x00ff00) {
  const points = [start.clone(), end.clone()];
  const geom = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.Line(geom, new THREE.LineBasicMaterial({ color }));
}

const THICK_LINE_WIDTH = 5;

function makeBezierCurve(v0, v1, v2, v3, color = 0x00ff00, resolution = new THREE.Vector2(800, 400)) {
  const curve = new THREE.CubicBezierCurve3(v0, v1, v2, v3);
  const points = curve.getPoints(64);
  const positions = [];
  points.forEach((p) => {
    positions.push(p.x, p.y, p.z);
  });
  const geom = new LineGeometry();
  geom.setPositions(positions);
  const mat = new LineMaterial({ color, linewidth: THICK_LINE_WIDTH, resolution });
  return new Line2(geom, mat);
}

function makeRedCross(center, size = CROSS_SIZE) {
  const group = new THREE.Group();
  group.position.copy(center);
  const mat = new THREE.MeshBasicMaterial({ color: 0xff0000, side: THREE.DoubleSide });
  const w = size * 0.15;
  const box1 = new THREE.Mesh(new THREE.PlaneGeometry(size * 2, w), mat);
  box1.rotation.z = Math.PI / 4;
  const box2 = new THREE.Mesh(new THREE.PlaneGeometry(size * 2, w), mat);
  box2.rotation.z = -Math.PI / 4;
  group.add(box1);
  group.add(box2);
  return group;
}

function makeTangentHalfArc(center, radius = TANGENT_CIRCLE_RADIUS, plane = "xz", resolution = new THREE.Vector2(800, 400)) {
  const group = new THREE.Group();
  group.position.copy(center);
  const points = halfArcPoints(radius, plane);
  const positions = [];
  points.forEach((p) => {
    positions.push(p.x, p.y, p.z);
  });
  const geom = new LineGeometry();
  geom.setPositions(positions);
  const mat = new LineMaterial({ color: 0x00aaff, linewidth: THICK_LINE_WIDTH, resolution });
  group.add(new Line2(geom, mat));
  return group;
}

function makeForwardArrow(origin, length = ARROW_LEN, color = 0xe74c3c, direction = null) {
  const group = new THREE.Group();
  group.position.copy(origin);
  const zero = new THREE.Vector3(0, 0, 0);
  const dir = direction ? direction.clone().normalize() : new THREE.Vector3(1, 0, 0);
  const arrow = new THREE.ArrowHelper(dir, zero, length, color);
  group.add(arrow);
  return group;
}

function getObjectForward(object, out = new THREE.Vector3()) {
  const quat = new THREE.Quaternion();
  object.getWorldQuaternion(quat);
  return out.set(1, 0, 0).applyQuaternion(quat);
}

const labelStyle = "color:rgba(255,255,255,0.95);text-shadow:0 1px 4px rgba(0,0,0,0.9);font:600 14px system-ui,sans-serif;white-space:nowrap;pointer-events:none;";

function makeLabel3D(text) {
  const div = document.createElement("div");
  div.textContent = text;
  div.style.cssText = labelStyle;
  const obj = new CSS2DObject(div);
  return obj;
}

function run() {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.style.position = "relative";

  const baseCameraPos = new THREE.Vector3();
  const baseLookAt = new THREE.Vector3();
  const mouseOffset = new THREE.Vector3();
  const targetParallaxOffset = new THREE.Vector3();
  const targetMouseNorm = { x: 0, y: 0 };
  let parallaxReady = false;

  const width = container.clientWidth || 800;
  const height = 400;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);

  const ambient = new THREE.AmbientLight(0x606080, 0.85);
  scene.add(ambient);
  const hemi = new THREE.HemisphereLight(0xb0c0e8, 0x506080, 0.7);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff5e6, 1.4);
  sun.position.set(0, 25, 5);
  scene.add(sun);
  const key = new THREE.DirectionalLight(0xffffff, 0.7);
  key.position.set(5, 10, 5);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xa0b0e0, 0.5);
  fill.position.set(-5, 5, -5);
  scene.add(fill);
  const back = new THREE.DirectionalLight(0x6080a0, 0.4);
  back.position.set(0, -3, 5);
  scene.add(back);

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  const css2DRenderer = new CSS2DRenderer();
  css2DRenderer.setSize(width, height);
  css2DRenderer.domElement.style.position = "absolute";
  css2DRenderer.domElement.style.top = "0";
  css2DRenderer.domElement.style.left = "0";
  css2DRenderer.domElement.style.pointerEvents = "none";
  container.appendChild(css2DRenderer.domElement);

  container.addEventListener("mousemove", (e) => {
    const r = container.getBoundingClientRect();
    targetMouseNorm.x = (e.clientX - r.left) / r.width * 2 - 1;
    targetMouseNorm.y = -((e.clientY - r.top) / r.height * 2 - 1);
  });
  container.addEventListener("mouseleave", () => {
    targetMouseNorm.x = 0;
    targetMouseNorm.y = 0;
  });

  const overlayRoot = new THREE.Group();
  scene.add(overlayRoot);

  let gltfScene = null;
  const objects = {};

  const loader = new GLTFLoader();
  loader.load(
    glbUrl,
    (gltf) => {
      gltfScene = gltf.scene;
      const glbWrapper = new THREE.Group();
      glbWrapper.rotation.x = -Math.PI / 2;
      glbWrapper.add(gltfScene);
      scene.add(glbWrapper);

      objects.HarryPotter = findByName(gltfScene, "HarryPotter");
      objects.HPEmpty = findByName(gltfScene, "HPEmpty");
      objects.Slytherin = findByName(gltfScene, "Slytherin");
      objects.Ring1 = findByName(gltfScene, "Ring");
      objects.Snitch = findByName(gltfScene, "Snitch");
      objects.Curvature = findByName(gltfScene, "Curvature");
      objects.BigNo1 = findByName(gltfScene, "BigNo1");
      objects.BigNo2 = findByName(gltfScene, "BigNo2");

      glbWrapper.updateMatrixWorld(true);
      const hpWorldQ = new THREE.Quaternion();
      objects.HarryPotter.getWorldQuaternion(hpWorldQ);
      const hp = getWorldPosition(objects.HarryPotter, new THREE.Vector3()) || new THREE.Vector3(0, 0, 0);
      const sl = getWorldPosition(objects.Slytherin, new THREE.Vector3()) || new THREE.Vector3(0, 0, 0);
      const r1 = getWorldPosition(objects.Ring1, new THREE.Vector3()) || new THREE.Vector3(0, 0, 0);
      const sn = getWorldPosition(objects.Snitch, new THREE.Vector3()) || new THREE.Vector3(0, 0, 0);
      const cu = getWorldPosition(objects.Curvature, new THREE.Vector3()) || new THREE.Vector3(0, 0, 0);
      const b1 = getWorldPosition(objects.BigNo1, new THREE.Vector3()) || new THREE.Vector3(0, 0, 0);
      const b2 = getWorldPosition(objects.BigNo2, new THREE.Vector3()) || new THREE.Vector3(0, 0, 0);

      const gizmo = makeGizmo(hp.clone());
      gizmo.quaternion.copy(hpWorldQ);
      gizmo.updateMatrixWorld();
      overlayRoot.add(gizmo);
      gizmo.visible = false;

      const resolution = new THREE.Vector2(container.clientWidth || 800, container.clientHeight || 400);
      const ringDir = r1.clone().sub(sl).normalize();
      const ringLen = r1.distanceTo(sl);
      const ringPerp = new THREE.Vector3(-ringDir.z, 0.3, ringDir.x).normalize();
      const ringLoop = ringLen * 0.6;
      const c1 = sl.clone().add(ringDir.clone().multiplyScalar(ringLen * 0.35)).add(ringPerp.clone().multiplyScalar(ringLoop));
      const ringEndTangent = new THREE.Vector3(-1, 0, 0);
      const ringTangentDist = 1.8;
      const c2 = r1.clone().add(ringEndTangent.clone().multiplyScalar(-ringTangentDist));
      const bezier = makeBezierCurve(sl, c1, c2, r1, 0x3498db, resolution);
      overlayRoot.add(bezier);
      bezier.visible = false;

      const snitchDir = sn.clone().sub(hp).normalize();
      const snitchLen = sn.distanceTo(hp);
      const snitchPerp = new THREE.Vector3(-snitchDir.z, 0.4, snitchDir.x).normalize();
      const snitchLoop = snitchLen * 0.55;
      const snC1 = hp.clone().add(snitchDir.clone().multiplyScalar(snitchLen * 0.4)).add(snitchPerp.clone().multiplyScalar(snitchLoop));
      const snC2 = sn.clone().add(snitchDir.clone().multiplyScalar(-snitchLen * 0.4)).add(snitchPerp.clone().multiplyScalar(-snitchLoop));
      const snitchPath = makeBezierCurve(hp, snC1, snC2, sn, 0xf1c40f, resolution);
      overlayRoot.add(snitchPath);
      snitchPath.visible = false;

      const redCross = makeRedCross(cu.clone());
      overlayRoot.add(redCross);
      redCross.visible = false;

      const redCrossBigNo1 = makeRedCross(b1.clone());
      overlayRoot.add(redCrossBigNo1);
      redCrossBigNo1.visible = false;
      const redCrossBigNo2 = makeRedCross(b2.clone());
      overlayRoot.add(redCrossBigNo2);
      redCrossBigNo2.visible = false;

      const tangentCircle = makeTangentHalfArc(cu.clone().add(new THREE.Vector3(-0.2, -0.3, 1.2)), TANGENT_CIRCLE_RADIUS, "xz", resolution);
      overlayRoot.add(tangentCircle);
      tangentCircle.visible = false;

      const FORWARD_ARROW_LEN = 1.5;
      const dirBigNo1 = getObjectForward(objects.BigNo1);
      const forwardArrowBigNo1 = makeForwardArrow(b1.clone(), FORWARD_ARROW_LEN, 0xff6644, dirBigNo1);
      overlayRoot.add(forwardArrowBigNo1);
      forwardArrowBigNo1.visible = false;
      const dirBigNo2 = getObjectForward(objects.BigNo2).multiplyScalar(-1.0);
      const forwardArrowBigNo2 = makeForwardArrow(b2.clone(), FORWARD_ARROW_LEN, 0xff6644, dirBigNo2);
      overlayRoot.add(forwardArrowBigNo2);
      forwardArrowBigNo2.visible = false;

      const labelX = makeLabel3D("y");
      labelX.position.set(ARROW_LEN + 0.25, 0, 0);
      gizmo.add(labelX);
      labelX.visible = false;
      const labelY = makeLabel3D("x");
      labelY.position.set(0, -(ARROW_LEN + 0.25), 0);
      gizmo.add(labelY);
      labelY.visible = false;
      const labelZ = makeLabel3D("z");
      labelZ.position.set(0, 0, ARROW_LEN + 0.25);
      gizmo.add(labelZ);
      labelZ.visible = false;
      const labelTheta = makeLabel3D("θ");
      labelTheta.position.set(0, -GIZMO_RADIUS * 0.85, GIZMO_RADIUS * 0.85);
      gizmo.add(labelTheta);
      labelTheta.visible = false;
      const labelPhi = makeLabel3D("φ");
      labelPhi.position.set(-GIZMO_RADIUS * 0.85, -GIZMO_RADIUS * 0.85, 0);
      gizmo.add(labelPhi);
      labelPhi.visible = false;

      const ringCurve = new THREE.CubicBezierCurve3(sl, c1, c2, r1);
      const labelGatePass = makeLabel3D("Gate Pass");
      labelGatePass.position.copy(ringCurve.getPoint(0.5)).add(new THREE.Vector3(0, 0.45, 0));
      scene.add(labelGatePass);
      labelGatePass.visible = false;

      const snitchCurve = new THREE.CubicBezierCurve3(hp, snC1, snC2, sn);
      const labelCatchSnitch = makeLabel3D("Catch Snitch");
      labelCatchSnitch.position.copy(snitchCurve.getPoint(0.5)).add(new THREE.Vector3(0, 0.45, 0));
      scene.add(labelCatchSnitch);
      labelCatchSnitch.visible = false;

      const labelNoCurvature = makeLabel3D("No extreme curvature");
      labelNoCurvature.position.copy(cu).add(new THREE.Vector3(0, -0.95, 0));
      scene.add(labelNoCurvature);
      labelNoCurvature.visible = false;

      const labelNoDown = makeLabel3D("No extreme downward direction");
      labelNoDown.position.copy(b1).add(new THREE.Vector3(0, -0.95, 0));
      scene.add(labelNoDown);
      labelNoDown.visible = false;

      const labelNoUp = makeLabel3D("No extreme upward direction");
      labelNoUp.position.copy(b2).add(new THREE.Vector3(0, -0.95, 0));
      scene.add(labelNoUp);
      labelNoUp.visible = false;

      const presets = [
        {
          pos: hp.clone().add(new THREE.Vector3(2, 2, 2)),
          lookAt: hp,
          show: () => {
            gizmo.visible = true;
            bezier.visible = false;
            snitchPath.visible = false;
            redCross.visible = false;
            redCrossBigNo1.visible = false;
            redCrossBigNo2.visible = false;
            tangentCircle.visible = false;
            forwardArrowBigNo1.visible = false;
            forwardArrowBigNo2.visible = false;
            labelX.visible = true;
            labelY.visible = true;
            labelZ.visible = true;
            labelTheta.visible = true;
            labelPhi.visible = true;
            labelGatePass.visible = false;
            labelCatchSnitch.visible = false;
            labelNoCurvature.visible = false;
            labelNoDown.visible = false;
            labelNoUp.visible = false;
          },
        },
        {
          pos: sl.clone().lerp(r1, 0.5).add(new THREE.Vector3(-10, 9, 5)),
          lookAt: sl.clone().lerp(r1, 0.5),
          show: () => {
            gizmo.visible = false;
            bezier.visible = true;
            snitchPath.visible = false;
            redCross.visible = false;
            redCrossBigNo1.visible = false;
            redCrossBigNo2.visible = false;
            tangentCircle.visible = false;
            forwardArrowBigNo1.visible = false;
            forwardArrowBigNo2.visible = false;
            labelX.visible = false;
            labelY.visible = false;
            labelZ.visible = false;
            labelTheta.visible = false;
            labelPhi.visible = false;
            labelGatePass.visible = true;
            labelCatchSnitch.visible = false;
            labelNoCurvature.visible = false;
            labelNoDown.visible = false;
            labelNoUp.visible = false;
          },
        },
        {
          pos: hp.clone().add(sn.clone()).multiplyScalar(0.5).add(new THREE.Vector3(3, 2, 2)),
          lookAt: sn.clone().add(hp).multiplyScalar(0.5),
          show: () => {
            gizmo.visible = false;
            bezier.visible = false;
            snitchPath.visible = true;
            redCross.visible = false;
            redCrossBigNo1.visible = false;
            redCrossBigNo2.visible = false;
            tangentCircle.visible = false;
            forwardArrowBigNo1.visible = false;
            forwardArrowBigNo2.visible = false;
            labelX.visible = false;
            labelY.visible = false;
            labelZ.visible = false;
            labelTheta.visible = false;
            labelPhi.visible = false;
            labelGatePass.visible = false;
            labelCatchSnitch.visible = true;
            labelNoCurvature.visible = false;
            labelNoDown.visible = false;
            labelNoUp.visible = false;
          },
        },
        {
          pos: cu.clone().add(new THREE.Vector3(0, 2, 4)),
          lookAt: cu,
          show: () => {
            gizmo.visible = false;
            bezier.visible = false;
            snitchPath.visible = false;
            redCross.visible = true;
            redCrossBigNo1.visible = false;
            redCrossBigNo2.visible = false;
            tangentCircle.visible = true;
            forwardArrowBigNo1.visible = false;
            forwardArrowBigNo2.visible = false;
            labelX.visible = false;
            labelY.visible = false;
            labelZ.visible = false;
            labelTheta.visible = false;
            labelPhi.visible = false;
            labelGatePass.visible = false;
            labelCatchSnitch.visible = false;
            labelNoCurvature.visible = true;
            labelNoDown.visible = false;
            labelNoUp.visible = false;
          },
        },
        {
          pos: b1.clone().add(new THREE.Vector3(0, 2, 4)),
          lookAt: b1,
          show: () => {
            gizmo.visible = false;
            bezier.visible = false;
            snitchPath.visible = false;
            redCross.visible = false;
            redCrossBigNo1.visible = true;
            redCrossBigNo2.visible = false;
            tangentCircle.visible = false;
            forwardArrowBigNo1.visible = true;
            forwardArrowBigNo2.visible = false;
            labelX.visible = false;
            labelY.visible = false;
            labelZ.visible = false;
            labelTheta.visible = false;
            labelPhi.visible = false;
            labelGatePass.visible = false;
            labelCatchSnitch.visible = false;
            labelNoCurvature.visible = false;
            labelNoDown.visible = true;
            labelNoUp.visible = false;
          },
        },
        {
          pos: b2.clone().add(new THREE.Vector3(0, 2, 4)),
          lookAt: b2,
          show: () => {
            gizmo.visible = false;
            bezier.visible = false;
            snitchPath.visible = false;
            redCross.visible = false;
            redCrossBigNo1.visible = false;
            redCrossBigNo2.visible = true;
            tangentCircle.visible = false;
            forwardArrowBigNo1.visible = false;
            forwardArrowBigNo2.visible = true;
            labelX.visible = false;
            labelY.visible = false;
            labelZ.visible = false;
            labelTheta.visible = false;
            labelPhi.visible = false;
            labelGatePass.visible = false;
            labelCatchSnitch.visible = false;
            labelNoCurvature.visible = false;
            labelNoDown.visible = false;
            labelNoUp.visible = true;
          },
        },
      ];

      let currentPreset = 0;

      function applyPreset(idx) {
        currentPreset = ((idx % PRESET_COUNT) + PRESET_COUNT) % PRESET_COUNT;
        const p = presets[currentPreset];
        baseCameraPos.copy(p.pos);
        baseLookAt.copy(p.lookAt);
        camera.position.copy(p.pos);
        camera.lookAt(p.lookAt);
        p.show();
      }

      applyPreset(0);
      parallaxReady = true;

      window.quiddichApplyPreset = (delta) => {
        applyPreset(currentPreset + delta);
      };

      const prevBtn = document.getElementById(containerId + "-prev");
      const nextBtn = document.getElementById(containerId + "-next");
      if (prevBtn) {
        prevBtn.addEventListener("click", () => applyPreset(currentPreset - 1));
        prevBtn.style.display = "none";
      }
      if (nextBtn) {
        nextBtn.addEventListener("click", () => applyPreset(currentPreset + 1));
        nextBtn.style.display = "none";
      }

      const overlayEl = document.createElement("div");
      overlayEl.style.cssText = "position:absolute;inset:0;pointer-events:none;z-index:1;display:flex;align-items:center;justify-content:space-between;padding:0 2%;box-sizing:border-box;";
      const size = 72;
      const leftBtn = document.createElement("div");
      leftBtn.style.cssText = `width:${size}px;height:${size}px;pointer-events:auto;cursor:pointer;flex-shrink:0;`;
      leftBtn.setAttribute("aria-label", "Previous");
      const rightBtn = document.createElement("div");
      rightBtn.style.cssText = `width:${size}px;height:${size}px;pointer-events:auto;cursor:pointer;flex-shrink:0;`;
      rightBtn.setAttribute("aria-label", "Next");
      const svgLeft = `<svg viewBox="0 0 100 100" width="100%" height="100%"><defs><mask id="qv-left-mask"><circle cx="50" cy="50" r="46" fill="white"/><path d="M50 22 L50 78 M28 50 L48 38 L48 62 Z" fill="black"/></mask></defs><circle cx="50" cy="50" r="46" fill="rgba(255, 0, 0, 0.4)" mask="url(#qv-left-mask)"/></svg>`;
      const svgRight = `<svg viewBox="0 0 100 100" width="100%" height="100%"><defs><mask id="qv-right-mask"><circle cx="50" cy="50" r="46" fill="white"/><path d="M50 22 L50 78 M72 50 L52 38 L52 62 Z" fill="black"/></mask></defs><circle cx="50" cy="50" r="46" fill="rgba(255, 0, 0, 0.4)" mask="url(#qv-right-mask)"/></svg>`;
      leftBtn.innerHTML = svgLeft;
      rightBtn.innerHTML = svgRight;
      leftBtn.addEventListener("click", () => applyPreset(currentPreset - 1));
      rightBtn.addEventListener("click", () => applyPreset(currentPreset + 1));
      overlayEl.appendChild(leftBtn);
      overlayEl.appendChild(rightBtn);
      container.appendChild(overlayEl);
    },
    undefined,
    (err) => {
      console.error("GLB load error", err);
      container.innerHTML = "<p style=\"color:#aaa;padding:1rem;\">3D viewer not available here (e.g. Cursor/VS Code). Open this notebook in Jupyter in your browser to see the scene.</p>";
    }
  );

  function animate() {
    requestAnimationFrame(animate);
    if (parallaxReady) {
      targetParallaxOffset.set(
        targetMouseNorm.x * PARALLAX_AMOUNT,
        targetMouseNorm.y * PARALLAX_AMOUNT,
        0
      );
      mouseOffset.lerp(targetParallaxOffset, PARALLAX_LERP);
      camera.position.copy(baseCameraPos).add(mouseOffset);
      camera.lookAt(baseLookAt);
    }
    renderer.render(scene, camera);
    css2DRenderer.render(scene, camera);
  }
  animate();

  window.addEventListener("resize", () => {
    const w = container.clientWidth || 800;
    const h = container.clientHeight || 400;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    css2DRenderer.setSize(w, h);
    scene.traverse((o) => {
      if (o instanceof Line2 && o.material && o.material.resolution) {
        o.material.resolution.set(w, h);
      }
    });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", run);
} else {
  run();
}
