import * as THREE from "https://esm.sh/three@0.165.0";

const CONE_RANGE = 4.8;
const CONE_HALF_ANGLE = 0.48;
const MASK_SIZE = 160;
const RAY_COUNT = 112;
const FLOOR_Y = 0.055;
const HIT_EPS = 0.01;

function cross(ax, az, bx, bz) {
  return ax * bz - az * bx;
}

function normalizeAngle(a) {
  let out = a;
  while (out <= -Math.PI) out += Math.PI * 2;
  while (out > Math.PI) out -= Math.PI * 2;
  return out;
}

function smoothstep(edge0, edge1, x) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function collectEdges(wallRects, halfX, halfZ) {
  const edges = [];
  for (const rect of wallRects) {
    for (let i = 0; i < rect.length; i++) {
      edges.push({ a: rect[i], b: rect[(i + 1) % rect.length] });
    }
  }

  const bounds = [
    { x: -halfX, z: -halfZ },
    { x: halfX, z: -halfZ },
    { x: halfX, z: halfZ },
    { x: -halfX, z: halfZ },
  ];
  for (let i = 0; i < bounds.length; i++) {
    edges.push({ a: bounds[i], b: bounds[(i + 1) % bounds.length] });
  }
  return edges;
}

function raySegmentDistance(origin, dir, a, b) {
  const sx = b.x - a.x;
  const sz = b.z - a.z;
  const den = cross(dir.x, dir.z, sx, sz);
  if (Math.abs(den) < 1e-7) return Infinity;

  const qx = a.x - origin.x;
  const qz = a.z - origin.z;
  const t = cross(qx, qz, sx, sz) / den;
  const u = cross(qx, qz, dir.x, dir.z) / den;
  if (t <= HIT_EPS || u < -1e-6 || u > 1 + 1e-6) return Infinity;
  return t;
}

function rayRange(origin, angle, edges) {
  const dir = { x: Math.cos(angle), z: Math.sin(angle) };
  let best = CONE_RANGE;
  for (const edge of edges) {
    const t = raySegmentDistance(origin, dir, edge.a, edge.b);
    if (t < best) best = t;
  }
  return Math.max(0, Math.min(CONE_RANGE, best));
}

function textureAngleToWorldAngle(textureAngle) {
  return -textureAngle;
}

/**
 * Red floor-projected headlight cone. A small offscreen 2D map mask clips the
 * shader texture against wall edges and arena bounds before Three.js renders it.
 */
export function createVacuumLightCone(scene, { wallRects, halfX, halfZ }) {
  const edges = collectEdges(wallRects, halfX, halfZ);
  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = MASK_SIZE;
  maskCanvas.height = MASK_SIZE;
  const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
  const maskImage = maskCtx.createImageData(MASK_SIZE, MASK_SIZE);
  const maskTex = new THREE.CanvasTexture(maskCanvas);
  maskTex.magFilter = THREE.LinearFilter;
  maskTex.minFilter = THREE.LinearFilter;
  maskTex.wrapS = THREE.ClampToEdgeWrapping;
  maskTex.wrapT = THREE.ClampToEdgeWrapping;

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uMask: { value: maskTex },
      uColor: { value: new THREE.Color(0xff3030) },
      uTime: { value: 0 },
    },
    transparent: true,
    depthTest: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform sampler2D uMask;
      uniform vec3 uColor;
      uniform float uTime;
      varying vec2 vUv;

      void main() {
        float alpha = texture2D(uMask, vUv).a;
        if (alpha < 0.01) discard;
        float pulse = 0.86 + 0.14 * sin(uTime * 5.0);
        gl_FragColor = vec4(uColor * (1.0 + 0.22 * pulse), alpha * 0.62 * pulse);
      }
    `,
  });

  const projector = new THREE.Mesh(
    new THREE.PlaneGeometry(CONE_RANGE * 2, CONE_RANGE * 2),
    material,
  );
  projector.rotation.x = -Math.PI / 2;
  projector.position.y = FLOOR_Y;
  projector.renderOrder = 4;
  scene.add(projector);

  const ranges = new Float32Array(RAY_COUNT);
  const diameter = CONE_RANGE * 2;

  function updateMask(agent, worldHeading) {
    const origin = { x: agent.x, z: agent.z };
    for (let i = 0; i < RAY_COUNT; i++) {
      const f = i / (RAY_COUNT - 1);
      const rel = -CONE_HALF_ANGLE + f * CONE_HALF_ANGLE * 2;
      ranges[i] = rayRange(origin, worldHeading + rel, edges);
    }

    const data = maskImage.data;
    for (let py = 0; py < MASK_SIZE; py++) {
      const v = (py + 0.5) / MASK_SIZE;
      const lz = (0.5 - v) * diameter;
      for (let px = 0; px < MASK_SIZE; px++) {
        const u = (px + 0.5) / MASK_SIZE;
        const lx = (u - 0.5) * diameter;
        const r = Math.hypot(lx, lz);
        const off = (py * MASK_SIZE + px) * 4;

        let alpha = 0;
        if (r <= CONE_RANGE && r > 0.08) {
          const textureAngle = Math.atan2(lz, lx);
          const worldAngle = textureAngleToWorldAngle(textureAngle);
          const rel = normalizeAngle(worldAngle - worldHeading);
          const absRel = Math.abs(rel);
          if (absRel <= CONE_HALF_ANGLE) {
            const sample = ((rel + CONE_HALF_ANGLE) / (CONE_HALF_ANGLE * 2)) * (RAY_COUNT - 1);
            const i0 = Math.max(0, Math.min(RAY_COUNT - 1, Math.floor(sample)));
            const i1 = Math.max(0, Math.min(RAY_COUNT - 1, i0 + 1));
            const f = sample - i0;
            const maxR = ranges[i0] * (1 - f) + ranges[i1] * f;
            const edgeFade = 1 - smoothstep(CONE_HALF_ANGLE * 0.74, CONE_HALF_ANGLE, absRel);
            const radialFade = 1 - smoothstep(0.12, 1.0, r / CONE_RANGE);
            const hitFade = 1 - smoothstep(maxR - 0.015, maxR + 0.005, r);
            const sourceFade = smoothstep(0.08, 0.42, r);
            alpha = Math.round(255 * edgeFade * radialFade * hitFade * sourceFade);
          }
        }

        data[off] = 255;
        data[off + 1] = 52;
        data[off + 2] = 48;
        data[off + 3] = alpha;
      }
    }
    maskCtx.putImageData(maskImage, 0, 0);
    maskTex.needsUpdate = true;
  }

  function update(agent) {
    updateMask(agent, agent.th);

    projector.position.set(agent.x, FLOOR_Y, agent.z);
    material.uniforms.uTime.value = performance.now() / 1000;
  }

  return { update };
}
