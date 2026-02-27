from __future__ import annotations

import base64
import html
from typing import Literal

JointType = Literal["R", "P", "H", "C", "U", "S"]

THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"

COMMON_JS = r"""
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);
    const camera = new THREE.PerspectiveCamera(50, WIDTH / HEIGHT, 0.1, 100);
    camera.position.set(1, 1, 1);
    camera.lookAt(0, 0, 0.2);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(WIDTH, HEIGHT);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    const target = new THREE.Vector3(0, 0, 0.2);
    let azimuth = 0.7, elevation = 0.7, radius = 1.73;
    let drag = false, prevX = 0, prevY = 0;
    renderer.domElement.addEventListener('pointerdown', e => { drag = true; prevX = e.clientX; prevY = e.clientY; });
    renderer.domElement.addEventListener('pointerup', () => drag = false);
    renderer.domElement.addEventListener('pointerleave', () => drag = false);
    renderer.domElement.addEventListener('pointermove', e => {
      if (!drag) return;
      azimuth -= (e.clientX - prevX) * 0.005;
      elevation = Math.max(0.1, Math.min(Math.PI - 0.1, elevation + (e.clientY - prevY) * 0.005));
      prevX = e.clientX; prevY = e.clientY;
    });
    function updateControls() {
      camera.position.set(
        target.x + radius * Math.sin(elevation) * Math.cos(azimuth),
        target.y + radius * Math.cos(elevation),
        target.z + radius * Math.sin(elevation) * Math.sin(azimuth)
      );
      camera.lookAt(target);
    }
    radius = 1.62; azimuth = 0.67; elevation = 0.9;
    const ambient = new THREE.AmbientLight(0x404040);
    scene.add(ambient);
    const light = new THREE.DirectionalLight(0xffffff, 0.9);
    light.position.set(2, 2, 2);
    scene.add(light);
"""


def _script_revolute() -> str:
    return r"""
    const boxW = 0.14, boxL = 0.4;
    const baseGeo = new THREE.BoxGeometry(boxW, boxW, boxL);
    const baseMat = new THREE.MeshBasicMaterial({ color: 0x607080 });
    const base = new THREE.Mesh(baseGeo, baseMat);
    base.position.z = -boxL / 2;
    scene.add(base);
    const cylGeo = new THREE.CylinderGeometry(0.06, 0.06, boxW * 1.2, 16);
    const cylMat = new THREE.MeshBasicMaterial({ color: 0x607080 });
    const cylinder = new THREE.Mesh(cylGeo, cylMat);
    cylinder.rotation.z = -Math.PI / 2;
    scene.add(cylinder);
    const armGeo = new THREE.BoxGeometry(boxW, boxW, boxL);
    const armMat = new THREE.MeshBasicMaterial({ color: 0xff5533 });
    const arm = new THREE.Mesh(armGeo, armMat);
    const armPivot = new THREE.Group();
    arm.position.z = boxL / 2;
    armPivot.add(arm);
    scene.add(armPivot);
    const start = performance.now() / 1000;
    const amp = Math.PI / 4;
    function animate() {
      requestAnimationFrame(animate);
      const t = (performance.now() / 1000 - start) % 4;
      armPivot.rotation.x = amp * Math.sin((t / 4) * Math.PI * 2);
      updateControls();
      renderer.render(scene, camera);
    }
    animate();
"""


def _script_prismatic() -> str:
    return r"""
    const baseW = 0.2, baseL = 0.5;
    const armW = 0.14, armL = 0.35;
    const baseGeo = new THREE.BoxGeometry(baseW, baseW, baseL);
    const baseMat = new THREE.MeshBasicMaterial({ color: 0x607080 });
    const base = new THREE.Mesh(baseGeo, baseMat);
    scene.add(base);
    const armGeo = new THREE.BoxGeometry(armW, armW, armL);
    const armMat = new THREE.MeshBasicMaterial({ color: 0xdc143c });
    const arm = new THREE.Mesh(armGeo, armMat);
    scene.add(arm);
    const slideAmp = armL / 4;
    const start = performance.now() / 1000;
    function animate() {
      requestAnimationFrame(animate);
      const t = (performance.now() / 1000 - start) % 4;
      arm.position.z = slideAmp * Math.sin((t / 4) * Math.PI * 2) + 3 * armL / 4;
      updateControls();
      renderer.render(scene, camera);
    }
    animate();
"""


def _script_helical() -> str:
    return r"""
    function helicoidGeometry(radius, length, turns, radialSegs, angularSegs) {
      const zScale = length / (turns * Math.PI * 2);
      const z0 = -length / 2;
      const verts = [];
      for (let i = 0; i <= radialSegs; i++) {
        const u = (i / radialSegs) * radius;
        for (let j = 0; j <= angularSegs; j++) {
          const v = (j / angularSegs) * turns * Math.PI * 2;
          verts.push(u * Math.cos(v), u * Math.sin(v), z0 + zScale * v);
        }
      }
      const indices = [];
      for (let i = 0; i < radialSegs; i++)
        for (let j = 0; j < angularSegs; j++) {
          const a = i * (angularSegs + 1) + j, b = a + 1, c = a + (angularSegs + 1), d = c + 1;
          indices.push(a, c, b, b, c, d);
        }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
      geo.setIndex(indices);
      geo.computeVertexNormals();
      return geo;
    }
    const cylR = 0.08, cylL = 0.5, turns = 4;
    const baseGeo = helicoidGeometry(cylR, cylL, turns, 6, 48);
    const baseMat = new THREE.MeshBasicMaterial({ color: 0x607080 });
    const base = new THREE.Mesh(baseGeo, baseMat);
    scene.add(base);
    const armW = 0.12, armL = 0.35;
    const armGeo = new THREE.BoxGeometry(armW, armL, armW);
    const armMat = new THREE.MeshBasicMaterial({ color: 0xe040fb });
    const arm = new THREE.Mesh(armGeo, armMat);
    arm.position.y = armL / 2;
    const armPivot = new THREE.Group();
    armPivot.add(arm);
    scene.add(armPivot);
    const zMin = -cylL / 2, zMax = cylL / 2;
    const start = performance.now() / 1000;
    function animate() {
      requestAnimationFrame(animate);
      const t = (performance.now() / 1000 - start) % 4;
      const phase = (1 + Math.sin((t / 4) * Math.PI * 2)) / 2;
      armPivot.position.z = zMin + phase * cylL;
      armPivot.rotation.z = phase * turns * Math.PI * 2;
      updateControls();
      renderer.render(scene, camera);
    }
    animate();
"""


def _script_cylindrical() -> str:
    return r"""
    const cylR = 0.1, cylL = 0.9;
    const baseGeo = new THREE.CylinderGeometry(cylR, cylR, cylL, 24);
    const baseMat = new THREE.MeshBasicMaterial({ color: 0x607080 });
    const base = new THREE.Mesh(baseGeo, baseMat);
    base.rotation.x = Math.PI / 2;
    scene.add(base);
    const armLen = 0.56, armW = 0.1;
    const armGeo = new THREE.BoxGeometry(armLen, armW, armW);
    const armMat = new THREE.MeshBasicMaterial({ color: 0xf9a825 });
    const arm = new THREE.Mesh(armGeo, armMat);
    arm.position.set(armLen / 2, 0, 0);
    const armPivot = new THREE.Group();
    armPivot.add(arm);
    scene.add(armPivot);
    const zCenter = 0.0, zAmp = 0.4 * cylL;
    const start = performance.now() / 1000;
    function animate() {
      requestAnimationFrame(animate);
      const t = (performance.now() / 1000 - start) % 4;
      const theta = (t / 4) * Math.PI * 2;
      const d = zAmp * Math.sin(theta);
      armPivot.rotation.z = theta;
      armPivot.position.set(0, 0, zCenter + d);
      updateControls();
      renderer.render(scene, camera);
    }
    animate();
"""


def _script_universal() -> str:
    return r"""
    function halfTorus(radius, tubeRadius, color) {
      const arc = Math.PI;
      const tubularSegments = 48;
      const radialSegments = 16;
      let geometry = new THREE.TorusGeometry(radius, tubeRadius, radialSegments, tubularSegments, arc);
      const mat = new THREE.MeshBasicMaterial({ color: color });
      const mesh = new THREE.Mesh(geometry, mat);
      const group = new THREE.Group();
      group.add(mesh);
      return group;
    }
    function closedHalfTorusDShape(torusRadius, tubeRadius, color) {
      const torusGroup = halfTorus(torusRadius, tubeRadius, color);
      const cylGeo = new THREE.CylinderGeometry(tubeRadius, tubeRadius, 2 * torusRadius, 24);
      const cylMat = new THREE.MeshBasicMaterial({ color: 0x808080 });
      const cylinder = new THREE.Mesh(cylGeo, cylMat);
      const capGeo = new THREE.SphereGeometry(tubeRadius, 24, 16);
      const capMat = new THREE.MeshBasicMaterial({ color: 0x808080 });
      const cap1 = new THREE.Mesh(capGeo, capMat);
      const cap2 = new THREE.Mesh(capGeo, capMat);
      cap1.position.set(torusRadius, 0, 0);
      cap2.position.set(-torusRadius, 0, 0);
      cylinder.position.set(0, 0, 0);
      cylinder.rotation.z = Math.PI / 2;
      const group = new THREE.Group();
      group.add(torusGroup);
      group.add(cylinder);
      group.add(cap1);
      group.add(cap2);
      return group;
    }
    function closedHalfTorusDShapeWithArm(torusRadius, tubeRadius, armLen, color) {
      const base = closedHalfTorusDShape(torusRadius, tubeRadius, color);
      const armGeo = new THREE.CylinderGeometry(tubeRadius, tubeRadius, armLen, 24);
      const armMat = new THREE.MeshBasicMaterial({ color: color });
      const arm = new THREE.Mesh(armGeo, armMat);
      arm.position.y = 2 * torusRadius;
      const group = new THREE.Group();
      group.add(base);
      group.add(arm);
      group.arm = arm;  // Store for further animation access
      return group;
    }

    const ringR = 0.12, ringTube = 0.048;
    const armLen = 0.26, shaftR = ringTube * 1.05;
    const baseColor = 0x607080, armColor = 0xff4081;

    let base, arm;
    const baseQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, -Math.PI / 2, Math.PI, 'XYZ'));
    const hingeX = new THREE.Vector3(1, 0, 0);
    const hingeZ = new THREE.Vector3(0, 0, 1);
    const qHinge1 = new THREE.Quaternion();
    const qHinge2 = new THREE.Quaternion();

    function initUniversalJoint() {
      base = closedHalfTorusDShapeWithArm(ringR, ringTube, armLen, baseColor);
      scene.add(base);

      arm = closedHalfTorusDShapeWithArm(ringR, ringTube, armLen, armColor);
      arm.quaternion.copy(baseQuat);
      arm.scale.set(-1, -1, -1);
      scene.add(arm);
    }

    const start = performance.now() / 1000;

    function animate() {
      requestAnimationFrame(animate);
      const t = (performance.now() / 1000 - start) % 4;

      if (!base || !arm) return;

      let upDown = 0, leftRight = Math.PI;
      if (t < 2) {
        upDown = Math.sin((t / 2) * Math.PI * 2) * (Math.PI / 5);
      } else {
        leftRight = Math.sin(((t - 2) / 2) * Math.PI * 2) * (Math.PI / 6) + Math.PI;
      }

      qHinge1.setFromAxisAngle(hingeX, upDown);
      qHinge2.setFromAxisAngle(hingeZ, leftRight);
      arm.quaternion.copy(baseQuat).multiply(qHinge1).multiply(qHinge2);

      updateControls();
      renderer.render(scene, camera);
    }

    initUniversalJoint();
    animate();
"""


def _script_spherical() -> str:
    return r"""
    const baseR = 0.14, baseLen = 0.28;
    const ballR = 0.07;
    const armW = 0.06, armLen = 0.38;
    // Single base+socket cylinder
    const baseGeo = new THREE.CylinderGeometry(baseR, baseR, baseLen, 32);
    const base = new THREE.Mesh(baseGeo, new THREE.MeshBasicMaterial({ color: 0x607080 }));
    base.rotation.z = Math.PI / 2;
    // Move the base farther left so the ball at the origin is visible
    base.position.x = -(baseLen / 2 + ballR/4);
    base.position.z = (baseR / 2);
    scene.add(base);

    const group = new THREE.Group();
    group.position.z = ballR;
    group.rotation.order = 'ZYX';
    const sphereGeo = new THREE.SphereGeometry(ballR, 24, 20);
    const sphere = new THREE.Mesh(sphereGeo, new THREE.MeshBasicMaterial({ color: 0x69f0ae }));
    group.add(sphere);
    const rotatable = new THREE.Group();
    const armGeo = new THREE.BoxGeometry(armW, armW, armLen);
    const arm = new THREE.Mesh(armGeo, new THREE.MeshBasicMaterial({ color: 0x69f0ae }));
    arm.rotation.y = -Math.PI / 2;
    arm.position.x = ballR + armLen / 2;
    rotatable.add(arm);
    group.add(rotatable);
    scene.add(group);

    // maxConeAngle < PI/2 guarantees the arm stays in the +X hemisphere,
    // never swinging back toward the base which lives along -Z.
    const maxConeAngle = Math.PI / 3;
    const start = performance.now() / 1000;
    function animate() {
      requestAnimationFrame(animate);
      const t = (performance.now() / 1000 - start);

      // DOF 1 – polar swing: angle of arm from its rest direction (+X), 0..maxConeAngle
      const theta = maxConeAngle * 0.5 * (1 + Math.sin(t * 0.7));
      // DOF 2 – azimuthal sweep around +X axis
      const phi = t * 0.5;
      // DOF 3 – twist: spin around the arm's own axis
      const twist = t * 1.5;

      // Rotate (1,0,0) toward (theta,phi): axis perpendicular to +X in the YZ plane at angle phi
      const swingAxis = new THREE.Vector3(0, -Math.sin(phi), Math.cos(phi));
      group.quaternion.setFromAxisAngle(swingAxis, theta);

      // Twist is applied in the arm's local frame (child of group), so it stays aligned
      rotatable.rotation.x = twist;

      updateControls();
      renderer.render(scene, camera);
    }
    animate();
"""


_SCRIPTS: dict[JointType, str] = {
    "R": _script_revolute(),
    "P": _script_prismatic(),
    "H": _script_helical(),
    "C": _script_cylindrical(),
    "U": _script_universal(),
    "S": _script_spherical(),
}

_TITLES: dict[JointType, str] = {
    "R": "Revolute (R): 1 DOF",
    "P": "Prismatic (P): 1 DOF",
    "H": "Helical (H): 1 DOF",
    "C": "Cylindrical (C): 2 DOF",
    "U": "Universal (U): 2 DOF",
    "S": "Spherical (S): 3 DOF",
}


def generate_joint_html(
    joint_type: JointType,
    width: int = 360,
    height: int = 280,
) -> str:
    if joint_type not in _SCRIPTS:
        raise ValueError(f"joint_type must be one of {list(_SCRIPTS.keys())}")
    title = _TITLES[joint_type]
    script = COMMON_JS + _SCRIPTS[joint_type]
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
</head>
<body style="margin:0;">
  <div id="container"></div>
  <p style="margin:4px 8px;font:12px sans-serif;color:#888;">{html.escape(title)}</p>
  <script src="{THREE_CDN}"></script>
  <script>
    const container = document.getElementById('container');
    const WIDTH = {width};
    const HEIGHT = {height};
    {script}
  </script>
</body>
</html>"""


def _iframe_src_from_html(html_str: str) -> str:
    b64 = base64.b64encode(html_str.encode("utf-8")).decode("ascii")
    return f"data:text/html;charset=utf-8;base64,{b64}"


def embed_joint_viz(
    joint_type: JointType,
    width: int = 360,
    height: int = 280,
):
    from IPython.display import HTML

    raw = generate_joint_html(joint_type, width=width, height=height)
    src = _iframe_src_from_html(raw)
    iframe = f'<iframe src="{html.escape(src)}" width="{width}" height="{height + 24}" frameborder="0" sandbox="allow-scripts"></iframe>'
    return HTML(iframe)


def embed_all_joints(width: int = 360, height: int = 280):
    from IPython.display import HTML

    parts = []
    for j in ["R", "P", "H", "C", "U", "S"]:
        raw = generate_joint_html(j, width=width, height=height)
        src = _iframe_src_from_html(raw)
        parts.append(
            f'<iframe src="{html.escape(src)}" width="{width}" height="{height + 24}" frameborder="0" sandbox="allow-scripts"></iframe>'
        )
    grid = (
        f'<div style="display:grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">'
        + "".join(parts)
        + "</div>"
    )
    return HTML(grid)
