/**
 * OBJ loading and simple normalization for VacMan / CatMan meshes.
 */
import * as THREE from "https://esm.sh/three@0.165.0";
import { OBJLoader } from "https://esm.sh/three@0.165.0/examples/jsm/loaders/OBJLoader.js";
import { CAT_OCT_R, VAC_RADIUS } from "./constants.js";

const objLoader = new OBJLoader();

export function loadOBJ(url) {
  return new Promise((res, rej) => objLoader.load(url, res, undefined, rej));
}

export function fitModel(obj, targetSize) {
  const box = new THREE.Box3().setFromObject(obj);
  const centre = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const scale = targetSize / Math.max(size.x, size.y, size.z);
  obj.traverse((ch) => {
    if (!ch.isMesh) return;
    ch.geometry.translate(-centre.x, -centre.y + size.y / 2, -centre.z);
    ch.geometry.scale(scale, scale, scale);
    ch.material = new THREE.MeshStandardMaterial({
      vertexColors: !!ch.geometry.attributes.color,
      color: ch.geometry.attributes.color ? 0xffffff : 0xcccccc,
      roughness: 0.55,
      metalness: 0.05,
    });
    ch.castShadow = true;
  });
}

export function decorateVacuum(vacObj) {
  const ring = makeCollisionRing(VAC_RADIUS, 0x59d7ff);
  vacObj.add(ring);
  return ring;
}

export function addCatCollisionRing(catObj) {
  const ring = makeCollisionRing(CAT_OCT_R, 0xff5252);
  catObj.add(ring);
  return ring;
}

function makeCollisionRing(radius, color) {
  const ringMat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.68,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const ringGeo = new THREE.RingGeometry(radius * 0.92, radius, 48);
  ringGeo.rotateX(-Math.PI / 2);
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.position.y = 0.08;
  ring.renderOrder = 4;
  return ring;
}

export function assetUrls() {
  return {
    vacUrl: new URL("../../../assets/vacman.obj", import.meta.url).href,
    catUrl: new URL("../../../assets/catman.obj", import.meta.url).href,
  };
}
