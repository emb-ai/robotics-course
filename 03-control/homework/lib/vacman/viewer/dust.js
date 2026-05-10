/**
 * Floor dust grid (canvas texture) and vacuum cleaning radius.
 */
import * as THREE from "https://esm.sh/three@0.165.0";
import { BASE_VISUAL_R, CLEAN_R } from "./constants.js";

const DUST_SPACING = 0.25;

export function createDustSystem({
  scene,
  ARENA_X,
  ARENA_Z,
  halfX,
  halfZ,
  basePos,
  wallRects,
  cellFreeFromWalls,
}) {
  const gridW = Math.max(1, Math.ceil(ARENA_X / DUST_SPACING));
  const gridH = Math.max(1, Math.ceil(ARENA_Z / DUST_SPACING));
  const cellSizeX = ARENA_X / gridW;
  const cellSizeZ = ARENA_Z / gridH;
  const cellSizeMin = Math.min(cellSizeX, cellSizeZ);

  const dustCanvas = document.createElement("canvas");
  dustCanvas.width = gridW;
  dustCanvas.height = gridH;
  const dustCtx = dustCanvas.getContext("2d");
  const dustTex = new THREE.CanvasTexture(dustCanvas);
  dustTex.magFilter = THREE.NearestFilter;

  const dustOverlay = new THREE.Mesh(
    new THREE.PlaneGeometry(ARENA_X, ARENA_Z),
    new THREE.MeshBasicMaterial({ map: dustTex, transparent: true, depthWrite: false }),
  );
  dustOverlay.rotation.x = -Math.PI / 2;
  dustOverlay.position.y = 0.02;
  scene.add(dustOverlay);

  const dustGrid = new Uint8Array(gridW * gridH);
  let totalDust = 0;
  let cleaned = 0;

  const cellHash = new Uint8Array(gridW * gridH);
  for (let i = 0; i < cellHash.length; i++) cellHash[i] = ((i * 2654435761) >>> 0) & 0xff;

  let dustImageData = null;

  function paintDust() {
    if (!dustImageData) dustImageData = dustCtx.createImageData(gridW, gridH);
    const d = dustImageData.data;
    d.fill(0);
    for (let i = 0; i < gridW * gridH; i++) {
      if (dustGrid[i]) {
        const h = cellHash[i];
        const o = i * 4;
        d[o] = 140 + (h & 0x1f);
        d[o + 1] = 110 + ((h >> 3) & 0x1f);
        d[o + 2] = 70 + ((h >> 5) & 0x0f);
        d[o + 3] = 200;
      }
    }
    dustCtx.putImageData(dustImageData, 0, 0);
    dustTex.needsUpdate = true;
  }

  function fillDust() {
    dustGrid.fill(0);
    totalDust = 0;
    for (let gy = 0; gy < gridH; gy++) {
      for (let gx = 0; gx < gridW; gx++) {
        const wx = ((gx + 0.5) / gridW) * ARENA_X - halfX;
        const wz = ((gy + 0.5) / gridH) * ARENA_Z - halfZ;
        if (Math.hypot(wx - basePos.x, wz - basePos.z) < BASE_VISUAL_R + cellSizeMin * 0.5) continue;
        if (!cellFreeFromWalls(wx, wz, wallRects)) continue;
        const idx = gy * gridW + gx;
        dustGrid[idx] = 1;
        totalDust++;
      }
    }
    paintDust();
  }

  function cleanAt(wx, wz) {
    const cx = Math.floor(((wx + halfX) / ARENA_X) * gridW);
    const cy = Math.floor(((wz + halfZ) / ARENA_Z) * gridH);
    const rCells = Math.ceil(CLEAN_R / cellSizeMin) + 1;
    let hit = false;
    for (let dy = -rCells; dy <= rCells; dy++) {
      for (let dx = -rCells; dx <= rCells; dx++) {
        const px = cx + dx;
        const py = cy + dy;
        if (px < 0 || px >= gridW || py < 0 || py >= gridH) continue;
        const wcx = ((px + 0.5) / gridW) * ARENA_X - halfX;
        const wcz = ((py + 0.5) / gridH) * ARENA_Z - halfZ;
        if (Math.hypot(wcx - wx, wcz - wz) > CLEAN_R) continue;
        const idx = py * gridW + px;
        if (dustGrid[idx]) {
          dustGrid[idx] = 0;
          cleaned++;
          hit = true;
        }
      }
    }
    if (hit) paintDust();
  }

  function resetCleaned() {
    cleaned = 0;
  }

  return {
    fillDust,
    cleanAt,
    resetCleaned,
    get dustGrid() {
      return dustGrid;
    },
    get totalDust() {
      return totalDust;
    },
    get cleaned() {
      return cleaned;
    },
    gridSize: gridW,
    gridWidth: gridW,
    gridHeight: gridH,
    cellSizeX,
    cellSizeZ,
    halfX,
    halfZ,
    cellSizeMin,
  };
}
