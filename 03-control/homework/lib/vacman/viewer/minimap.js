/**
 * Lightweight overhead minimap with walls, dust, base, and agent positions.
 */
import { BASE_VISUAL_R } from "./constants.js";

export function createMinimap(container, { halfX, halfZ, wallRects, basePos, dust }) {
  const wrap = document.createElement("div");
  wrap.style.cssText = `position:absolute;right:18px;bottom:48px;width:170px;height:170px;
    background:rgba(0,0,0,0.84);border:1px solid rgba(61,255,140,0.35);border-radius:0;
    box-shadow:0 12px 28px rgba(0,0,0,0.32);padding:8px;z-index:12;`;
  container.appendChild(wrap);

  const title = document.createElement("div");
  title.textContent = "RADAR";
  title.style.cssText =
    "font:bold 12px/1.2 'Courier New', Courier, 'SF Mono', Consolas, monospace;color:#e8fff0;margin-bottom:6px;letter-spacing:0;";
  wrap.appendChild(title);

  const canvas = document.createElement("canvas");
  canvas.width = 150;
  canvas.height = 150;
  canvas.style.cssText = "display:block;width:150px;height:150px;border-radius:0;image-rendering:pixelated;";
  wrap.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  const worldW = halfX * 2;
  const worldH = halfZ * 2;
  const mapScale = Math.min(canvas.width / worldW, canvas.height / worldH);
  const mapW = worldW * mapScale;
  const mapH = worldH * mapScale;
  const mapX0 = (canvas.width - mapW) / 2;
  const mapY0 = (canvas.height - mapH) / 2;

  function toMapX(x) {
    return mapX0 + (halfX - x) * mapScale;
  }

  function toMapY(z) {
    return mapY0 + (halfZ - z) * mapScale;
  }

  function toMapRadius(r) {
    return Math.max(2, r * mapScale);
  }

  function drawMapBounds() {
    ctx.strokeStyle = "rgba(61,255,140,0.22)";
    ctx.lineWidth = 1;
    ctx.strokeRect(mapX0 + 0.5, mapY0 + 0.5, mapW - 1, mapH - 1);
  }

  function drawWalls() {
    ctx.fillStyle = "#2636b8";
    ctx.strokeStyle = "#6f86ff";
    ctx.lineWidth = 1;
    for (const rect of wallRects) {
      ctx.beginPath();
      ctx.moveTo(toMapX(rect[0].x), toMapY(rect[0].z));
      for (let i = 1; i < rect.length; i++) {
        ctx.lineTo(toMapX(rect[i].x), toMapY(rect[i].z));
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
  }

  function drawDust() {
    const grid = dust.dustGrid;
    const gridW = dust.gridWidth ?? dust.gridSize;
    const gridH = dust.gridHeight ?? dust.gridSize;
    if (!grid || !gridW || !gridH) return;
    const pxW = mapW / gridW;
    const pxH = mapH / gridH;
    const pelletR = Math.max(0.7, Math.min(pxW, pxH) * 0.18);
    ctx.fillStyle = "rgba(255, 210, 31, 0.92)";
    for (let gy = 0; gy < gridH; gy++) {
      for (let gx = 0; gx < gridW; gx++) {
        if (!grid[gy * gridW + gx]) continue;
        ctx.beginPath();
        ctx.arc(
          mapX0 + mapW - (gx + 0.5) * pxW,
          mapY0 + mapH - (gy + 0.5) * pxH,
          pelletR,
          0,
          Math.PI * 2,
        );
        ctx.fill();
      }
    }
  }

  function drawPacman(agent) {
    const x = toMapX(agent.x);
    const y = toMapY(agent.z);
    const radius = 5.5;
    const mouthHalf = Math.PI * 0.24;
    const heading = Math.PI + agent.th;
    ctx.fillStyle = "#ffd21f";
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.arc(x, y, radius, heading + mouthHalf, heading + Math.PI * 2 - mouthHalf);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#050505";
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.arc(x, y, radius + 0.2, heading - mouthHalf, heading + mouthHalf);
    ctx.closePath();
    ctx.fill();
  }

  function drawCatman(agent) {
    const x = toMapX(agent.x);
    const y = toMapY(agent.z);
    const s = 8;
    ctx.fillStyle = "#ff5252";
    ctx.fillRect(x - s / 2, y - s / 2, s, s);
    ctx.strokeStyle = "#ffd1d1";
    ctx.lineWidth = 1;
    ctx.strokeRect(x - s / 2 + 0.5, y - s / 2 + 0.5, s - 1, s - 1);
  }

  function render({ vac, cat, baseActive = true }) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#03070c";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    drawDust();
    drawWalls();
    drawMapBounds();

    ctx.strokeStyle = baseActive ? "#5af2a9" : "rgba(90, 242, 169, 0.28)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(toMapX(basePos.x), toMapY(basePos.z), toMapRadius(BASE_VISUAL_R), 0, Math.PI * 2);
    ctx.stroke();

    drawCatman(cat);
    drawPacman(vac);
  }

  return { render };
}
