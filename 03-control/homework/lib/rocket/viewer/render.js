import {
  bodyToWorldOffset,
  DEFAULT_WIND,
  gimbalLocalToWorldOffset,
  PARAMS,
  ROCKET,
  windSurfaceFactor,
} from "./dynamics.js";

function randomRange(min, max) {
  return min + Math.random() * (max - min);
}

export function createSceneElements(starCount = 780, flowerCount = 260) {
  const stars = Array.from({ length: starCount }, () => ({
    x: randomRange(-2200, 2200),
    z: randomRange(4, 2100),
    r: randomRange(0.9, 3.4),
    phase: randomRange(0, Math.PI * 2),
    tint: Math.random(),
    sparkle: Math.random() < 0.34,
  }));

  const flowers = Array.from({ length: flowerCount }, (_, i) => ({
    x: -190 + i * (380 / flowerCount) + randomRange(-1.6, 1.6),
    z: randomRange(-42, -0.8),
    radius: randomRange(0.055, 0.095),
    petals: Math.random() < 0.55 ? 5 : 6,
    phase: randomRange(0, Math.PI * 2),
    color: Math.random() < 0.5 ? "#ffdf64" : Math.random() < 0.5 ? "#ff8fb7" : "#c8a4ff",
  }));

  const windStreams = Array.from({ length: 42 }, () => createWindStreamSeed());

  return { stars, flowers, windStreams };
}

export function resizeCanvasToDisplaySize(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

export function worldToScreen(wx, wz, canvas, camera, metersH) {
  const ppm = canvas.height / metersH;
  const sx = canvas.width / 2 + (wx - camera.x) * ppm;
  const sy = canvas.height / 2 - (wz - camera.z) * ppm;
  return [sx, sy];
}

function screenYToWorldZ(sy, canvas, camera, metersH) {
  const ppm = canvas.height / metersH;
  return camera.z + (canvas.height / 2 - sy) / ppm;
}

function mapBodyPolyToScreen(pts, theta, xW, zW, canvas, camera, metersH) {
  return pts.map(([bx, bz]) => {
    const [dx, dz] = bodyToWorldOffset(bx, bz, theta);
    return worldToScreen(xW + dx, zW + dz, canvas, camera, metersH);
  });
}

function mapGimbalPolyToScreen(pts, delta, theta, xW, zW, canvas, camera, metersH) {
  return pts.map(([lx, lz]) => {
    const [dx, dz] = gimbalLocalToWorldOffset(lx, lz, delta, theta);
    return worldToScreen(xW + dx, zW + dz, canvas, camera, metersH);
  });
}

function drawLandingTarget(ctx, canvas, camera, metersH, groundY, timeSeconds, target, showGoalGlow = true) {
  const ppm = canvas.height / metersH;
  const padW = target.width ?? 3.2;
  const padH = target.height ?? 0.35;
  const padX = target.x ?? 0;
  const baseW = padW * 1.32;
  const topW = padW;
  const baseLeft = worldToScreen(padX - baseW / 2, 0, canvas, camera, metersH);
  const baseRight = worldToScreen(padX + baseW / 2, 0, canvas, camera, metersH);
  const topRight = worldToScreen(padX + topW / 2, padH, canvas, camera, metersH);
  const topLeft = worldToScreen(padX - topW / 2, padH, canvas, camera, metersH);
  const [centerX] = worldToScreen(padX, 0, canvas, camera, metersH);
  const padPx = padW * ppm;
  const pulse = 0.82 + 0.18 * Math.sin(timeSeconds * 3.5);
  const beamH = Math.max(42, padPx * 2.1);

  if (showGoalGlow) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const beam = ctx.createLinearGradient(0, groundY - beamH, 0, groundY);
    beam.addColorStop(0.00, "rgba(61,255,140,0.00)");
    beam.addColorStop(0.42, `rgba(61,255,140,${0.10 * pulse})`);
    beam.addColorStop(1.00, `rgba(61,255,140,${0.42 * pulse})`);
    ctx.fillStyle = beam;
    ctx.fillRect(centerX - padPx * 0.68, groundY - beamH, padPx * 1.36, beamH);
    ctx.restore();
  }

  fillScreenPoly(ctx, [baseLeft, baseRight, topRight, topLeft], "#3a3d3b");

  ctx.strokeStyle = showGoalGlow ? `rgba(61,255,140,${0.45 + 0.25 * pulse})` : "rgba(232,255,240,0.50)";
  ctx.lineWidth = Math.max(2, ppm * 0.05);
  ctx.beginPath();
  ctx.moveTo(baseLeft[0], baseLeft[1]);
  ctx.lineTo(baseRight[0], baseRight[1]);
  ctx.lineTo(topRight[0], topRight[1]);
  ctx.lineTo(topLeft[0], topLeft[1]);
  ctx.closePath();
  ctx.stroke();

  const stripeY = topLeft[1] + (baseLeft[1] - topLeft[1]) * 0.52;
  ctx.strokeStyle = `rgba(232,255,240,${0.65 + 0.2 * pulse})`;
  ctx.lineWidth = Math.max(1, ppm * 0.035);
  ctx.beginPath();
  ctx.moveTo(centerX - padPx * 0.32, stripeY);
  ctx.lineTo(centerX + padPx * 0.32, stripeY);
  ctx.stroke();

  const poleBase = topRight;
  const poleTop = [poleBase[0], poleBase[1] - Math.max(12, 1.3 * ppm)];
  strokeScreenSeg(ctx, poleBase, poleTop, "#d8d8d8", Math.max(1.5, 0.035 * ppm));
  ctx.fillStyle = "#ff3030";
  ctx.beginPath();
  ctx.moveTo(poleTop[0], poleTop[1]);
  ctx.lineTo(poleTop[0] + Math.max(13, 0.55 * ppm), poleTop[1] + Math.max(5, 0.22 * ppm));
  ctx.lineTo(poleTop[0], poleTop[1] + Math.max(10, 0.44 * ppm));
  ctx.closePath();
  ctx.fill();
}

function drawAscentTarget(ctx, canvas, camera, metersH, timeSeconds, target) {
  const ppm = canvas.height / metersH;
  const [, levelY] = worldToScreen(target.x, target.z, canvas, camera, metersH);
  const pulse = 0.82 + 0.18 * Math.sin(timeSeconds * 3.5);
  const bandH = Math.max(18, (target.zTolerance ?? 4) * 2 * ppm);
  const theta = target.theta ?? 0;
  const dirX = Math.sin(theta);
  const dirY = -Math.cos(theta);
  const normalX = Math.cos(theta);
  const normalY = Math.sin(theta);

  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  const band = ctx.createLinearGradient(0, levelY - bandH / 2, 0, levelY + bandH / 2);
  band.addColorStop(0.00, "rgba(61,255,140,0)");
  band.addColorStop(0.48, `rgba(61,255,140,${0.16 * pulse})`);
  band.addColorStop(0.52, `rgba(190,255,220,${0.22 * pulse})`);
  band.addColorStop(1.00, "rgba(61,255,140,0)");
  ctx.fillStyle = band;
  ctx.fillRect(0, levelY - bandH / 2, canvas.width, bandH);

  ctx.strokeStyle = `rgba(190,255,220,${0.70 + 0.20 * pulse})`;
  ctx.lineWidth = Math.max(2, ppm * 0.05);
  ctx.beginPath();
  ctx.moveTo(0, levelY);
  ctx.lineTo(canvas.width, levelY);
  ctx.stroke();

  ctx.fillStyle = `rgba(61,255,140,${0.18 + 0.18 * pulse})`;
  const dashW = Math.max(26, 1.4 * ppm);
  const dashGap = Math.max(18, 0.8 * ppm);
  const offset = (timeSeconds * 28) % (dashW + dashGap);
  for (let x = -offset; x < canvas.width; x += dashW + dashGap) {
    ctx.fillRect(x, levelY - Math.max(1, 0.018 * ppm), dashW, Math.max(2, 0.036 * ppm));
  }

  const arrowStep = Math.max(90, 4.5 * ppm);
  const arrowSize = Math.max(13, 0.55 * ppm);
  const arrowOffset = (timeSeconds * 34) % arrowStep;
  ctx.shadowColor = "#3dff8c";
  ctx.shadowBlur = Math.max(8, 0.24 * ppm);
  for (let x = -arrowOffset; x < canvas.width + arrowStep; x += arrowStep) {
    const tipX = x + dirX * arrowSize;
    const tipY = levelY + dirY * arrowSize;
    const backX = x - dirX * arrowSize * 0.62;
    const backY = levelY - dirY * arrowSize * 0.62;
    const side = arrowSize * 0.45;

    ctx.fillStyle = `rgba(61,255,140,${0.34 + 0.34 * pulse})`;
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(backX + normalX * side, backY + normalY * side);
    ctx.lineTo(backX - normalX * side, backY - normalY * side);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();
}

function drawStars(ctx, canvas, camera, metersH, timeSeconds, stars) {
  for (const star of stars) {
    const [sx, sy] = worldToScreen(star.x, star.z, canvas, camera, metersH);
    if (sx < -8 || sx > canvas.width + 8 || sy < -8 || sy > canvas.height + 8) continue;
    const twinkle = 0.58 + 0.42 * Math.sin(timeSeconds * (1.45 + star.r * 0.45) + star.phase);
    const alpha = Math.min(1, 0.5 + 0.62 * twinkle);
    const size = Math.max(1, Math.round(star.r));
    const color = star.tint < 0.34 ? "214,240,255" : star.tint < 0.67 ? "255,246,204" : "232,224,255";

    ctx.fillStyle = `rgba(${color},${alpha.toFixed(2)})`;
    ctx.fillRect(Math.round(sx), Math.round(sy), size, size);

    if (star.sparkle && alpha > 0.62) {
      ctx.globalAlpha = alpha * 0.78;
      ctx.strokeStyle = `rgb(${color})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx - size * 2.8, sy);
      ctx.lineTo(sx + size * 2.8, sy);
      ctx.moveTo(sx, sy - size * 2.8);
      ctx.lineTo(sx, sy + size * 2.8);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }
}

function createWindStreamSeed() {
  return {
    points: [],
    x: 0,
    y: 0,
    carry: 0,
    phase: randomRange(0, Math.PI * 2),
    seed: Math.floor(randomRange(1, 1000000)),
    direction: 0,
    maxPoints: Math.floor(randomRange(24, 56)),
  };
}

function pseudoRandom(seed) {
  const s = Math.sin(seed * 12.9898) * 43758.5453;
  return s - Math.floor(s);
}

function resetWindStream(stream, canvas, direction) {
  const dpr = window.devicePixelRatio || 1;
  const margin = canvas.width * 0.08;
  stream.points.length = 0;
  stream.direction = direction;
  stream.x = direction >= 0
    ? randomRange(-margin, canvas.width * 0.72)
    : randomRange(canvas.width * 0.28, canvas.width + margin);
  stream.y = randomRange(canvas.height * 0.12, canvas.height * 0.74);
  stream.carry = randomRange(0, 3 * dpr);
  stream.phase = randomRange(0, Math.PI * 2);
  stream.seed = Math.floor(randomRange(1, 1000000));
  stream.maxPoints = Math.floor(randomRange(24, 56));
  stream.points.push({
    x: stream.x,
    y: stream.y,
    phase: stream.phase,
  });
}

function drawWindStreamlines(ctx, canvas, camera, metersH, timeSeconds, streams, wind, dt) {
  if (!wind?.enabled) {
    for (const stream of streams) stream.points.length = 0;
    return;
  }

  const baseAccel = wind.baseAccel ?? wind.accel ?? 0;
  const direction = baseAccel < 0 ? -1 : 1;
  const dpr = window.devicePixelRatio || 1;
  const intensity = Math.min(1, Math.abs(baseAccel) / Math.max(0.001, wind.strength ?? DEFAULT_WIND.strength));
  const stepPx = Math.max(1.4 * dpr, 2);

  if (intensity < 0.035) return;

  ctx.save();
  for (const stream of streams) {
    if (stream.points.length === 0 || stream.direction !== direction) {
      resetWindStream(stream, canvas, direction);
    }

    const z = screenYToWorldZ(stream.y, canvas, camera, metersH);
    const surface = windSurfaceFactor(z, wind);
    const speedPx = (22 + 86 * intensity) * surface * dpr;
    stream.carry += speedPx * Math.min(dt, 0.05);

    const addCount = Math.min(5, Math.floor(stream.carry / stepPx));
    stream.carry -= addCount * stepPx;

    for (let i = 0; i < addCount; i++) {
      const idx = stream.points.length;
      const randomWiggle = (pseudoRandom(stream.seed + idx * 17) - 0.5) * (0.55 + intensity) * dpr;
      const smoothWiggle = Math.sin(timeSeconds * 1.7 + stream.phase + idx * 0.63) * (0.18 + 0.85 * intensity) * dpr;
      stream.x += direction * stepPx;
      stream.y += randomWiggle + smoothWiggle;
      stream.points.push({
        x: stream.x,
        y: stream.y,
        phase: randomRange(0, Math.PI * 2),
      });
    }

    const maxPoints = Math.round(stream.maxPoints * (0.55 + 0.75 * intensity));
    while (stream.points.length > maxPoints) stream.points.shift();

    const head = stream.points[stream.points.length - 1];
    const offscreen = !head || head.x < -canvas.width * 0.14 || head.x > canvas.width * 1.14 || head.y < -30 || head.y > canvas.height - 24 * dpr;
    if (offscreen) {
      resetWindStream(stream, canvas, direction);
      continue;
    }

    const localSurface = windSurfaceFactor(screenYToWorldZ(head.y, canvas, camera, metersH), wind);
    const px = Math.max(1, Math.round(1.25 * dpr));
    for (let i = 0; i < stream.points.length; i++) {
      const p = stream.points[i];
      const tail = stream.points.length <= 1 ? 1 : i / (stream.points.length - 1);
      const shimmer = 0.76 + 0.24 * Math.sin(timeSeconds * 1.3 + p.phase);
      const alpha = (0.05 + 0.23 * intensity) * localSurface * tail * shimmer;
      if (alpha < 0.006) continue;
      const wiggleY = Math.sin(timeSeconds * 1.1 + p.phase) * 0.45 * dpr;
      ctx.fillStyle = `rgba(196,202,204,${alpha.toFixed(3)})`;
      ctx.fillRect(Math.round(p.x), Math.round(p.y + wiggleY), px, px);
    }
  }
  ctx.restore();
}

function drawFlowers(ctx, canvas, camera, metersH, flowers) {
  const ppm = canvas.height / metersH;
  if (ppm < 3) return;

  for (const flower of flowers) {
    const [sx, sy] = worldToScreen(flower.x, flower.z, canvas, camera, metersH);
    if (sx < -10 || sx > canvas.width + 10 || sy < -10 || sy > canvas.height + 24) continue;

    const petalR = Math.max(1.4, flower.radius * ppm);
    const cx = sx;
    const cy = sy;

    ctx.fillStyle = flower.color;
    for (let i = 0; i < flower.petals; i++) {
      const a = flower.phase + i * Math.PI * 2 / flower.petals;
      ctx.save();
      ctx.translate(cx + Math.cos(a) * petalR * 0.9, cy + Math.sin(a) * petalR * 0.65);
      ctx.rotate(a);
      ctx.beginPath();
      ctx.ellipse(0, 0, petalR * 0.75, petalR * 0.42, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    ctx.fillStyle = "#fff2a0";
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(1, petalR * 0.32), 0, Math.PI * 2);
    ctx.fill();
  }
}

export function drawScene(ctx, canvas, camera, metersH, timeSeconds, sceneElements, scenarioName, target, wind = null, dt = 1 / 60) {
  const ppm = canvas.height / metersH;

  ctx.fillStyle = "#101628";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  drawStars(ctx, canvas, camera, metersH, timeSeconds, sceneElements.stars);
  drawWindStreamlines(ctx, canvas, camera, metersH, timeSeconds, sceneElements.windStreams, wind, dt);

  const [, groundY] = worldToScreen(0, 0, canvas, camera, metersH);
  if (groundY < canvas.height) {
    ctx.fillStyle = "#1f4b24";
    ctx.fillRect(0, groundY, canvas.width, canvas.height - groundY);
    drawFlowers(ctx, canvas, camera, metersH, sceneElements.flowers);
  }

  if (scenarioName === "landing") {
    drawLandingTarget(ctx, canvas, camera, metersH, groundY, timeSeconds, target);
  } else {
    if (groundY < canvas.height) {
      drawLandingTarget(ctx, canvas, camera, metersH, groundY, timeSeconds, { x: 0, width: 3.2, height: 0.35 }, false);
    }
    drawAscentTarget(ctx, canvas, camera, metersH, timeSeconds, target);
  }

  const minZ = camera.z - metersH / 2;
  const maxZ = camera.z + metersH / 2;
  const startTick = Math.ceil(minZ / 20) * 20;
  ctx.fillStyle = "rgba(255,255,255,0.55)";
  ctx.font = `${Math.max(10, Math.round(12 * (window.devicePixelRatio || 1)))}px monospace`;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (let z = startTick; z <= maxZ; z += 20) {
    const [, sy] = worldToScreen(0, z, canvas, camera, metersH);
    ctx.fillRect(0, sy, Math.max(8, ppm * 0.35), 1);
    ctx.fillText(`${z}m`, 12, sy);
  }
}

export function drawSmoke(ctx, canvas, camera, metersH, smoke) {
  for (const p of smoke) {
    const [sx, sy] = worldToScreen(p.x, p.z, canvas, camera, metersH);
    const ppm = canvas.height / metersH;
    const r = p.radius * ppm;
    if (r < 0.3) continue;

    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(165,165,165,${p.opacity.toFixed(3)})`;
    ctx.fill();
  }
}

export function drawExplosion(ctx, canvas, camera, metersH, explosion) {
  if (!explosion) return;

  const ppm = canvas.height / metersH;
  const [cx, cy] = worldToScreen(explosion.x, explosion.z, canvas, camera, metersH);
  const burst = Math.max(0, 1 - explosion.age / 0.55);
  const ember = Math.max(0, 1 - explosion.age / 1.35);
  const flashR = Math.max(8, (2.2 + explosion.age * 15) * ppm);
  const coreR = Math.max(5, (0.75 + explosion.age * 5.5) * ppm);
  const ringR = Math.max(7, (1.4 + explosion.age * 13) * ppm);
  const flashAlpha = 0.82 * burst;

  ctx.save();
  ctx.globalCompositeOperation = "lighter";

  if (flashAlpha > 0.01) {
    const flash = ctx.createRadialGradient(cx, cy, 0, cx, cy, flashR);
    flash.addColorStop(0.0, `rgba(255,255,230,${flashAlpha.toFixed(3)})`);
    flash.addColorStop(0.22, `rgba(255,216,88,${(flashAlpha * 0.9).toFixed(3)})`);
    flash.addColorStop(0.56, `rgba(255,85,24,${(flashAlpha * 0.45).toFixed(3)})`);
    flash.addColorStop(1.0, "rgba(255,85,24,0)");
    ctx.fillStyle = flash;
    ctx.beginPath();
    ctx.arc(cx, cy, flashR, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.strokeStyle = `rgba(255,232,128,${(0.72 * burst).toFixed(3)})`;
  ctx.lineWidth = Math.max(2, 0.09 * ppm);
  ctx.beginPath();
  ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = `rgba(255,244,190,${(0.86 * burst).toFixed(3)})`;
  ctx.beginPath();
  ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
  ctx.fill();

  for (const p of explosion.particles) {
    if (p.life <= 0) continue;
    const alpha = Math.max(0, p.life / p.ttl);
    const [sx, sy] = worldToScreen(p.x, p.z, canvas, camera, metersH);
    const r = Math.max(p.type === "spark" ? 1.5 : 2.5, p.radius * ppm);

    if (p.type === "smoke") {
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 0.38 * alpha;
      ctx.fillStyle = p.color;
      ctx.save();
      ctx.translate(sx, sy);
      ctx.rotate(p.angle);
      ctx.fillRect(-r, -r, r * 2, r * 2);
      ctx.restore();
      ctx.globalCompositeOperation = "lighter";
      continue;
    }

    ctx.globalAlpha = p.type === "spark" ? alpha : Math.min(1, 0.9 * alpha + 0.1 * ember);
    ctx.fillStyle = p.color;
    if (p.type === "spark") {
      ctx.fillRect(Math.round(sx - r * 2.2), Math.round(sy - r / 2), Math.round(r * 4.4), Math.max(1, Math.round(r)));
    } else if (p.shape === "triangle") {
      ctx.beginPath();
      ctx.moveTo(sx + Math.cos(p.angle) * r * 1.25, sy + Math.sin(p.angle) * r * 1.25);
      ctx.lineTo(sx + Math.cos(p.angle + Math.PI * 0.72) * r, sy + Math.sin(p.angle + Math.PI * 0.72) * r);
      ctx.lineTo(sx + Math.cos(p.angle - Math.PI * 0.72) * r, sy + Math.sin(p.angle - Math.PI * 0.72) * r);
      ctx.closePath();
      ctx.fill();
    } else {
      ctx.save();
      ctx.translate(sx, sy);
      ctx.rotate(p.angle);
      ctx.fillRect(-r * 0.72, -r * 0.72, r * 1.44, r * 1.44);
      ctx.restore();
    }
  }

  ctx.globalCompositeOperation = "source-over";
  for (const d of explosion.debris ?? []) {
    if (d.life <= 0) continue;
    const alpha = Math.max(0, d.life / d.ttl);
    const [sx, sy] = worldToScreen(d.x, d.z, canvas, camera, metersH);
    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(d.angle);
    ctx.globalAlpha = 0.85 * alpha;
    ctx.fillStyle = d.color;
    if (d.type === "nose") {
      ctx.beginPath();
      ctx.moveTo(0.55 * d.length * ppm, 0);
      ctx.lineTo(-0.45 * d.length * ppm, -0.5 * d.width * ppm);
      ctx.lineTo(-0.45 * d.length * ppm, 0.5 * d.width * ppm);
      ctx.closePath();
      ctx.fill();
    } else if (d.type === "fin") {
      ctx.beginPath();
      ctx.moveTo(-0.5 * d.length * ppm, 0.5 * d.width * ppm);
      ctx.lineTo(0.52 * d.length * ppm, 0.34 * d.width * ppm);
      ctx.lineTo(0.06 * d.length * ppm, -0.54 * d.width * ppm);
      ctx.closePath();
      ctx.fill();
    } else if (d.type === "engine") {
      ctx.fillRect(-0.48 * d.length * ppm, -0.5 * d.width * ppm, d.length * ppm, d.width * ppm);
      ctx.fillStyle = `rgba(255,123,34,${(0.9 * alpha).toFixed(3)})`;
      ctx.fillRect(0.1 * d.length * ppm, -0.34 * d.width * ppm, 0.5 * d.length * ppm, 0.68 * d.width * ppm);
    } else {
      ctx.fillRect(-0.5 * d.length * ppm, -0.5 * d.width * ppm, d.length * ppm, d.width * ppm);
      if (d.type === "body") {
        ctx.fillStyle = `rgba(255,90,42,${(0.62 * alpha).toFixed(3)})`;
        ctx.fillRect(-0.18 * d.length * ppm, -0.5 * d.width * ppm, 0.12 * d.length * ppm, d.width * ppm);
      }
    }
    ctx.restore();
  }

  ctx.restore();
}

export function drawSuccessEffect(ctx, canvas, camera, metersH, effect) {
  if (!effect) return;

  const ppm = canvas.height / metersH;
  const [cx, cy] = worldToScreen(effect.x, effect.z, canvas, camera, metersH);
  const pulseR = Math.max(6, (1.1 + effect.age * 5.5) * ppm);
  const pulseAlpha = Math.max(0, 0.5 * (1 - effect.age / 0.9));

  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = `rgba(61,255,140,${pulseAlpha.toFixed(3)})`;
  ctx.lineWidth = Math.max(2, 0.06 * ppm);
  ctx.beginPath();
  ctx.arc(cx, cy, pulseR, 0, Math.PI * 2);
  ctx.stroke();

  for (const p of effect.particles) {
    if (p.life <= 0) continue;
    const alpha = Math.max(0, p.life / p.ttl);
    const [sx, sy] = worldToScreen(p.x, p.z, canvas, camera, metersH);
    const r = Math.max(2, p.radius * ppm);
    ctx.globalAlpha = alpha;
    ctx.fillStyle = p.color;
    ctx.fillRect(Math.round(sx - r / 2), Math.round(sy - r / 2), Math.round(r), Math.round(r));
  }

  ctx.restore();
}

function fillScreenPoly(ctx, screenPts, fillStyle) {
  ctx.fillStyle = fillStyle;
  ctx.beginPath();
  ctx.moveTo(screenPts[0][0], screenPts[0][1]);
  for (let i = 1; i < screenPts.length; i++) ctx.lineTo(screenPts[i][0], screenPts[i][1]);
  ctx.closePath();
  ctx.fill();
}

function strokeScreenSeg(ctx, a, b, strokeStyle, lineWidthPx) {
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidthPx;
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.stroke();
}

function fillScreenCircle(ctx, center, rPx, fillStyle, strokeStyle = null, lineWidthPx = 1) {
  ctx.beginPath();
  ctx.arc(center[0], center[1], rPx, 0, Math.PI * 2);
  ctx.fillStyle = fillStyle;
  ctx.fill();
  if (strokeStyle != null) {
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidthPx;
    ctx.stroke();
  }
}

export function drawRocket(ctx, canvas, camera, metersH, state, delta, thrust) {
  const { x, z, theta } = state;
  const ppm = canvas.height / metersH;
  const L = ROCKET.body_length;
  const W = ROCKET.body_width;

  if (thrust > 0) {
    const thrustRatio = Math.min(1, thrust / PARAMS.T_max);
    let flameLen = L * 0.2 + L * thrustRatio;
    flameLen *= 1.0 + 0.05 * (2 * Math.random() - 1);
    const halfBase = W * 0.2;
    const flameLocal = [
      [-halfBase, -L * 0.10],
      [halfBase, -L * 0.10],
      [0, -L * 0.10 - flameLen],
    ];
    const [p0, p1, p2] = mapGimbalPolyToScreen(flameLocal, delta, theta, x, z, canvas, camera, metersH);
    const grad = ctx.createLinearGradient(p2[0], p2[1], p0[0], p0[1]);
    grad.addColorStop(0, "#ffd85a");
    grad.addColorStop(0.45, "#ff7b22");
    grad.addColorStop(1, "#ff2b20");
    ctx.globalAlpha = 0.9 * thrustRatio;
    fillScreenPoly(ctx, [p0, p1, p2], grad);
    ctx.globalAlpha = 1;
  }

  const finL = [
    [-W / 2, -L * 0.15],
    [-W * 1.1, -L * 0.38],
    [-W / 2, -L * 0.35],
  ];
  const finR = [
    [W / 2, -L * 0.15],
    [W * 1.1, -L * 0.38],
    [W / 2, -L * 0.35],
  ];
  fillScreenPoly(ctx, mapBodyPolyToScreen(finL, theta, x, z, canvas, camera, metersH), "#858585");
  fillScreenPoly(ctx, mapBodyPolyToScreen(finR, theta, x, z, canvas, camera, metersH), "#858585");

  const llS = mapBodyPolyToScreen([[-W / 2, -L * 0.35]], theta, x, z, canvas, camera, metersH)[0];
  const llE = mapBodyPolyToScreen([[-W * 0.9, -L * 0.47]], theta, x, z, canvas, camera, metersH)[0];
  const lrS = mapBodyPolyToScreen([[W / 2, -L * 0.35]], theta, x, z, canvas, camera, metersH)[0];
  const lrE = mapBodyPolyToScreen([[W * 0.9, -L * 0.47]], theta, x, z, canvas, camera, metersH)[0];
  strokeScreenSeg(ctx, llS, llE, "#606060", Math.max(1, ppm * 0.04));
  strokeScreenSeg(ctx, lrS, lrE, "#606060", Math.max(1, ppm * 0.04));

  const fuselage = [
    [-W / 2, -L * 0.35],
    [W / 2, -L * 0.35],
    [W / 2, L * 0.35],
    [-W / 2, L * 0.35],
  ];
  fillScreenPoly(ctx, mapBodyPolyToScreen(fuselage, theta, x, z, canvas, camera, metersH), "#f0f0f0");

  const noseR = W / 2;
  const noseZc = L * 0.35;
  const nosePts = [];
  for (let i = 0; i <= 24; i++) {
    const a = (i / 24) * Math.PI;
    nosePts.push([noseR * Math.cos(a), noseZc + noseR * Math.sin(a)]);
  }
  fillScreenPoly(ctx, mapBodyPolyToScreen(nosePts, theta, x, z, canvas, camera, metersH), "#e03030");

  const phR = W * 0.18;
  const phCenter = mapBodyPolyToScreen([[0, L * 0.08]], theta, x, z, canvas, camera, metersH)[0];
  fillScreenCircle(ctx, phCenter, phR * ppm, "#4a8fbd", "#303030", Math.max(1, ppm * 0.03));

  const stripe = [
    [-W / 2, -L * 0.06],
    [W / 2, -L * 0.06],
    [W / 2, -L * 0.04],
    [-W / 2, -L * 0.04],
  ];
  fillScreenPoly(ctx, mapBodyPolyToScreen(stripe, theta, x, z, canvas, camera, metersH), "#27b4ff");

  const pivot = mapBodyPolyToScreen([[0, -L * 0.35]], theta, x, z, canvas, camera, metersH)[0];
  fillScreenCircle(ctx, pivot, W * 0.08 * ppm, "#404040");

  const topW = W * 0.35;
  const botW = W * 0.55;
  const nozzleH = L * 0.10;
  const nozzle = [
    [-topW / 2, 0],
    [topW / 2, 0],
    [botW / 2, -nozzleH],
    [-botW / 2, -nozzleH],
  ];
  fillScreenPoly(ctx, mapGimbalPolyToScreen(nozzle, delta, theta, x, z, canvas, camera, metersH), "#505050");
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

export function drawHUD(ctx, canvas, sim) {
  const { z, vx, vz, theta } = sim.state;
  const speed = Math.sqrt(vx * vx + vz * vz);
  const thetaWrapped = ((theta % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  const dpr = window.devicePixelRatio || 1;
  const fontSize = Math.round(14 * dpr);
  const pad = Math.round(10 * dpr);
  const lineH = fontSize + Math.round(4 * dpr);
  const panelW = Math.round(190 * dpr);
  const panelH = lineH * 6 + pad * 2;
  const windValue = sim.wind
    ? `${sim.wind.accel >= 0 ? "+" : "-"}${Math.abs(sim.wind.accel).toFixed(2).padStart(4)}`
    : "+0.00";
  const windLine = sim.wind?.enabled
    ? `WIND   ${windValue} m/s2`
    : "WIND    OFF";
  const lines = [
    `ALT    ${z.toFixed(1).padStart(6)} m`,
    `SPEED  ${speed.toFixed(1).padStart(6)} m/s`,
    `ANGLE  ${(thetaWrapped * 180 / Math.PI).toFixed(1).padStart(6)} deg`,
    `THRUST ${(sim.thrust / PARAMS.T_max * 100).toFixed(0).padStart(6)} %`,
    `GIMBAL ${(sim.delta * 180 / Math.PI).toFixed(1).padStart(6)} deg`,
    windLine,
  ];

  ctx.fillStyle = "rgba(0,0,0,0.84)";
  roundRect(ctx, pad, pad, panelW, panelH, Math.round(4 * dpr));
  ctx.fill();

  ctx.strokeStyle = "rgba(61,255,140,0.35)";
  ctx.lineWidth = Math.max(1, dpr);
  ctx.stroke();

  ctx.font = `${fontSize}px monospace`;
  ctx.fillStyle = "#e8fff0";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  for (let i = 0; i < lines.length; i++) {
    ctx.fillText(lines[i], pad * 2, pad * 2 + i * lineH);
  }

  if (sim.paused) {
    ctx.font = `bold ${Math.round(28 * dpr)}px monospace`;
    ctx.fillStyle = "rgba(232,255,240,0.88)";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("PAUSED", canvas.width / 2, canvas.height / 2);
  }

  if (sim.status === "success" || sim.status === "exploding") {
    const label = sim.status === "success" ? "SUCCESS" : "RESETTING";
    const color = sim.status === "success" ? "#3dff8c" : "#ffd85a";
    ctx.font = `bold ${Math.round(24 * dpr)}px monospace`;
    ctx.fillStyle = color;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, canvas.width / 2, Math.round(72 * dpr));

    if (sim.status === "success") {
      ctx.font = `${Math.round(13 * dpr)}px monospace`;
      ctx.fillStyle = "rgba(232,255,240,0.88)";
      ctx.fillText("CONGRATULATIONS", canvas.width / 2, Math.round(98 * dpr));
    }
  }
}

export function drawKeyHints(ctx, canvas) {
  const dpr = window.devicePixelRatio || 1;
  const h = Math.round(30 * dpr);
  const text = "↑/↓ THRUST   ←/→ GIMBAL   W WIND   SPACE PAUSE   R RESET   1 ASCENT   2 LANDING";

  ctx.fillStyle = "rgba(0,0,0,0.50)";
  ctx.fillRect(0, canvas.height - h, canvas.width, h);
  ctx.strokeStyle = "rgba(61,255,140,0.25)";
  ctx.beginPath();
  ctx.moveTo(0, canvas.height - h + 0.5);
  ctx.lineTo(canvas.width, canvas.height - h + 0.5);
  ctx.stroke();

  ctx.font = `${Math.round(12 * dpr)}px monospace`;
  ctx.fillStyle = "rgba(232,255,240,0.78)";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, canvas.width / 2, canvas.height - h / 2);
}
