/**
 * VacMan viewer: wires world graph, Three.js scene, dust, UI, and the game loop.
 *
 * Kinematics:  v = (vL + vR)/2,  ω = (vR − vL)/L
 *              ẋ = v cos θ,  ż = v sin θ,  θ̇ = ω
 *
 * Cat pathfinding: Minkowski-expanded wall rects (octagon footprint) → visibility graph
 * (AABB culling + proper segment–edge crossings + interior samples) → A*.
 */
import * as THREE from "https://esm.sh/three@0.165.0";
import {
  BASE_VISUAL_R,
  CAT_OCT_R,
  CATCH_R,
  CAT_WAYPOINT_R,
  REPLAN_INTERVAL,
  VAC_RADIUS,
} from "./constants.js";
import { loadCases, validateCase } from "./cases.js";
import { clamp, dist, segmentVisible } from "./math2d.js";
import {
  astar,
  buildWorldTransform,
  collectWallRects,
  cellFreeFromWalls,
  makeCatCspaceContext,
  pointFreeInObstacles,
  pushOutOfWalls,
} from "./navigation.js";
import { createCatPlanner } from "./cat_path.js";
import { createDustSystem } from "./dust.js";
import { attachResize, createArenaScene, layoutBaseStation } from "./scene.js";
import { addCatCollisionRing, assetUrls, decorateVacuum, fitModel, loadOBJ } from "./models.js";
import { createGameUi } from "./ui.js?v=keyboard-focus";
import { createMinimap } from "./minimap.js";
import { vacuumWheelSpeedsFromKeys } from "./controls.js?v=keyboard-focus";
import { createVacuumLightCone } from "./cone_shader.js";
import {
  catmanLinearSpeed,
  differentialDriveTwist,
  stepPlanarUnicycle,
} from "./dynamics.js";

const CAT_TRACE_MAX_EVENTS = 2500;
const MIN_BATTERY_CHARGE = 180;
const DUST_SWEEP_CHARGE_FACTOR = 1.8;
const ARENA_TRAVEL_CHARGE_FACTOR = 4.0;

function traceNum(v) {
  return Number.isFinite(v) ? Math.round(v * 1000) / 1000 : v;
}

function tracePoint(p) {
  return p ? { x: traceNum(p.x), z: traceNum(p.z) } : null;
}

function tracePath(path) {
  if (!path || !path.length) return { count: 0, length: 0, points: [] };
  let length = 0;
  for (let i = 1; i < path.length; i++) length += dist(path[i - 1], path[i]);
  return {
    count: path.length,
    length: traceNum(length),
    points: path.map(tracePoint),
  };
}

function tracePlannerState(planner) {
  return {
    wpIdx: planner.cat.wpIdx,
    partial: planner.cat.partial,
    blockedAtWaypoint: !!planner.cat.blockedAtWaypoint,
    goal: tracePoint(planner.cat.goal),
    waypoint: tracePoint(planner.currentWaypoint()),
    path: tracePath(planner.cat.path),
  };
}

function estimateBatteryCharge(dust, arenaX, arenaZ) {
  const dustSweep = dust.totalDust * dust.cellSizeMin * DUST_SWEEP_CHARGE_FACTOR;
  const arenaTravel = Math.hypot(arenaX, arenaZ) * ARENA_TRAVEL_CHARGE_FACTOR;
  return Math.ceil(Math.max(MIN_BATTERY_CHARGE, dustSweep + arenaTravel));
}

function chargeText(charge) {
  return String(Math.ceil(Math.max(0, charge))).padStart(4);
}

function createCatTrace(enabled, container) {
  if (!enabled) return { enabled: false, record() {}, clear() {} };

  const events = [];
  let nextId = 0;
  const panel = document.createElement("pre");
  panel.style.cssText = [
    "position:absolute",
    "right:8px",
    "bottom:8px",
    "z-index:50",
    "max-width:430px",
    "max-height:220px",
    "overflow:hidden",
    "padding:8px",
    "color:#d9f99d",
    "background:rgba(0,0,0,0.72)",
    "border:1px solid rgba(217,249,157,0.45)",
    "font:11px/1.35 Courier New,monospace",
    "white-space:pre-wrap",
    "pointer-events:none",
  ].join(";");
  panel.textContent = "CAT TRACE recording";
  container.appendChild(panel);

  function dump() {
    return JSON.stringify(events, null, 2);
  }

  function updatePanel(event) {
    const lines = [
      `CAT TRACE ${events.length}/${CAT_TRACE_MAX_EVENTS}`,
      `last ${event.type} @ ${traceNum(event.t)}s frame ${event.frame ?? "-"}`,
      `cat ${JSON.stringify(event.cat ?? null)}`,
      `target ${JSON.stringify(event.target ?? null)}`,
      `move ${event.moveStatus ?? "-"} still ${event.stillFrames ?? 0}`,
    ];
    if (event.planner) {
      lines.push(`wp ${event.planner.wpIdx} blocked ${event.planner.blockedAtWaypoint}`);
      lines.push(`path ${event.planner.path.count} len ${event.planner.path.length}`);
    }
    panel.textContent = lines.join("\n");
  }

  function record(type, data = {}) {
    const event = { id: nextId++, type, ...data };
    events.push(event);
    if (events.length > CAT_TRACE_MAX_EVENTS) events.shift();
    window.__VACMAN_CAT_TRACE__ = events;
    window.__VACMAN_CAT_LAST__ = event;
    updatePanel(event);
    if (type !== "cat-frame" && window.VACMAN_TRACE_LOGS) console.log("[vacman-cat]", JSON.stringify(event));
  }

  window.__VACMAN_CAT_TRACE__ = events;
  window.dumpVacmanCatTrace = dump;
  window.clearVacmanCatTrace = () => {
    events.length = 0;
    nextId = 0;
    panel.textContent = "CAT TRACE cleared";
  };
  window.copyVacmanCatTrace = async () => {
    const text = dump();
    await navigator.clipboard.writeText(text);
    return text.length;
  };
  console.info("[vacman-cat] recording enabled: use dumpVacmanCatTrace() or copyVacmanCatTrace()");

  return { enabled: true, record, clear: window.clearVacmanCatTrace };
}

export async function startVacmanViewer(config = {}) {
  const cases = await loadCases();
  if (!cases.length) throw new Error("open_cases.json: no cases");
  const params = new URLSearchParams(window.location.search);
  const requestedCase = config.caseIndex ?? params.get("case") ?? 0;
  const caseIdx = clamp(parseInt(requestedCase, 10) || 0, 0, cases.length - 1);
  const caseData = cases[caseIdx];
  validateCase(caseData);

  const { halfX, halfZ, toWorld, toWorldHeading } = buildWorldTransform(caseData);
  const ARENA_X = halfX * 2;
  const ARENA_Z = halfZ * 2;
  const baseJ = caseData.base;
  const catJ = caseData.catman;
  const bw = toWorld(baseJ[0], baseJ[1]);
  const basePos = new THREE.Vector3(bw.x, 0, bw.z);
  const wallRects = collectWallRects(caseData, toWorld);

  const { collisionObstacles, planningObstacles, cspaceObstacles, graphNodes, visAdj } = makeCatCspaceContext(wallRects);

  const containerId = config.containerId ?? "game-container";
  const container = document.getElementById(containerId) || document.body;

  const arena = createArenaScene(container, {
    halfX,
    halfZ,
    ARENA_X,
    ARENA_Z,
    wallRects,
  });

  layoutBaseStation(arena.baseGroup, basePos, halfX, halfZ, arena);

  const dust = createDustSystem({
    scene: arena.worldGroup,
    ARENA_X,
    ARENA_Z,
    halfX,
    halfZ,
    basePos,
    wallRects,
    cellFreeFromWalls,
  });

  dust.fillDust();
  const batteryCharge = estimateBatteryCharge(dust, ARENA_X, ARENA_Z);

  const { vacUrl, catUrl } = assetUrls();
  const [vacObj, catObj] = await Promise.all([loadOBJ(vacUrl), loadOBJ(catUrl)]);
  fitModel(vacObj, 1.0);
  fitModel(catObj, 1.0);
  arena.worldGroup.add(vacObj);
  arena.worldGroup.add(catObj);

  decorateVacuum(vacObj);
  addCatCollisionRing(catObj);

  const vacStart = toWorld(baseJ[0], baseJ[1]);
  const catStartRaw = toWorld(catJ[0], catJ[1]);
  function clampCatPoint(p) {
    return {
      x: clamp(p.x, -halfX + CAT_OCT_R, halfX - CAT_OCT_R),
      z: clamp(p.z, -halfZ + CAT_OCT_R, halfZ - CAT_OCT_R),
    };
  }

  function freeForCat(p) {
    return pointFreeInObstacles(p, collisionObstacles);
  }

  function chooseNearestFreeCatPoint(rawPoint, from = null, preferred = []) {
    const raw = clampCatPoint(rawPoint);
    const anchor = from ? clampCatPoint(from) : raw;
    let best = null;
    let bestScore = Infinity;

    function consider(p) {
      const q = clampCatPoint(p);
      if (!freeForCat(q)) return;
      const score = dist(q, raw) + 0.05 * dist(q, anchor);
      if (score < bestScore) {
        best = q;
        bestScore = score;
      }
    }

    for (const p of preferred) consider(p);
    consider(raw);
    const rings = [0.06, 0.12, 0.22, 0.36, 0.54, 0.76, 1.02, 1.34, 1.72, 2.16];
    const samples = 56;
    for (const r of rings) {
      for (let i = 0; i < samples; i++) {
        const a = (i / samples) * Math.PI * 2;
        consider({ x: raw.x + Math.cos(a) * r, z: raw.z + Math.sin(a) * r });
      }
      if (best && bestScore < r + 0.12) break;
    }
    return best;
  }

  function projectCatBodyPoint(x, z, from = null) {
    const raw = clampCatPoint({ x, z });
    if (freeForCat(raw)) return raw;
    const pushed = clampCatPoint(pushOutOfWalls(x, z, wallRects, CAT_OCT_R));
    if (freeForCat(pushed)) return pushed;
    return chooseNearestFreeCatPoint(raw, from, [pushed]) ?? pushed;
  }

  function projectCatChaseTarget(x, z, from) {
    const raw = clampCatPoint({ x, z });
    if (freeForCat(raw)) return raw;

    const pushed = projectCatBodyPoint(x, z, from);
    let best = freeForCat(pushed) ? pushed : null;
    let bestScore = best ? dist(best, raw) + 0.08 * dist(best, from) : Infinity;
    const rings = [0.18, 0.32, 0.5, 0.72, 0.96, 1.22, 1.52];
    const samples = 40;
    for (const r of rings) {
      for (let i = 0; i < samples; i++) {
        const a = (i / samples) * Math.PI * 2;
        const p = clampCatPoint({ x: x + Math.cos(a) * r, z: z + Math.sin(a) * r });
        if (!freeForCat(p)) continue;
        const score = dist(p, raw) + 0.08 * dist(p, from);
        if (score < bestScore) {
          best = p;
          bestScore = score;
        }
      }
      if (best && bestScore < r + 0.2) break;
    }
    return best ?? pushed;
  }
  const catStart = projectCatBodyPoint(catStartRaw.x, catStartRaw.z);
  const vac = { x: vacStart.x, z: vacStart.z, th: toWorldHeading(baseJ[2] ?? 0) };
  const cat = { x: catStart.x, z: catStart.z, th: toWorldHeading(catJ[2] ?? 0) };

  const catPlanner = createCatPlanner({
    graphNodes,
    visAdj,
    planningObstacles,
    collisionObstacles,
    cspaceObstacles,
    segmentVisible,
    dist,
    astar,
  });

  const ui = createGameUi(container, arena.renderer.domElement);
  cases.forEach((item, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = item.id || `case ${idx + 1}`;
    if (idx === caseIdx) opt.selected = true;
    ui.caseSelect.appendChild(opt);
  });
  ui.caseSelect.onchange = () => {
    const nextParams = new URLSearchParams(window.location.search);
    nextParams.set("case", ui.caseSelect.value);
    window.location.search = `?${nextParams.toString()}`;
  };

  const minimap = createMinimap(container, { halfX, halfZ, wallRects, basePos, dust });
  const lightCone = createVacuumLightCone(arena.worldGroup, { wallRects, halfX, halfZ });

  let state = "PLAYING";
  let elapsed = 0;
  let batteryRemaining = batteryCharge;
  let leftBase = false;
  let vacmanMoved = false;
  let replanAcc = 0;
  const catSpeed = catmanLinearSpeed(caseData.catman_speed);
  const catTrace = createCatTrace(Boolean(config.catDebug) || params.has("catdebug") || params.get("debug") === "cat", container);
  let catFrame = 0;
  let catStillFrames = 0;

  const camSmooth = new THREE.Vector3(
    vac.x + arena.cameraState.lateralOffset,
    arena.cameraState.baseHeight,
    vac.z - arena.cameraState.followDistance,
  );
  const lookSmooth = new THREE.Vector3(
    vac.x,
    arena.cameraState.lookLift,
    vac.z + arena.cameraState.lookAhead,
  );

  function updateHudSides(vL = 0, vR = 0) {
    ui.hudStats.innerHTML = [
      `CHARGE ${chargeText(batteryRemaining)}`,
      `DUST   ${String(dust.cleaned).padStart(4)}`,
      `BASE   ${leftBase ? "RETURN" : "START "}`,
      `CATMAN V ${catSpeed.toFixed(2).padStart(6)}`,
    ].join("<br>");
    ui.hudR.textContent = "";
  }

  function resetGame() {
    Object.assign(vac, { x: vacStart.x, z: vacStart.z, th: toWorldHeading(baseJ[2] ?? 0) });
    Object.assign(cat, { x: catStart.x, z: catStart.z, th: toWorldHeading(catJ[2] ?? 0) });
    state = "PLAYING";
    elapsed = 0;
    batteryRemaining = batteryCharge;
    leftBase = false;
    vacmanMoved = false;
    replanAcc = 0;
    catFrame = 0;
    catStillFrames = 0;
    catPlanner.resetPath();
    catTrace.clear();
    catTrace.record("reset", {
      t: traceNum(elapsed),
      frame: catFrame,
      cat: tracePoint(cat),
      target: tracePoint(vac),
      planner: tracePlannerState(catPlanner),
    });
    dust.resetCleaned();
    dust.fillDust();
    ui.overlay.style.display = "none";
    ui.clearKeys();
  }

  ui.oBtn.onclick = resetGame;

  attachResize(container, arena.camera, arena.renderer);

  let prevT = performance.now();

  function tick() {
    requestAnimationFrame(tick);
    const now = performance.now();
    const dt = Math.min((now - prevT) / 1000, 0.05);
    prevT = now;

    if (ui.resetRequested) {
      ui.resetRequested = false;
      resetGame();
    }

    const tSec = now / 1000;
    const goalPulse = 0.82 + 0.18 * Math.sin(tSec * 3.5);
    arena.goalGlow.visible = leftBase;
    arena.goalGlowMaterial.opacity = leftBase ? 0.45 * goalPulse : 0;
    arena.voxelMat.emissiveIntensity = leftBase ? 0.16 + 0.16 * goalPulse : 0.05;
    arena.beaconVoxelMat.emissiveIntensity = leftBase ? 0.55 + 0.45 * goalPulse : 0.08;
    for (const m of arena.beaconVoxels) {
      m.material.emissiveIntensity = leftBase ? 0.55 + 0.45 * Math.sin(tSec * 4 + m.position.x) : 0.08;
    }

    if (state === "PLAYING") {
      elapsed += dt;
      batteryRemaining = Math.max(0, batteryRemaining - dt);
      const { vL, vR } = vacuumWheelSpeedsFromKeys(ui.keys);
      if (Math.abs(vL) > 1e-3 || Math.abs(vR) > 1e-3) vacmanMoved = true;
      const catActive = vacmanMoved;
      updateHudSides(vL, vR);

      if (batteryRemaining <= 0) {
        state = "OUT_OF_CHARGE";
        ui.oText.innerHTML =
          `<span style="color:#ffd85a;font:bold 34px/1.15 monospace">OUT OF CHARGE</span>` +
          `<br><span style="font:16px/1.45 monospace">DUST ${dust.cleaned}</span>` +
          `<br><span style="font:16px/1.45 monospace">CHARGE 0</span>`;
        ui.overlay.style.display = "flex";
        ui.setBar(ui.barL, 0);
        ui.setBar(ui.barR, 0);
      } else {
        const vacTwist = differentialDriveTwist(vL, vR);
        stepPlanarUnicycle(vac, dt, vacTwist);
        const pushed = pushOutOfWalls(vac.x, vac.z, wallRects, VAC_RADIUS);
        vac.x = clamp(pushed.x, -halfX + VAC_RADIUS, halfX - VAC_RADIUS);
        vac.z = clamp(pushed.z, -halfZ + VAC_RADIUS, halfZ - VAC_RADIUS);

        const baseCenterDistance = Math.hypot(vac.x - basePos.x, vac.z - basePos.z);
        if (baseCenterDistance > BASE_VISUAL_R) leftBase = true;

        vacObj.position.set(vac.x, 0, vac.z);
        vacObj.rotation.y = -vac.th + Math.PI / 2;

        dust.cleanAt(vac.x, vac.z);

        if (catActive) {
          catFrame++;
          replanAcc += dt;
          const catBefore = { x: cat.x, z: cat.z };
          const catFree = projectCatBodyPoint(cat.x, cat.z, cat);
          const entryProjection = dist(cat, catFree);
          if (entryProjection > 1e-3) {
            catTrace.record("cat-project", {
              t: traceNum(elapsed),
              frame: catFrame,
              from: tracePoint(cat),
              cat: tracePoint(catFree),
              shift: traceNum(entryProjection),
            });
          }
          cat.x = catFree.x;
          cat.z = catFree.z;
          const chaseTarget = projectCatChaseTarget(vac.x, vac.z, cat);

          catPlanner.advanceWaypoints(cat, CAT_WAYPOINT_R);
          const usableBeforePlan = catPlanner.pathUsable(cat);
          const targetMoved = catPlanner.targetMoved(chaseTarget);
          if (
            !usableBeforePlan ||
            (targetMoved && replanAcc >= REPLAN_INTERVAL)
          ) {
            const reason = !usableBeforePlan
              ? (catPlanner.cat.blockedAtWaypoint ? "blocked-at-waypoint" : "path-unusable")
              : "target-moved";
            catTrace.record("cat-plan-request", {
              t: traceNum(elapsed),
              frame: catFrame,
              reason,
              cat: tracePoint(cat),
              target: tracePoint(chaseTarget),
              targetDist: traceNum(dist(cat, chaseTarget)),
              directVisible: segmentVisible(cat, chaseTarget, cspaceObstacles),
              planner: tracePlannerState(catPlanner),
            });
            replanAcc = 0;
            catPlanner.planCatPath(cat, chaseTarget);
            catPlanner.advanceWaypoints(cat, CAT_WAYPOINT_R);
            catTrace.record("cat-plan-result", {
              t: traceNum(elapsed),
              frame: catFrame,
              reason,
              cat: tracePoint(cat),
              target: tracePoint(chaseTarget),
              planner: tracePlannerState(catPlanner),
            });
          }

          catPlanner.shortcutPath(cat, chaseTarget);
          if (!catPlanner.pathUsable(cat) && replanAcc >= REPLAN_INTERVAL) {
            catTrace.record("cat-plan-request", {
              t: traceNum(elapsed),
              frame: catFrame,
              reason: "post-shortcut-unusable",
              cat: tracePoint(cat),
              target: tracePoint(chaseTarget),
              targetDist: traceNum(dist(cat, chaseTarget)),
              directVisible: segmentVisible(cat, chaseTarget, cspaceObstacles),
              planner: tracePlannerState(catPlanner),
            });
            replanAcc = 0;
            catPlanner.planCatPath(cat, chaseTarget);
            catPlanner.advanceWaypoints(cat, CAT_WAYPOINT_R);
            catTrace.record("cat-plan-result", {
              t: traceNum(elapsed),
              frame: catFrame,
              reason: "post-shortcut-unusable",
              cat: tracePoint(cat),
              target: tracePoint(chaseTarget),
              planner: tracePlannerState(catPlanner),
            });
          }

          const wp = catPlanner.currentWaypoint();
          const usableForMove = catPlanner.pathUsable(cat);
          let moveStatus = wp ? "path-unusable" : "no-waypoint";
          if (wp && catPlanner.cat.blockedAtWaypoint) moveStatus = "blocked-at-waypoint";
          if (wp && usableForMove) {
            const dx = wp.x - cat.x;
            const dz = wp.z - cat.z;
            const d = Math.hypot(dx, dz);
            if (d > 1e-4) {
              cat.th = Math.atan2(dz, dx);
              const step = Math.min(catSpeed * dt, d);
              const next = {
                x: cat.x + (dx / d) * step,
                z: cat.z + (dz / d) * step,
              };
              if (catPlanner.movementSafe({ x: cat.x, z: cat.z }, next)) {
                cat.x = next.x;
                cat.z = next.z;
                moveStatus = "moved";
              } else {
                moveStatus = "blocked-step";
                catTrace.record("cat-blocked-step", {
                  t: traceNum(elapsed),
                  frame: catFrame,
                  cat: tracePoint(cat),
                  next: tracePoint(next),
                  target: tracePoint(chaseTarget),
                  waypoint: tracePoint(wp),
                  planner: tracePlannerState(catPlanner),
                });
                replanAcc = REPLAN_INTERVAL;
                catPlanner.resetPath();
              }
            } else {
              moveStatus = "at-waypoint";
            }
          }

          const cp = projectCatBodyPoint(cat.x, cat.z, catBefore);
          const exitProjection = dist(cat, cp);
          if (exitProjection > 1e-3) {
            catTrace.record("cat-project-after-move", {
              t: traceNum(elapsed),
              frame: catFrame,
              from: tracePoint(cat),
              cat: tracePoint(cp),
              shift: traceNum(exitProjection),
              moveStatus,
            });
          }
          cat.x = cp.x;
          cat.z = cp.z;
          catPlanner.advanceWaypoints(cat, CAT_WAYPOINT_R);

          const moved = dist(catBefore, cat);
          const targetDist = dist(cat, chaseTarget);
          if (targetDist >= CATCH_R && moved < 1e-5) catStillFrames++;
          else catStillFrames = 0;

          if (catStillFrames === 10 || (catStillFrames > 10 && catStillFrames % 30 === 0)) {
            catTrace.record("cat-freeze", {
              t: traceNum(elapsed),
              frame: catFrame,
              cat: tracePoint(cat),
              vac: tracePoint(vac),
              target: tracePoint(chaseTarget),
              targetDist: traceNum(targetDist),
              moveStatus,
              moved: traceNum(moved),
              stillFrames: catStillFrames,
              directVisible: segmentVisible(cat, chaseTarget, cspaceObstacles),
              planner: tracePlannerState(catPlanner),
            });
          } else if (catTrace.enabled && catFrame % 30 === 0) {
            catTrace.record("cat-frame", {
              t: traceNum(elapsed),
              frame: catFrame,
              cat: tracePoint(cat),
              vac: tracePoint(vac),
              target: tracePoint(chaseTarget),
              targetDist: traceNum(targetDist),
              moveStatus,
              moved: traceNum(moved),
              stillFrames: catStillFrames,
              planner: tracePlannerState(catPlanner),
            });
          }
        }

        catObj.position.set(cat.x, 0, cat.z);
        catObj.rotation.y = catActive ? -cat.th + Math.PI / 2 : catObj.rotation.y;

        if (catActive && Math.hypot(vac.x - cat.x, vac.z - cat.z) < CATCH_R) {
          state = "LOST";
          ui.oText.innerHTML =
            `<span style="color:#ff5252;font:bold 34px/1.15 monospace">CAUGHT</span>` +
            `<br><span style="font:16px/1.45 monospace">DUST ${dust.cleaned}</span>` +
            `<br><span style="font:16px/1.45 monospace">CHARGE ${chargeText(batteryRemaining).trim()}</span>`;
          ui.overlay.style.display = "flex";
        } else if (leftBase && baseCenterDistance < BASE_VISUAL_R) {
          state = "WON";
          ui.oText.innerHTML =
            `<span style="color:#00e676;font:bold 34px/1.15 monospace">BASE</span>` +
            `<br><span style="font:16px/1.45 monospace">DUST ${dust.cleaned}</span>` +
            `<br><span style="font:16px/1.45 monospace">CHARGE ${chargeText(batteryRemaining).trim()}</span>`;
          ui.overlay.style.display = "flex";
        }

        ui.setBar(ui.barL, vL);
        ui.setBar(ui.barR, vR);
      }
    }

    lightCone.update(vac);
    minimap.render({ vac, cat, baseActive: leftBase });

    const zoom = arena.cameraState.zoom;
    const desiredCam = new THREE.Vector3(
        vac.x + arena.cameraState.lateralOffset * zoom,
      arena.cameraState.baseHeight * zoom,
        vac.z - arena.cameraState.followDistance * zoom,
    );
    const tgt = new THREE.Vector3(
        vac.x,
      arena.cameraState.lookLift,
        vac.z + arena.cameraState.lookAhead,
    );
    camSmooth.lerp(desiredCam, 0.08);
    lookSmooth.lerp(tgt, 0.12);
    arena.camera.position.copy(camSmooth);
    arena.camera.lookAt(lookSmooth);

    arena.renderer.render(arena.scene, arena.camera);
  }

  tick();
}
