import {
  applyKeyboardControls,
  createExplosion,
  createRocketState,
  createSuccessEffect,
  DEFAULT_WIND,
  SCENARIOS,
  spawnSmoke,
  stepSimulation,
  TARGETS,
  updateExplosion,
  updateSuccessEffect,
  updateSmoke,
} from "./dynamics.js";
import {
  createSceneElements,
  drawExplosion,
  drawHUD,
  drawKeyHints,
  drawRocket,
  drawScene,
  drawSmoke,
  drawSuccessEffect,
  resizeCanvasToDisplaySize,
} from "./render.js";

const DEFAULT_CONFIG = {
  containerId: null,
  canvasId: "rocket-canvas",
  initialScenario: "ascent",
  windEnabled: true,
  windStrength: DEFAULT_WIND.strength,
  windSeed: DEFAULT_WIND.seed,
};

const RESET_AFTER_CRASH = 2.15;
const RESET_AFTER_SUCCESS = 1.8;
const DEBUG_ROCKET = false;
const ROCKET_KEY_CODES = new Set([
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "Space",
  "KeyR",
  "KeyW",
  "Digit1",
  "Digit2",
  "Escape",
]);

function angleError(a, b) {
  return Math.atan2(Math.sin(a - b), Math.cos(a - b));
}

function logRocket(event, payload = {}) {
  if (!DEBUG_ROCKET) return;
  console.log(`[rocket] ${event}`, payload);
}

function isSoftLanding(state, target) {
  const speed = Math.hypot(state.vx, state.vz);
  return (
    Math.abs(state.x - target.x) < target.xTolerance &&
    speed < target.speedTolerance &&
    Math.abs(state.vz) < target.verticalSpeedTolerance &&
    Math.abs(state.theta) < target.thetaTolerance &&
    Math.abs(state.omega) < target.omegaTolerance
  );
}

function isAscentSuccess(state, target) {
  return (
    state.z >= target.z &&
    Math.abs(angleError(state.theta, target.theta)) < target.thetaTolerance
  );
}

function isAscentAngleWrong(state, target) {
  return state.z > target.z && Math.abs(angleError(state.theta, target.theta)) >= target.thetaTolerance;
}

function ascentDebugState(state, target) {
  const err = angleError(state.theta, target.theta);
  return {
    z: Number(state.z.toFixed(2)),
    vz: Number(state.vz.toFixed(2)),
    vx: Number(state.vx.toFixed(2)),
    thetaDeg: Number((state.theta * 180 / Math.PI).toFixed(2)),
    targetThetaDeg: Number((target.theta * 180 / Math.PI).toFixed(2)),
    angleErrorDeg: Number((err * 180 / Math.PI).toFixed(2)),
    omega: Number(state.omega.toFixed(2)),
    overTarget: state.z >= target.z,
    angleOk: Math.abs(err) < target.thetaTolerance,
    rising: state.vz > 0,
  };
}

export function startRocketViewer(config = {}) {
  const query = new URLSearchParams(window.location.search);
  const scenarioParam = query.get("scenario");
  const windParam = query.get("wind");
  const settings = {
    ...DEFAULT_CONFIG,
    ...config,
    ...(SCENARIOS[scenarioParam] ? { initialScenario: scenarioParam } : {}),
    ...(windParam === "0" || windParam === "false" ? { windEnabled: false } : {}),
    ...(windParam === "1" || windParam === "true" ? { windEnabled: true } : {}),
  };
  let canvas = document.getElementById(settings.canvasId);
  if (!canvas && settings.containerId) {
    const container = document.getElementById(settings.containerId);
    canvas = container?.querySelector("canvas") ?? null;
    if (!canvas && container) {
      canvas = document.createElement("canvas");
      canvas.id = settings.canvasId;
      canvas.style.display = "block";
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      container.appendChild(canvas);
    }
  }
  if (!canvas) {
    throw new Error(`Rocket viewer canvas not found: #${settings.canvasId}`);
  }
  canvas.tabIndex = 0;
  canvas.setAttribute("aria-label", "Rocket control surface");
  canvas.style.outline = "none";
  let controlsActive = false;
  let notebookKeyboardPreviouslyEnabled = null;

  const ctx = canvas.getContext("2d");
  const keys = {};
  const sceneElements = createSceneElements();
  let currentScenarioName = SCENARIOS[settings.initialScenario] ? settings.initialScenario : "ascent";
  let currentWindEnabled = Boolean(settings.windEnabled);
  let sim = createRocketState(SCENARIOS[currentScenarioName], {
    enabled: currentWindEnabled,
    strength: settings.windStrength,
    seed: settings.windSeed,
  });
  const camera = { x: sim.state.x, z: Math.max(sim.state.z, 30) };

  function reset(scenarioName = currentScenarioName) {
    currentScenarioName = SCENARIOS[scenarioName] ? scenarioName : "ascent";
    sim = createRocketState(SCENARIOS[currentScenarioName], {
      enabled: currentWindEnabled,
      strength: settings.windStrength,
      seed: settings.windSeed,
    });
    camera.x = sim.state.x;
    camera.z = Math.max(sim.state.z, 30);
  }

  function toggleWind() {
    currentWindEnabled = !currentWindEnabled;
    sim.wind.enabled = currentWindEnabled;
    if (!currentWindEnabled) {
      sim.wind.accel = 0;
      sim.wind.baseAccel = 0;
    }
  }

  function startExplosion(state) {
    sim.status = "exploding";
    sim.statusTime = 0;
    sim.thrust = 0;
    sim.delta = 0;
    sim.explosion = createExplosion(state.x, Math.max(0, state.z));
  }

  function markSuccess() {
    sim.status = "success";
    sim.statusTime = 0;
    sim.thrust = 0;
    sim.delta = 0;
    sim.successEffect = createSuccessEffect(sim.state.x, Math.max(0, sim.state.z));
  }

  function evaluateOutcome(stepInfo) {
    const scenarioName = currentScenarioName;
    const target = TARGETS[scenarioName];
    const impactState = stepInfo.impactState ?? sim.state;

    if (stepInfo.groundContact && Math.abs(impactState.theta) > target.thetaTolerance) {
      logRocket("ground contact at bad angle -> explosion", { scenarioName, state: ascentDebugState(impactState, target) });
      startExplosion(impactState);
      return;
    }

    if (scenarioName === "ascent") {
      if (sim.state.z >= target.z) {
        logRocket("ascent threshold check", ascentDebugState(sim.state, target));
      }
      if (isAscentSuccess(sim.state, target)) {
        logRocket("ascent success -> congratulations", ascentDebugState(sim.state, target));
        markSuccess();
      } else if (isAscentAngleWrong(sim.state, target)) {
        logRocket("ascent wrong angle -> explosion", ascentDebugState(sim.state, target));
        startExplosion(sim.state);
      } else if (stepInfo.groundContact && sim.liftedOff) {
        logRocket("ascent ground contact after liftoff -> explosion", ascentDebugState(impactState, target));
        startExplosion(stepInfo.impactState ?? sim.state);
      }
      return;
    }

    if (!stepInfo.groundContact) return;

    if (isSoftLanding(impactState, target)) {
      logRocket("landing success -> congratulations", {
        x: Number(impactState.x.toFixed(2)),
        z: Number(impactState.z.toFixed(2)),
        speed: Number(Math.hypot(impactState.vx, impactState.vz).toFixed(2)),
        vz: Number(impactState.vz.toFixed(2)),
        thetaDeg: Number((impactState.theta * 180 / Math.PI).toFixed(2)),
      });
      markSuccess();
    } else {
      logRocket("landing failed -> explosion", {
        x: Number(impactState.x.toFixed(2)),
        z: Number(impactState.z.toFixed(2)),
        speed: Number(Math.hypot(impactState.vx, impactState.vz).toFixed(2)),
        vz: Number(impactState.vz.toFixed(2)),
        thetaDeg: Number((impactState.theta * 180 / Math.PI).toFixed(2)),
      });
      startExplosion(impactState);
    }
  }

  function clearKeys() {
    for (const key of Object.keys(keys)) keys[key] = false;
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
    canvas.focus({ preventScroll: true });
    disableNotebookKeyboard();
  }

  function releaseControls() {
    controlsActive = false;
    clearKeys();
    restoreNotebookKeyboard();
  }

  function claimKeyboardEvent(event) {
    if (!ROCKET_KEY_CODES.has(event.code)) return false;
    if (!controlsActive) return false;
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation?.();
    return true;
  }

  canvas.addEventListener("pointerdown", activateControls);
  document.addEventListener("pointerdown", (event) => {
    if (!canvas.contains(event.target)) releaseControls();
  }, { capture: true });

  window.addEventListener("keydown", (event) => {
    if (!claimKeyboardEvent(event)) return;
    logRocket("keydown", { code: event.code, scenarioName: currentScenarioName });
    if (event.code === "Escape") {
      releaseControls();
      canvas.blur();
      return;
    }
    if (event.code === "Space") {
      sim.paused = !sim.paused;
      return;
    }
    if (event.code === "KeyR") {
      reset();
      return;
    }
    if (event.code === "KeyW") {
      toggleWind();
      return;
    }
    if (event.code === "Digit1") {
      reset("ascent");
      return;
    }
    if (event.code === "Digit2") {
      reset("landing");
      return;
    }
    keys[event.code] = true;
  }, { capture: true });

  window.addEventListener("keyup", (event) => {
    if (!claimKeyboardEvent(event)) return;
    keys[event.code] = false;
  }, { capture: true });

  canvas.addEventListener("blur", releaseControls);
  window.addEventListener("blur", releaseControls);

  let lastTime = null;

  function frame(timestamp) {
    requestAnimationFrame(frame);
    resizeCanvasToDisplaySize(canvas);

    if (lastTime === null) {
      lastTime = timestamp;
      return;
    }

    const wallDt = (timestamp - lastTime) / 1000;
    lastTime = timestamp;
    const dt = Math.min(wallDt, 0.1);

    if (!sim.paused) {
      if (sim.status === "flying") {
        applyKeyboardControls(sim, keys, dt);
        const stepInfo = stepSimulation(sim, dt);
        evaluateOutcome(stepInfo);

        if (sim.status === "flying" && sim.thrust > 50) {
          spawnSmoke(sim.smoke, sim.state.x, sim.state.z, sim.state.theta, sim.delta);
        }
      } else {
        sim.statusTime += wallDt;
        if (sim.status === "exploding") {
          updateExplosion(sim.explosion, dt);
          if (sim.statusTime >= RESET_AFTER_CRASH) reset();
        } else if (sim.status === "success" && sim.statusTime >= RESET_AFTER_SUCCESS) {
          updateSuccessEffect(sim.successEffect, dt);
          reset();
        } else if (sim.status === "success") {
          updateSuccessEffect(sim.successEffect, dt);
        }
      }
      updateSmoke(sim.smoke, dt);
    }

    const activeTarget = TARGETS[currentScenarioName];
    const targetCameraX = sim.state.x * 0.72 + activeTarget.x * 0.28;
    const targetCameraZ = currentScenarioName === "ascent"
      ? Math.max(30, sim.state.z * 0.55 + activeTarget.z * 0.45)
      : Math.max(18, (sim.state.z + activeTarget.z) * 0.5 + 8);
    const lerpRate = 1 - Math.exp(-3 * dt);
    camera.x += (targetCameraX - camera.x) * lerpRate;
    camera.z += (targetCameraZ - camera.z) * lerpRate;

    const targetDz = Math.abs(sim.state.z - activeTarget.z);
    const metersH = currentScenarioName === "ascent"
      ? Math.max(90, Math.abs(sim.state.z - camera.z) * 2 + 70, targetDz * 1.25 + 55)
      : Math.max(95, targetDz * 1.35 + 45);
    const timeSeconds = timestamp / 1000;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawScene(ctx, canvas, camera, metersH, timeSeconds, sceneElements, currentScenarioName, activeTarget, sim.wind, dt);
    drawSmoke(ctx, canvas, camera, metersH, sim.smoke);
    if (sim.status !== "exploding") {
      drawRocket(ctx, canvas, camera, metersH, sim.state, sim.delta, sim.thrust);
    }
    drawExplosion(ctx, canvas, camera, metersH, sim.explosion);
    drawSuccessEffect(ctx, canvas, camera, metersH, sim.successEffect);
    drawHUD(ctx, canvas, sim);
    drawKeyHints(ctx, canvas);
  }

  requestAnimationFrame(frame);
}
