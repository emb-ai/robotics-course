export const PARAMS = {
  m: 50,
  I: 100,
  l: 1.5,
  g: 9.81,
  T_max: 981,
  T_min: 0,
  delta_max: 0.26,
};

export const ROCKET = {
  body_length: 5,
  body_width: 1,
};

export const THRUST_RATE = 200;
export const GIMBAL_RATE = 0.5;
export const PHYSICS_DT = 0.01;

export const DEFAULT_WIND = {
  enabled: true,
  strength: 2.2,
  seed: 98765,
  timeScale: 1.35,
  surfaceFadeStart: 0.25,
  surfaceFadeEnd: 8,
  centerOfPressureOffset: 1.1,
  rotationCoupling: 0.24,
};

export const SCENARIOS = {
  ascent: { x: 0, z: 0, vx: 0, vz: 0, theta: 0, omega: 0 },
  landing: { x: 18, z: 130, vx: -1.2, vz: -7, theta: 0, omega: 0 },
};

export const TARGETS = {
  ascent: {
    x: 0,
    z: 100,
    theta: Math.PI / 8,
    zTolerance: 4,
    thetaTolerance: Math.PI / 14,
  },
  landing: {
    x: 0,
    z: 0,
    width: 3.2,
    height: 0.35,
    xTolerance: 1.6,
    speedTolerance: 2.8,
    verticalSpeedTolerance: 2.4,
    thetaTolerance: 5 * Math.PI / 180,
    omegaTolerance: 0.65,
  },
};

// Planar rocket, z up. theta is pitch from vertical toward +x.
// Controls: thrust T and gimbal delta.
export function dynamics(state, thrust, delta, windAccel = 0, wind = DEFAULT_WIND) {
  const { vx, vz, theta, omega } = state;
  const { m, I, l, g } = PARAMS;
  const cpOffset = wind.centerOfPressureOffset ?? DEFAULT_WIND.centerOfPressureOffset;
  const coupling = wind.rotationCoupling ?? DEFAULT_WIND.rotationCoupling;
  const windOmegaRate = -coupling * (m / I) * windAccel * cpOffset * Math.cos(theta);

  return {
    vx,
    vz,
    ax: (thrust / m) * Math.sin(theta + delta) + windAccel,
    az: (thrust / m) * Math.cos(theta + delta) - g,
    thetaRate: omega,
    omegaRate: -(thrust * l / I) * Math.sin(delta) + windOmegaRate,
  };
}

export function cloneState(state) {
  return { ...state };
}

function addDerivative(state, derivative, scale) {
  return {
    x: state.x + derivative.vx * scale,
    z: state.z + derivative.vz * scale,
    vx: state.vx + derivative.ax * scale,
    vz: state.vz + derivative.az * scale,
    theta: state.theta + derivative.thetaRate * scale,
    omega: state.omega + derivative.omegaRate * scale,
  };
}

export function rk4Step(state, thrust, delta, dt, windAccel = 0, wind = DEFAULT_WIND) {
  const k1 = dynamics(state, thrust, delta, windAccel, wind);
  const s2 = addDerivative(state, k1, 0.5 * dt);
  const k2 = dynamics(s2, thrust, delta, windAccel, wind);
  const s3 = addDerivative(state, k2, 0.5 * dt);
  const k3 = dynamics(s3, thrust, delta, windAccel, wind);
  const s4 = addDerivative(state, k3, dt);
  const k4 = dynamics(s4, thrust, delta, windAccel, wind);

  return {
    x: state.x + (dt / 6) * (k1.vx + 2 * k2.vx + 2 * k3.vx + k4.vx),
    z: state.z + (dt / 6) * (k1.vz + 2 * k2.vz + 2 * k3.vz + k4.vz),
    vx: state.vx + (dt / 6) * (k1.ax + 2 * k2.ax + 2 * k3.ax + k4.ax),
    vz: state.vz + (dt / 6) * (k1.az + 2 * k2.az + 2 * k3.az + k4.az),
    theta: state.theta + (dt / 6) * (k1.thetaRate + 2 * k2.thetaRate + 2 * k3.thetaRate + k4.thetaRate),
    omega: state.omega + (dt / 6) * (k1.omegaRate + 2 * k2.omegaRate + 2 * k3.omegaRate + k4.omegaRate),
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function smoothstep(edge0, edge1, x) {
  if (edge0 === edge1) return x >= edge1 ? 1 : 0;
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function fade(t) {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function hashUint32(seed, index) {
  let h = (seed ^ Math.imul(index, 0x9e3779b9)) >>> 0;
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d) >>> 0;
  h ^= h >>> 15;
  h = Math.imul(h, 0x846ca68b) >>> 0;
  h ^= h >>> 16;
  return h >>> 0;
}

function gradient1D(seed, index) {
  return (hashUint32(seed, index) / 0xffffffff) * 2 - 1;
}

function perlin1D(seed, x) {
  const i0 = Math.floor(x);
  const t = x - i0;
  const g0 = gradient1D(seed, i0);
  const g1 = gradient1D(seed, i0 + 1);
  const v0 = g0 * t;
  const v1 = g1 * (t - 1);
  return lerp(v0, v1, fade(t)) * 2;
}

function windTimeOffset(seed) {
  return 3.5 + (hashUint32(seed, 0x51f15e) / 0xffffffff) * 19;
}

export function windSurfaceFactor(z, wind = DEFAULT_WIND) {
  const start = wind.surfaceFadeStart ?? DEFAULT_WIND.surfaceFadeStart;
  const end = wind.surfaceFadeEnd ?? DEFAULT_WIND.surfaceFadeEnd;
  return smoothstep(start, end, Math.max(0, z));
}

export function createWindState(config = {}) {
  const wind = { ...DEFAULT_WIND, ...config };
  return {
    enabled: wind.enabled,
    strength: wind.strength,
    seed: wind.seed,
    timeScale: wind.timeScale,
    surfaceFadeStart: wind.surfaceFadeStart,
    surfaceFadeEnd: wind.surfaceFadeEnd,
    centerOfPressureOffset: wind.centerOfPressureOffset,
    rotationCoupling: wind.rotationCoupling,
    time: 0,
    timeOffset: windTimeOffset(wind.seed),
    baseAccel: 0,
    accel: 0,
    surfaceFactor: 0,
  };
}

export function updateWind(wind, altitude, dt) {
  if (!wind?.enabled) {
    if (wind) {
      wind.accel = 0;
      wind.surfaceFactor = windSurfaceFactor(altitude, wind);
    }
    return;
  }

  wind.time += dt;
  const windTime = wind.time + wind.timeOffset;
  const primary = perlin1D(wind.seed, windTime / wind.timeScale);
  const secondary = perlin1D(wind.seed ^ 0x5bd1e995, windTime / (wind.timeScale * 0.52) + 19.7);
  wind.baseAccel = clamp((0.86 * primary + 0.18 * secondary) * 1.35, -1, 1) * wind.strength;
  wind.surfaceFactor = windSurfaceFactor(altitude, wind);
  wind.accel = wind.baseAccel * wind.surfaceFactor;
}

export function createRocketState(scenario = SCENARIOS.ascent, windConfig = {}) {
  return {
    state: cloneState(scenario),
    scenario,
    wind: createWindState(windConfig),
    elapsed: 0,
    thrust: 0,
    delta: 0,
    paused: false,
    smoke: [],
    explosion: null,
    successEffect: null,
    status: "flying",
    statusTime: 0,
    liftedOff: scenario.z > 1,
  };
}

export function applyKeyboardControls(sim, keys, dt) {
  if (keys.ArrowUp) {
    sim.thrust = Math.min(PARAMS.T_max, sim.thrust + THRUST_RATE * dt);
  }
  if (keys.ArrowDown) {
    sim.thrust = Math.max(PARAMS.T_min, sim.thrust - THRUST_RATE * dt);
  }

  if (keys.ArrowLeft) {
    sim.delta = Math.max(-PARAMS.delta_max, sim.delta - GIMBAL_RATE * dt);
  } else if (keys.ArrowRight) {
    sim.delta = Math.min(PARAMS.delta_max, sim.delta + GIMBAL_RATE * dt);
  } else if (sim.delta > 0) {
    sim.delta = Math.max(0, sim.delta - GIMBAL_RATE * dt);
  } else if (sim.delta < 0) {
    sim.delta = Math.min(0, sim.delta + GIMBAL_RATE * dt);
  }
}

export function stepSimulation(sim, dt) {
  const steps = Math.max(1, Math.round(dt / PHYSICS_DT));
  const stepDt = dt / steps;
  let groundContact = false;
  let impactState = null;

  sim.elapsed += dt;

  for (let i = 0; i < steps; i++) {
    updateWind(sim.wind, sim.state.z, stepDt);
    sim.state = rk4Step(sim.state, sim.thrust, sim.delta, stepDt, sim.wind?.accel ?? 0, sim.wind);

    if (sim.state.z > 1.0) sim.liftedOff = true;

    if (sim.state.z < 0) {
      groundContact = true;
      impactState = cloneState(sim.state);
      sim.state.z = 0;
      if (sim.state.vz < 0) sim.state.vz = 0;
    }
  }

  return { groundContact, impactState };
}

export function rot2D(x, z, angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [x * c - z * s, x * s + z * c];
}

export function bodyToWorldOffset(x, z, theta) {
  return rot2D(x, z, -theta);
}

export function gimbalLocalToWorldOffset(x, z, delta, theta) {
  const pivotZ = -ROCKET.body_length * 0.35;
  const [bodyX, bodyZ] = rot2D(x, z, delta);
  return bodyToWorldOffset(bodyX, bodyZ + pivotZ, theta);
}

export function spawnSmoke(smoke, x, z, theta, delta) {
  const nozzleH = ROCKET.body_length * 0.10;
  const [dx, dz] = gimbalLocalToWorldOffset(0, -nozzleH, delta, theta);

  smoke.push({
    x: x + dx,
    z: z + dz,
    vx: (Math.random() - 0.5) * 0.5,
    vz: (Math.random() - 0.5) * 0.5 - 0.3,
    opacity: 0.6,
    radius: 0.2 + Math.random() * 0.15,
  });
}

export function updateSmoke(smoke, dt) {
  for (let i = smoke.length - 1; i >= 0; i--) {
    const p = smoke[i];
    p.x += p.vx * dt;
    p.z += p.vz * dt;
    p.radius += 0.3 * dt;
    p.opacity -= 0.4 * dt;
    if (p.opacity < 0.01) smoke.splice(i, 1);
  }
}

export function createExplosion(x, z, count = 128) {
  const particles = [];
  for (let i = 0; i < count; i++) {
    const a = Math.random() * Math.PI * 2;
    const speed = 4 + Math.random() * 18;
    const isSmoke = Math.random() < 0.34;
    const isSpark = !isSmoke && Math.random() < 0.36;
    const ttl = isSmoke ? 1.3 + Math.random() * 0.75 : isSpark ? 0.55 + Math.random() * 0.38 : 0.8 + Math.random() * 0.45;
    particles.push({
      x,
      z,
      vx: Math.cos(a) * speed * (isSmoke ? 0.45 : 1),
      vz: Math.sin(a) * speed * (isSmoke ? 0.32 : 1) + (isSmoke ? 1.8 : 4.5),
      radius: isSmoke ? 0.35 + Math.random() * 0.55 : isSpark ? 0.055 + Math.random() * 0.09 : 0.18 + Math.random() * 0.38,
      life: ttl,
      ttl,
      type: isSmoke ? "smoke" : isSpark ? "spark" : "fire",
      shape: isSmoke ? "square" : Math.random() < 0.5 ? "triangle" : "square",
      angle: Math.random() * Math.PI * 2,
      spin: -9 + Math.random() * 18,
      color: isSmoke ? "#5f5f63" : Math.random() < 0.35 ? "#fff3a0" : Math.random() < 0.72 ? "#ff8f24" : "#ff2b20",
    });
  }
  const debrisKinds = [
    { type: "body", length: 1.2, width: 0.32, color: "#dce3ef" },
    { type: "body", length: 0.9, width: 0.28, color: "#aab4c3" },
    { type: "fin", length: 0.65, width: 0.48, color: "#ff8f24" },
    { type: "fin", length: 0.55, width: 0.38, color: "#ff5a2a" },
    { type: "nose", length: 0.72, width: 0.42, color: "#f2f6ff" },
    { type: "engine", length: 0.5, width: 0.34, color: "#5c6370" },
  ];
  const debris = Array.from({ length: 28 }, (_, i) => {
    const a = -Math.PI * 0.05 + Math.random() * Math.PI * 1.1;
    const speed = 2.5 + Math.random() * 12;
    const ttl = 1.15 + Math.random() * 0.85;
    const kind = debrisKinds[i] ?? {
      type: "shard",
      length: 0.28 + Math.random() * 0.58,
      width: 0.08 + Math.random() * 0.14,
      color: Math.random() < 0.55 ? "#c7ccd6" : "#858b96",
    };
    return {
      x,
      z,
      vx: Math.cos(a) * speed,
      vz: Math.sin(a) * speed + 3.2,
      angle: Math.random() * Math.PI * 2,
      spin: -8 + Math.random() * 16,
      type: kind.type,
      length: kind.length,
      width: kind.width,
      life: ttl,
      ttl,
      color: kind.color,
    };
  });
  return {
    x,
    z,
    age: 0,
    particles,
    debris,
  };
}

export function updateExplosion(explosion, dt) {
  if (!explosion) return false;
  explosion.age += dt;

  for (const p of explosion.particles) {
    p.x += p.vx * dt;
    p.z += p.vz * dt;
    p.vx *= Math.max(0, 1 - (p.type === "smoke" ? 0.65 : 0.18) * dt);
    p.vz -= (p.type === "smoke" ? 2.0 : 11) * dt;
    p.radius += (p.type === "smoke" ? 0.95 : 0.45) * dt;
    p.angle += p.spin * dt;
    p.life -= dt;
  }

  for (const d of explosion.debris) {
    d.x += d.vx * dt;
    d.z += d.vz * dt;
    d.vz -= 10.5 * dt;
    d.angle += d.spin * dt;
    d.life -= dt;
  }

  return explosion.particles.some((p) => p.life > 0) || explosion.debris.some((d) => d.life > 0);
}

export function createSuccessEffect(x, z, count = 70) {
  const particles = [];
  for (let i = 0; i < count; i++) {
    const a = Math.random() * Math.PI * 2;
    const speed = 1.4 + Math.random() * 5.2;
    const ttl = 0.9 + Math.random() * 0.7;
    particles.push({
      x,
      z,
      vx: Math.cos(a) * speed,
      vz: Math.sin(a) * speed + 2.4,
      radius: 0.08 + Math.random() * 0.18,
      life: ttl,
      ttl,
      color: Math.random() < 0.55 ? "#3dff8c" : Math.random() < 0.78 ? "#beffdc" : "#ffd85a",
    });
  }
  return { x, z, age: 0, particles };
}

export function updateSuccessEffect(effect, dt) {
  if (!effect) return false;
  effect.age += dt;

  for (const p of effect.particles) {
    p.x += p.vx * dt;
    p.z += p.vz * dt;
    p.vx *= 1 - 0.45 * dt;
    p.vz -= 4.5 * dt;
    p.radius += 0.12 * dt;
    p.life -= dt;
  }

  return effect.particles.some((p) => p.life > 0);
}
