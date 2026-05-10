/**
 * Shared planar unicycle on the (x, z) ground plane:
 *   ẋ = v cos θ,  ż = v sin θ,  θ̇ = ω
 *
 * VacMan: twist (v, ω) from differential-drive wheel speeds (see controls.js).
 * CatMan: point toward target, then advance at fixed v with ω = 0 (heading reset each step).
 */
import { AXLE, SPEED } from "./constants.js";

/** Kinematics: v = (vL + vR) / 2,  ω = (vR − vL) / L */
export function differentialDriveTwist(vL, vR, axleLength = AXLE) {
  return {
    v: (vL + vR) / 2,
    omega: (vR - vL) / axleLength,
  };
}

/**
 * Single integration step for both agents (in-place).
 * @param {{ x: number, z: number, th: number }} agent
 * @param {number} dt
 * @param {{ v: number, omega: number }} twist linear speed in heading direction, yaw rate
 */
export function stepPlanarUnicycle(agent, dt, { v, omega }) {
  agent.x += v * Math.cos(agent.th) * dt;
  agent.z += v * Math.sin(agent.th) * dt;
  agent.th += omega * dt;
}

/** CatMan cruise speed (scalar forward in current heading). */
export function catmanLinearSpeed(caseSpeed = SPEED) {
  return Number.isFinite(caseSpeed) && caseSpeed > 0 ? caseSpeed : SPEED;
}
