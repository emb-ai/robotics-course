import { EPS } from "./constants.js";

export const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export function cross(ax, az, bx, bz) {
  return ax * bz - az * bx;
}

export function dist(a, b) {
  return Math.hypot(a.x - b.x, a.z - b.z);
}

export function convexHull(points) {
  const pts = points.slice().sort((p, q) => (p.x !== q.x ? p.x - q.x : p.z - q.z));
  if (pts.length < 3) return pts;
  const cross2 = (o, a, b) => cross(a.x - o.x, a.z - o.z, b.x - o.x, b.z - o.z);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross2(lower[lower.length - 2], lower[lower.length - 1], p) <= EPS) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross2(upper[upper.length - 2], upper[upper.length - 1], p) <= EPS) upper.pop();
    upper.push(p);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

export function pointInConvex(px, pz, poly) {
  const n = poly.length;
  if (n < 3) return false;
  let sign = 0;
  for (let i = 0; i < n; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % n];
    const c = cross(b.x - a.x, b.z - a.z, px - a.x, pz - a.z);
    if (Math.abs(c) < EPS) continue;
    if (sign === 0) sign = c > 0 ? 1 : -1;
    else if ((c > 0 ? 1 : -1) !== sign) return false;
  }
  return true;
}

export function pointInConvexStrict(px, pz, poly) {
  const n = poly.length;
  if (n < 3) return false;
  let sign = 0;
  for (let i = 0; i < n; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % n];
    const c = cross(b.x - a.x, b.z - a.z, px - a.x, pz - a.z);
    if (Math.abs(c) <= EPS) return false;
    if (sign === 0) sign = c > 0 ? 1 : -1;
    else if ((c > 0 ? 1 : -1) !== sign) return false;
  }
  return true;
}

/** Strict crossing (interior to both segments); avoids false blocks at shared obstacle corners. */
export function segIntersectProper(a, b, c, d) {
  const o1 = cross(b.x - a.x, b.z - a.z, c.x - a.x, c.z - a.z);
  const o2 = cross(b.x - a.x, b.z - a.z, d.x - a.x, d.z - a.z);
  const o3 = cross(d.x - c.x, d.z - c.z, a.x - c.x, a.z - c.z);
  const o4 = cross(d.x - c.x, d.z - c.z, b.x - c.x, b.z - c.z);
  return (o1 > EPS && o2 < -EPS || o1 < -EPS && o2 > EPS) && (o3 > EPS && o4 < -EPS || o3 < -EPS && o4 > EPS);
}

export function pointOnSegment(px, pz, a, b) {
  const abx = b.x - a.x;
  const abz = b.z - a.z;
  const apx = px - a.x;
  const apz = pz - a.z;
  if (Math.abs(cross(abx, abz, apx, apz)) > EPS) return false;
  const dot = apx * abx + apz * abz;
  if (dot < -EPS) return false;
  const len2 = abx * abx + abz * abz;
  return dot <= len2 + EPS;
}

function segmentParam(px, pz, a, b) {
  const dx = b.x - a.x;
  const dz = b.z - a.z;
  const len2 = dx * dx + dz * dz;
  if (len2 <= EPS) return 0;
  return ((px - a.x) * dx + (pz - a.z) * dz) / len2;
}

function addUniqueParam(params, t) {
  if (t <= EPS || t >= 1 - EPS) return;
  if (!params.some((u) => Math.abs(u - t) <= 1e-6)) params.push(t);
}

export function segmentAabb(ax, az, bx, bz) {
  const pad = 0.02;
  return {
    minX: Math.min(ax, bx) - pad,
    maxX: Math.max(ax, bx) + pad,
    minZ: Math.min(az, bz) - pad,
    maxZ: Math.max(az, bz) + pad,
  };
}

export function polyAabb(poly) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (const p of poly) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minZ = Math.min(minZ, p.z);
    maxZ = Math.max(maxZ, p.z);
  }
  return { minX, maxX, minZ, maxZ };
}

export function aabbOverlap(A, B) {
  return A.minX <= B.maxX && A.maxX >= B.minX && A.minZ <= B.maxZ && A.maxZ >= B.minZ;
}

export function segmentVisible(a, b, obstacles) {
  if (obstacles.length === 0) return true;
  const sa = segmentAabb(a.x, a.z, b.x, b.z);
  const obsBoxes = obstacles.map(polyAabb);
  for (let oi = 0; oi < obstacles.length; oi++) {
    if (!aabbOverlap(sa, obsBoxes[oi])) continue;
    const poly = obstacles[oi];
    const n = poly.length;
    if (pointInConvexStrict(a.x, a.z, poly) || pointInConvexStrict(b.x, b.z, poly)) return false;

    const touchParams = [];
    for (let i = 0; i < n; i++) {
      const p1 = poly[i];
      const p2 = poly[(i + 1) % n];
      if (segIntersectProper(a, b, p1, p2)) return false;
      if (pointOnSegment(p1.x, p1.z, a, b)) addUniqueParam(touchParams, segmentParam(p1.x, p1.z, a, b));
      if (pointOnSegment(a.x, a.z, p1, p2)) addUniqueParam(touchParams, 0);
      if (pointOnSegment(b.x, b.z, p1, p2)) addUniqueParam(touchParams, 1);
    }

    touchParams.sort((u, v) => u - v);
    const params = [0, ...touchParams, 1];
    for (let i = 0; i < params.length - 1; i++) {
      if (params[i + 1] - params[i] <= 1e-6) continue;
      const t = (params[i] + params[i + 1]) / 2;
      const mx = a.x + t * (b.x - a.x);
      const mz = a.z + t * (b.z - a.z);
      if (pointInConvexStrict(mx, mz, poly)) return false;
    }
  }
  return true;
}

export function octagonVertices(r) {
  const v = [];
  for (let i = 0; i < 8; i++) {
    const ang = Math.PI / 8 + (i * Math.PI) / 4;
    v.push({ x: r * Math.cos(ang), z: r * Math.sin(ang) });
  }
  return v;
}

export function segmentToRect(p0, p1, halfT) {
  const dx = p1.x - p0.x;
  const dz = p1.z - p0.z;
  const L = Math.hypot(dx, dz);
  if (L < EPS) return null;
  const nx = (-dz / L) * halfT;
  const nz = (dx / L) * halfT;
  return [
    { x: p0.x + nx, z: p0.z + nz },
    { x: p0.x - nx, z: p0.z - nz },
    { x: p1.x - nx, z: p1.z - nz },
    { x: p1.x + nx, z: p1.z + nz },
  ];
}

export function minkowskiSumConvex(polyA, polyB) {
  const pts = [];
  for (const a of polyA) {
    for (const b of polyB) pts.push({ x: a.x + b.x, z: a.z + b.z });
  }
  return convexHull(pts);
}

export function clonePz(p) {
  return { x: p.x, z: p.z };
}

export function closestPointOnSeg(px, pz, a, b) {
  const abx = b.x - a.x;
  const abz = b.z - a.z;
  const t = clamp(((px - a.x) * abx + (pz - a.z) * abz) / (abx * abx + abz * abz + EPS), 0, 1);
  return { x: a.x + t * abx, z: a.z + t * abz };
}
