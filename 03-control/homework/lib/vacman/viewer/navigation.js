/**
 * World bounds, wall geometry, Minkowski C-space obstacles, visibility graph, A*,
 * and vacuum / cat collision against thick wall segments.
 */
import { ARENA_PAD, CAT_OCT_R, EPS, PATH_CLOSE_EPS, WALL_HALF_T } from "./constants.js";
import {
  dist,
  minkowskiSumConvex,
  octagonVertices,
  pointInConvex,
  segmentAabb,
  segmentToRect,
  segmentVisible,
  aabbOverlap,
  polyAabb,
  closestPointOnSeg,
  pointInConvexStrict,
} from "./math2d.js";

export function bboxFromCase(caseData) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  function addPt(x, z) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minZ = Math.min(minZ, z);
    maxZ = Math.max(maxZ, z);
  }
  for (const path of caseData.paths || []) {
    for (const pt of path) addPt(pt[0], pt[1]);
  }
  if (caseData.base) addPt(caseData.base[0], caseData.base[1]);
  if (caseData.catman) addPt(caseData.catman[0], caseData.catman[1]);
  return { minX, maxX, minZ, maxZ };
}

export function buildWorldTransform(caseData) {
  const b = bboxFromCase(caseData);
  const cx = (b.minX + b.maxX) / 2;
  const cz = (b.minZ + b.maxZ) / 2;
  const w = b.maxX - b.minX + 2 * ARENA_PAD;
  const d = b.maxZ - b.minZ + 2 * ARENA_PAD;
  const halfX = w / 2;
  const halfZ = d / 2;
  const toWorld = (jx, jz) => ({ x: cx - jx, z: jz - cz });
  const toWorldHeading = (theta) => Math.PI - theta;
  return { halfX, halfZ, cx, cz, toWorld, toWorldHeading };
}

export function collectWallRects(caseData, toWorld) {
  const rects = [];
  for (const path of caseData.paths || []) {
    if (!path || path.length < 2) continue;
    const n = path.length;
    for (let i = 0; i < n - 1; i++) {
      const p0 = toWorld(path[i][0], path[i][1]);
      const p1 = toWorld(path[i + 1][0], path[i + 1][1]);
      const r = segmentToRect(p0, p1, WALL_HALF_T);
      if (r) rects.push(r);
    }
    const pFirst = toWorld(path[0][0], path[0][1]);
    const pLast = toWorld(path[n - 1][0], path[n - 1][1]);
    if (dist(pFirst, pLast) > PATH_CLOSE_EPS) {
      const r = segmentToRect(pLast, pFirst, WALL_HALF_T);
      if (r) rects.push(r);
    }
  }
  return rects;
}

export function buildCspaceObstacles(wallRects, octNeg) {
  const obs = [];
  for (const rect of wallRects) {
    const hull = minkowskiSumConvex(rect, octNeg);
    if (hull.length >= 3) obs.push(hull);
  }
  return obs;
}

export function mergeCloseNodes(nodes, eps = 0.08) {
  const out = [];
  for (const p of nodes) {
    let dup = false;
    for (const q of out) {
      if (dist(p, q) < eps) {
        dup = true;
        break;
      }
    }
    if (!dup) out.push({ x: p.x, z: p.z });
  }
  return out;
}

export function buildVisibilityGraph(nodes, obstacles) {
  const n = nodes.length;
  const adj = Array.from({ length: n }, () => []);
  const obsBoxes = obstacles.map(polyAabb);
  for (let i = 0; i < n; i++) {
    const ai = nodes[i];
    for (let j = i + 1; j < n; j++) {
      const aj = nodes[j];
      const segBox = segmentAabb(ai.x, ai.z, aj.x, aj.z);
      let maybeBlocked = false;
      for (let oi = 0; oi < obstacles.length; oi++) {
        if (aabbOverlap(segBox, obsBoxes[oi])) {
          maybeBlocked = true;
          break;
        }
      }
      const w = dist(ai, aj);
      if (!maybeBlocked || segmentVisible(ai, aj, obstacles)) {
        adj[i].push({ j, w });
        adj[j].push({ j: i, w });
      }
    }
  }
  return adj;
}

export function pointFreeInObstacles(p, obstacles) {
  for (const poly of obstacles) {
    if (pointInConvexStrict(p.x, p.z, poly)) return false;
  }
  return true;
}

export function astar(nodes, adj, startIdx, goalIdx) {
  const n = nodes.length;
  const g = new Float64Array(n).fill(Infinity);
  const f = new Float64Array(n).fill(Infinity);
  const parent = new Int32Array(n).fill(-1);
  const open = new Set([startIdx]);
  g[startIdx] = 0;
  f[startIdx] = dist(nodes[startIdx], nodes[goalIdx]);

  while (open.size) {
    let u = -1;
    let best = Infinity;
    for (const i of open) {
      if (f[i] < best) {
        best = f[i];
        u = i;
      }
    }
    if (u < 0) break;
    open.delete(u);
    if (u === goalIdx) break;
    for (const e of adj[u]) {
      const ng = g[u] + e.w;
      if (ng < g[e.j]) {
        g[e.j] = ng;
        parent[e.j] = u;
        f[e.j] = ng + dist(nodes[e.j], nodes[goalIdx]);
        open.add(e.j);
      }
    }
  }
  if (parent[goalIdx] < 0 && goalIdx !== startIdx) return null;
  const path = [];
  let cur = goalIdx;
  const maxSteps = nodes.length + 2;
  const sx = nodes[startIdx].x;
  const sz = nodes[startIdx].z;
  for (let step = 0; step < maxSteps; step++) {
    if (cur < 0) return null;
    path.push(nodes[cur]);
    if (cur === startIdx) break;
    cur = parent[cur];
  }
  const last = path[path.length - 1];
  if (!last || dist(last, { x: sx, z: sz }) > 1e-6) return null;
  path.reverse();
  return path;
}

export function pushOutOfWalls(px, pz, wallRects, radius) {
  let x = px;
  let z = pz;
  const passes = 4;
  for (let pass = 0; pass < passes; pass++) {
    for (const rect of wallRects) {
      const n = rect.length;
      for (let i = 0; i < n; i++) {
        const a = rect[i];
        const b = rect[(i + 1) % n];
        const q = closestPointOnSeg(x, z, a, b);
        const d = Math.hypot(x - q.x, z - q.z);
        const need = radius;
        if (d < need && d > EPS) {
          const s = need / d;
          x = q.x + (x - q.x) * s;
          z = q.z + (z - q.z) * s;
        } else if (d <= EPS && pointInConvex(x, z, rect)) {
          const cx = rect.reduce((s_, v) => s_ + v.x, 0) / n;
          const cz = rect.reduce((s_, v) => s_ + v.z, 0) / n;
          const dx = x - cx;
          const dz = z - cz;
          const L = Math.hypot(dx, dz) + EPS;
          x += (dx / L) * need;
          z += (dz / L) * need;
        }
      }
    }
  }
  return { x, z };
}

export function cellFreeFromWalls(wx, wz, wallRects) {
  for (const rect of wallRects) {
    if (pointInConvex(wx, wz, rect)) return false;
  }
  return true;
}

export function makeCatCspaceContext(wallRects) {
  const oct = octagonVertices(CAT_OCT_R);
  const octNeg = oct.map((p) => ({ x: -p.x, z: -p.z }));
  const collisionObstacles = buildCspaceObstacles(wallRects, octNeg);
  const graphNodes = mergeCloseNodes(
    collisionObstacles
      .flatMap((poly) => poly.map((v) => ({ x: v.x, z: v.z })))
      .filter((p) => pointFreeInObstacles(p, collisionObstacles)),
  );
  const visAdj = buildVisibilityGraph(graphNodes, collisionObstacles);
  return {
    collisionObstacles,
    planningObstacles: collisionObstacles,
    cspaceObstacles: collisionObstacles,
    graphNodes,
    visAdj,
  };
}
