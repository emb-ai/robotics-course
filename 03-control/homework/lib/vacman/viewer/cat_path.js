/**
 * CatMan pursuit: dynamic start/goal + A* on a visibility graph.
 *
 * The graph is built in CatMan C-space: wall geometry expanded by the CatMan
 * octagonal footprint, with the current CatMan and chase-target positions added
 * as dynamic nodes. All edge weights are Euclidean segment lengths.
 */
import { clonePz } from "./math2d.js";

const TARGET_REPLAN_R = 0.12;
const WAYPOINT_BLOCKED_EPS = 1e-4;

export function createCatPlanner({
  graphNodes,
  visAdj,
  planningObstacles,
  collisionObstacles,
  cspaceObstacles,
  segmentVisible,
  dist,
  astar,
}) {
  const visibilityObstacles = cspaceObstacles ?? collisionObstacles ?? planningObstacles ?? [];
  const safetyObstacles = visibilityObstacles;
  const cat = { path: null, wpIdx: 0, goal: null, partial: false, blockedAtWaypoint: false };

  function visible(a, b, obstacles = visibilityObstacles) {
    return segmentVisible(a, b, obstacles);
  }

  function moveVisible(a, b) {
    return segmentVisible(a, b, safetyObstacles);
  }

  function setPathFromWaypoints(waypoints, goal = null, partial = false) {
    cat.path = waypoints.map(clonePz);
    if (cat.path.length === 1) cat.path.push(clonePz(cat.path[0]));
    cat.wpIdx = Math.min(1, Math.max(0, cat.path.length - 1));
    cat.goal = goal ? clonePz(goal) : (cat.path.length ? clonePz(cat.path[cat.path.length - 1]) : null);
    cat.partial = partial;
    cat.blockedAtWaypoint = false;
  }

  function connectVisibleNode(nodes, adj, idx, count) {
    for (let i = 0; i < count; i++) {
      if (visible(nodes[idx], nodes[i])) {
        const w = dist(nodes[idx], nodes[i]);
        adj[idx].push({ j: i, w });
        adj[i].push({ j: idx, w });
      }
    }
  }

  function reachableFrom(adj, startIdx) {
    const seen = new Uint8Array(adj.length);
    const stack = [startIdx];
    seen[startIdx] = 1;
    while (stack.length) {
      const u = stack.pop();
      for (const e of adj[u]) {
        if (seen[e.j]) continue;
        seen[e.j] = 1;
        stack.push(e.j);
      }
    }
    return seen;
  }

  function smoothPath(path) {
    if (!path || path.length <= 2) return path;
    const out = [path[0]];
    let i = 0;
    while (i < path.length - 1) {
      let next = i + 1;
      for (let j = path.length - 1; j > i + 1; j--) {
        if (moveVisible(path[i], path[j])) {
          next = j;
          break;
        }
      }
      out.push(path[next]);
      i = next;
    }
    return out;
  }

  function planCatPath(agent, target) {
    const start = { x: agent.x, z: agent.z };
    const goal = { x: target.x, z: target.z };
    const nodes = graphNodes.map((n) => ({ x: n.x, z: n.z }));
    const si = nodes.length;
    const gi = nodes.length + 1;
    nodes.push(start, goal);
    const adj = visAdj.map((row) => row.map((e) => ({ j: e.j, w: e.w }))).concat([[], []]);
    connectVisibleNode(nodes, adj, si, si);
    connectVisibleNode(nodes, adj, gi, gi);
    const goalReach = reachableFrom(adj, gi);

    function fallbackTowardVisible() {
      if (visible(start, goal)) {
        setPathFromWaypoints([start, goal], goal, false);
        return;
      }
      let bestK = -1;
      let bestD = Infinity;
      for (const requireGoalReach of [true, false]) {
        for (let k = 0; k < graphNodes.length; k++) {
          if (requireGoalReach && !goalReach[k]) continue;
          const nk = graphNodes[k];
          if (!visible(start, nk)) continue;
          const dk = dist(start, nk) + dist(nk, goal);
          if (dk < bestD) {
            bestD = dk;
            bestK = k;
          }
        }
        if (bestK >= 0) break;
      }
      if (bestK >= 0) {
        setPathFromWaypoints([start, graphNodes[bestK]], goal, true);
        return;
      }
      setPathFromWaypoints([start, start], goal, true);
    }

    if (dist(start, goal) < 0.4) {
      if (visible(start, goal)) {
        setPathFromWaypoints([start, goal], goal, false);
      } else {
        fallbackTowardVisible();
      }
      return;
    }

    if (adj[si].length === 0) {
      fallbackTowardVisible();
      return;
    }

    const path = astar(nodes, adj, si, gi);
    const pathOk =
      path &&
      path.length >= 2 &&
      dist(path[0], start) < 0.25 &&
      dist(path[path.length - 1], goal) < 0.25;
    if (pathOk) {
      setPathFromWaypoints(smoothPath(path), goal, false);
    } else {
      fallbackTowardVisible();
    }
  }

  function advanceWaypoints(agent, waypointRadius) {
    cat.blockedAtWaypoint = false;
    while (cat.path && cat.wpIdx < cat.path.length) {
      const wp = cat.path[cat.wpIdx];
      const d = dist(agent, wp);
      if (d > waypointRadius) break;
      const nextIdx = cat.wpIdx + 1;
      if (nextIdx < cat.path.length && !moveVisible({ x: agent.x, z: agent.z }, cat.path[nextIdx])) {
        cat.blockedAtWaypoint = d <= Math.min(WAYPOINT_BLOCKED_EPS, waypointRadius * 0.25);
        break;
      }
      cat.wpIdx++;
    }
  }

  function currentWaypoint() {
    if (!cat.path || cat.wpIdx >= cat.path.length) return null;
    return cat.path[cat.wpIdx];
  }

  function pathUsable(agent) {
    if (cat.blockedAtWaypoint) return false;
    const wp = currentWaypoint();
    if (!wp) return false;
    return moveVisible({ x: agent.x, z: agent.z }, wp);
  }

  function targetMoved(target, threshold = TARGET_REPLAN_R) {
    return cat.partial || !cat.goal || dist(cat.goal, target) > threshold;
  }

  function shortcutPath(agent, target) {
    if (!cat.path || cat.wpIdx >= cat.path.length) return false;
    const start = { x: agent.x, z: agent.z };
    const goal = { x: target.x, z: target.z };
    if (moveVisible(start, goal)) {
      setPathFromWaypoints([start, goal], goal, false);
      return true;
    }
    let best = cat.wpIdx;
    for (let i = cat.path.length - 1; i > cat.wpIdx; i--) {
      if (moveVisible(start, cat.path[i])) {
        best = i;
        break;
      }
    }
    cat.wpIdx = best;
    return pathUsable(agent);
  }

  function movementSafe(from, to) {
    return moveVisible(from, to);
  }

  function resetPath() {
    cat.path = null;
    cat.wpIdx = 0;
    cat.goal = null;
    cat.partial = false;
    cat.blockedAtWaypoint = false;
  }

  return {
    planCatPath,
    advanceWaypoints,
    currentWaypoint,
    pathUsable,
    targetMoved,
    shortcutPath,
    movementSafe,
    resetPath,
    cat,
  };
}
