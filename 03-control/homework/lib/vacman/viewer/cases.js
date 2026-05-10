/** Load and validate open-case JSON for the viewer. */

export async function loadCases() {
  const url = new URL("../open_cases.json", import.meta.url).href;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(
      `Failed to load cases: ${res.status}. Serve 03-control/homework as the HTTP root so /lib and /assets are reachable.`,
    );
  }
  const data = await res.json();
  if (!Array.isArray(data)) throw new Error("open_cases.json must be a JSON array");
  return data;
}

export function validateCase(c) {
  if (!c || typeof c !== "object") throw new Error("Invalid case");
  if (!Array.isArray(c.paths)) throw new Error(`Case "${c.id || "?"}": paths must be an array`);
  for (let pi = 0; pi < c.paths.length; pi++) {
    const path = c.paths[pi];
    if (!Array.isArray(path)) throw new Error(`Case "${c.id || "?"}": paths[${pi}] must be an array`);
    for (let i = 0; i < path.length; i++) {
      const pt = path[i];
      if (!Array.isArray(pt) || pt.length < 2 || typeof pt[0] !== "number" || typeof pt[1] !== "number") {
        throw new Error(`Case "${c.id || "?"}": paths[${pi}][${i}] must be [x,z] numbers`);
      }
    }
  }
  if (!Array.isArray(c.base) || c.base.length < 2) throw new Error(`Case "${c.id || "?"}": base [x,z] required`);
  if (!Array.isArray(c.catman) || c.catman.length < 2) throw new Error(`Case "${c.id || "?"}": catman [x,z] required`);
  if (typeof c.base[0] !== "number" || typeof c.base[1] !== "number") {
    throw new Error(`Case "${c.id || "?"}": base x,z must be numbers`);
  }
  if (typeof c.catman[0] !== "number" || typeof c.catman[1] !== "number") {
    throw new Error(`Case "${c.id || "?"}": catman x,z must be numbers`);
  }
  if (c.catman_speed != null && (!Number.isFinite(c.catman_speed) || c.catman_speed <= 0)) {
    throw new Error(`Case "${c.id || "?"}": catman_speed must be a positive number`);
  }
}
