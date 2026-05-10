/**
 * HUD, wheel-throttle bars, win/lose overlay. Keyboard state comes from controls.js.
 */
import { SPEED } from "./constants.js";
import { clamp } from "./math2d.js";
import { createKeyboardControls } from "./controls.js?v=keyboard-focus";

export function createGameUi(container, keyboardFocusTarget = container) {
  const monoFont = "'Courier New', Courier, 'SF Mono', Consolas, monospace";
  const hud = document.createElement("div");
  hud.style.cssText = `position:absolute;top:0;left:0;width:100%;padding:14px 20px;
    display:flex;justify-content:space-between;align-items:flex-start;
    pointer-events:none;font:14px/1.45 ${monoFont};
    color:#e0e0e0;text-shadow:0 1px 3px #000;z-index:10;`;
  container.appendChild(hud);

  const hudL = document.createElement("div");
  hudL.style.cssText = `min-width:260px;background:rgba(0,0,0,0.84);border:1px solid rgba(61,255,140,0.35);
    border-radius:4px;padding:10px 12px;color:#e8fff0;display:flex;align-items:flex-start;gap:14px;`;
  const hudStats = document.createElement("div");
  hudStats.style.cssText = "white-space:pre;";
  hudL.appendChild(hudStats);
  const hudC = document.createElement("div");
  hudC.style.cssText = "display:none;";
  const hudRWrap = document.createElement("div");
  hudRWrap.style.cssText = "text-align:right;display:flex;flex-direction:column;align-items:flex-end;gap:8px;";
  const caseToolbar = document.createElement("div");
  caseToolbar.style.cssText = "pointer-events:all;display:flex;align-items:center;gap:8px;";
  const caseLabel = document.createElement("label");
  caseLabel.textContent = "Scene";
  caseLabel.style.cssText = `font:12px/1.2 ${monoFont};color:#b7c2cc;text-transform:uppercase;`;
  const caseSelect = document.createElement("select");
  caseSelect.style.cssText = `padding:6px 10px;border-radius:4px;border:1px solid rgba(61,255,140,0.35);
    background:rgba(0,0,0,0.88);color:#e8fff0;font:12px/1.2 ${monoFont};`;
  caseToolbar.append(caseLabel, caseSelect);
  const hudR = document.createElement("div");
  hudRWrap.append(caseToolbar, hudR);
  hud.append(hudL, hudC, hudRWrap);

  const keyHints = document.createElement("div");
  keyHints.style.cssText = `position:absolute;left:0;bottom:0;width:100%;min-height:30px;padding:7px 14px;
    display:flex;align-items:center;justify-content:center;text-align:center;
    background:rgba(0,0,0,0.50);border-top:1px solid rgba(61,255,140,0.25);
    pointer-events:none;font:12px/1.35 ${monoFont};color:rgba(232,255,240,0.78);
    text-shadow:0 1px 3px #000;z-index:10;`;
  keyHints.textContent = "W/↑ FORWARD   S/↓ REVERSE   A/← TURN LEFT   D/→ TURN RIGHT   R RESET   SCENE MENU SWITCHES MAP";
  container.appendChild(keyHints);

  const overlay = document.createElement("div");
  overlay.style.cssText = `position:absolute;top:0;left:0;width:100%;height:100%;
    display:none;align-items:center;justify-content:center;flex-direction:column;
    background:rgba(0,0,0,0.72);z-index:20;pointer-events:all;`;
  container.appendChild(overlay);
  const oText = document.createElement("div");
  oText.style.cssText = `font:16px/1.45 ${monoFont};color:#fff;text-align:center;`;
  overlay.appendChild(oText);
  const oBtn = document.createElement("button");
  oBtn.textContent = "Play Again  (R)";
  oBtn.style.cssText = `margin-top:18px;padding:10px 28px;font:bold 16px/1.2 ${monoFont};
    border:1px solid rgba(190,255,220,0.85);border-radius:4px;cursor:pointer;background:#3dff8c;color:#061008;`;
  overlay.appendChild(oBtn);

  const barWrap = document.createElement("div");
  barWrap.style.cssText = `display:flex;gap:12px;align-items:flex-end;margin-left:auto;pointer-events:none;`;
  hudL.appendChild(barWrap);

  function makeBar(label) {
    const col = document.createElement("div");
    col.style.cssText = "display:flex;flex-direction:column;align-items:center;";
    const outer = document.createElement("div");
    outer.style.cssText =
      "width:14px;height:56px;background:#060b10;border:1px solid rgba(61,255,140,0.35);border-radius:2px;position:relative;overflow:hidden;";
    const fill = document.createElement("div");
    fill.style.cssText =
      "position:absolute;left:0;width:100%;background:#00e5ff;transition:height .06s,bottom .06s,top .06s;";
    outer.appendChild(fill);
    const lbl = document.createElement("div");
    lbl.textContent = label;
    lbl.style.cssText = `font:10px/1.2 ${monoFont};color:#9fb2aa;margin-top:3px;`;
    col.append(outer, lbl);
    return { el: col, fill };
  }
  const barL = makeBar("L");
  const barR = makeBar("R");
  barWrap.append(barL.el, barR.el);

  function setBar(bar, v) {
    const maxV = SPEED * 1.5;
    const frac = clamp(v / maxV, -1, 1);
    const f = bar.fill;
    if (frac >= 0) {
      f.style.bottom = "50%";
      f.style.top = "";
      f.style.height = frac * 50 + "%";
      f.style.background = "#00e5ff";
    } else {
      f.style.top = "50%";
      f.style.bottom = "";
      f.style.height = -frac * 50 + "%";
      f.style.background = "#ff5252";
    }
  }

  const keyboard = createKeyboardControls(window, keyboardFocusTarget, container);

  return {
    hudL,
    hudStats,
    hudC,
    hudR,
    overlay,
    oText,
    oBtn,
    caseSelect,
    barL,
    barR,
    keys: keyboard.keys,
    get resetRequested() {
      return keyboard.resetRequested;
    },
    set resetRequested(v) {
      keyboard.resetRequested = v;
    },
    setBar,
    clearKeys: keyboard.clearKeys,
  };
}
