/**
 * Keyboard input and differential-drive command from key state.
 */
import { SPEED } from "./constants.js";

/** Keys that drive the vacuum (captured for game, preventDefault). */
export const VACUUM_KEY_CODES = new Set([
  "KeyW",
  "KeyA",
  "KeyS",
  "KeyD",
  "ArrowUp",
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
]);

const UI_CONTROL_SELECTOR = "input,button,select,textarea,label,a[href],[contenteditable='true']";
const VACUUM_CAPTURE_KEY_CODES = new Set([...VACUUM_KEY_CODES, "KeyR", "Escape"]);

const MOVE_KEYS = {
  forward: ["KeyW", "ArrowUp"],
  back: ["KeyS", "ArrowDown"],
  left: ["KeyA", "ArrowLeft"],
  right: ["KeyD", "ArrowRight"],
};

function anyDown(keys, codes) {
  return codes.some((c) => keys[c]);
}

/** Wheel speeds (left / right) from keyboard; A/D are mapped to screen-left/screen-right steering. */
export function vacuumWheelSpeedsFromKeys(keys) {
  let vL = 0;
  let vR = 0;
  if (anyDown(keys, MOVE_KEYS.forward)) {
    vL += SPEED;
    vR += SPEED;
  }
  if (anyDown(keys, MOVE_KEYS.back)) {
    vL -= SPEED * 0.6;
    vR -= SPEED * 0.6;
  }
  if (anyDown(keys, MOVE_KEYS.left)) {
    vL += SPEED * 0.5;
    vR -= SPEED * 0.5;
  }
  if (anyDown(keys, MOVE_KEYS.right)) {
    vL -= SPEED * 0.5;
    vR += SPEED * 0.5;
  }
  return { vL, vR };
}

/**
 * Registers keydown/keyup/blur listeners. Returns mutable key state and restart flag.
 * @param {Window} targetWindow
 * @param {HTMLElement | null} focusTarget
 * @param {HTMLElement | null} focusScope
 */
export function createKeyboardControls(targetWindow = window, focusTarget = null, focusScope = focusTarget) {
  const keys = {};
  let resetRequested = false;
  let controlsActive = false;
  let notebookKeyboardPreviouslyEnabled = null;

  if (focusTarget) {
    focusTarget.tabIndex = 0;
    focusTarget.setAttribute("aria-label", "VacMan control surface");
    focusTarget.style.outline = "none";
  }

  function clearKeys() {
    Object.keys(keys).forEach((k) => {
      keys[k] = false;
    });
  }

  function isUiControl(target) {
    return target instanceof Element && Boolean(target.closest(UI_CONTROL_SELECTOR));
  }

  function notebookKeyboardManager() {
    return targetWindow.Jupyter?.notebook?.keyboard_manager ?? targetWindow.IPython?.notebook?.keyboard_manager ?? null;
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
    focusTarget.focus({ preventScroll: true });
    disableNotebookKeyboard();
  }

  function releaseControls() {
    controlsActive = false;
    clearKeys();
    restoreNotebookKeyboard();
  }

  function onPointerDown(e) {
    if (!focusTarget || !focusScope) return;
    if (!focusScope.contains(e.target) || isUiControl(e.target)) {
      releaseControls();
      return;
    }
    activateControls();
  }

  function hasKeyboardFocus() {
    return !focusTarget || controlsActive;
  }

  function claimKeyboardEvent(e) {
    if (!VACUUM_CAPTURE_KEY_CODES.has(e.code)) return false;
    if (!hasKeyboardFocus()) return false;
    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation?.();
    return true;
  }

  function onKeyDown(e) {
    if (!claimKeyboardEvent(e)) return;
    if (e.code === "Escape") {
      releaseControls();
      focusTarget?.blur();
      return;
    }
    if (VACUUM_KEY_CODES.has(e.code)) {
      keys[e.code] = true;
    }
    if (e.code === "KeyR") {
      resetRequested = true;
    }
  }

  function onKeyUp(e) {
    if (!claimKeyboardEvent(e)) return;
    if (VACUUM_KEY_CODES.has(e.code)) {
      keys[e.code] = false;
    }
  }

  targetWindow.document.addEventListener("pointerdown", onPointerDown, true);
  focusTarget?.addEventListener("blur", releaseControls);
  targetWindow.addEventListener("keydown", onKeyDown, { capture: true });
  targetWindow.addEventListener("keyup", onKeyUp, { capture: true });
  targetWindow.addEventListener("blur", releaseControls);

  function dispose() {
    targetWindow.document.removeEventListener("pointerdown", onPointerDown, true);
    focusTarget?.removeEventListener("blur", releaseControls);
    targetWindow.removeEventListener("keydown", onKeyDown, { capture: true });
    targetWindow.removeEventListener("keyup", onKeyUp, { capture: true });
    targetWindow.removeEventListener("blur", releaseControls);
  }

  return {
    keys,
    get resetRequested() {
      return resetRequested;
    },
    set resetRequested(v) {
      resetRequested = v;
    },
    clearKeys,
    dispose,
  };
}
