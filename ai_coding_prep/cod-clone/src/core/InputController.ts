import * as THREE from 'three';

export type ActionName =
  | 'fire'
  | 'ads'
  | 'reload'
  | 'sprint'
  | 'crouch'
  | 'jump'
  | 'grenade'
  | 'melee'
  | 'swap'
  | 'pause';

const KEY_ACTIONS: Record<string, ActionName> = {
  KeyR: 'reload',
  ShiftLeft: 'sprint',
  ShiftRight: 'sprint',
  ControlLeft: 'crouch',
  KeyC: 'crouch',
  Space: 'jump',
  KeyG: 'grenade',
  KeyV: 'melee',
  KeyQ: 'swap',
  Tab: 'swap',
  KeyP: 'pause',
  Escape: 'pause',
};

const SLOT_KEYS: Record<string, number> = { Digit1: 0, Digit2: 1, Digit3: 2 };

/**
 * Pointer-lock FPS input. Look deltas accumulate between reads so a slow frame
 * still receives every mouse movement instead of only the most recent event.
 */
export class InputController {
  private readonly held = new Set<ActionName>();
  private readonly pressed = new Set<ActionName>();
  private readonly keys = new Set<string>();
  private readonly look = new THREE.Vector2();
  private readonly moveVector = new THREE.Vector2();
  private readonly touchMove = new THREE.Vector2();
  private readonly stickState = { id: -1, centerX: 0, centerY: 0, radius: 1 };
  private readonly lookTouch = { id: -1, x: 0, y: 0 };
  private scriptedMove: THREE.Vector2 | null = null;
  private requestedSlot: number | null = null;
  private locked = false;
  private touchActive = false;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly elements: {
      stick: HTMLElement;
      knob: HTMLElement;
      lookPad: HTMLElement;
      fire: HTMLElement;
      ads: HTMLElement;
      reload: HTMLElement;
      jump: HTMLElement;
      crouch: HTMLElement;
      swap: HTMLElement;
      grenade: HTMLElement;
    },
  ) {
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
    window.addEventListener('blur', this.onBlur);
    document.addEventListener('pointerlockchange', this.onPointerLockChange);
    this.canvas.addEventListener('mousedown', this.onMouseDown);
    window.addEventListener('mouseup', this.onMouseUp);
    window.addEventListener('mousemove', this.onMouseMove);
    this.canvas.addEventListener('contextmenu', this.onContextMenu);

    this.elements.stick.addEventListener('pointerdown', this.onStickDown);
    this.elements.stick.addEventListener('pointermove', this.onStickMove);
    this.elements.stick.addEventListener('pointerup', this.onStickUp);
    this.elements.stick.addEventListener('pointercancel', this.onStickUp);
    this.elements.lookPad.addEventListener('pointerdown', this.onLookDown);
    this.elements.lookPad.addEventListener('pointermove', this.onLookMove);
    this.elements.lookPad.addEventListener('pointerup', this.onLookUp);
    this.elements.lookPad.addEventListener('pointercancel', this.onLookUp);

    // Show touch controls up front on a touch device. Waiting for the first
    // touch would hide the only controls the player has from the player who
    // needs them, so detect the device instead of the gesture.
    const coarsePointer =
      window.matchMedia?.('(pointer: coarse)').matches ?? false;
    if (coarsePointer || navigator.maxTouchPoints > 0) this.markTouchActive();

    this.bindTouchButton(this.elements.fire, 'fire');
    this.bindTouchButton(this.elements.ads, 'ads');
    this.bindTouchButton(this.elements.reload, 'reload');
    this.bindTouchButton(this.elements.jump, 'jump');
    this.bindTouchButton(this.elements.crouch, 'crouch');
    this.bindTouchButton(this.elements.swap, 'swap');
    this.bindTouchButton(this.elements.grenade, 'grenade');
  }

  get isPointerLocked(): boolean {
    return this.locked;
  }

  get isTouchActive(): boolean {
    return this.touchActive;
  }

  requestPointerLock(): void {
    if (this.locked) return;
    void this.canvas.requestPointerLock?.();
  }

  releasePointerLock(): void {
    if (this.locked) document.exitPointerLock?.();
  }

  /** Mouse/touch look accumulated since the previous call, in raw device units. */
  consumeLook(target: THREE.Vector2): THREE.Vector2 {
    target.copy(this.look);
    this.look.set(0, 0);
    return target;
  }

  /** Overrides stick/key movement for scripted bot playtests. */
  setScriptedMovement(move: THREE.Vector2 | null): void {
    this.scriptedMove = move;
  }

  readMovement(target: THREE.Vector2): THREE.Vector2 {
    if (this.scriptedMove) {
      target.copy(this.scriptedMove);
      if (target.lengthSq() > 1) target.normalize();
      return target;
    }

    this.moveVector.set(0, 0);
    if (this.keys.has('KeyA') || this.keys.has('ArrowLeft')) this.moveVector.x -= 1;
    if (this.keys.has('KeyD') || this.keys.has('ArrowRight')) this.moveVector.x += 1;
    if (this.keys.has('KeyW') || this.keys.has('ArrowUp')) this.moveVector.y += 1;
    if (this.keys.has('KeyS') || this.keys.has('ArrowDown')) this.moveVector.y -= 1;

    target.copy(this.moveVector).add(this.touchMove);
    if (target.lengthSq() > 1) target.normalize();
    return target;
  }

  isHeld(action: ActionName): boolean {
    return this.held.has(action);
  }

  wasPressed(action: ActionName): boolean {
    return this.pressed.has(action);
  }

  consumeSlotRequest(): number | null {
    const slot = this.requestedSlot;
    this.requestedSlot = null;
    return slot;
  }

  /** Call once at the end of every frame so edge triggers fire exactly once. */
  endFrame(): void {
    this.pressed.clear();
  }

  dispose(): void {
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
    window.removeEventListener('blur', this.onBlur);
    document.removeEventListener('pointerlockchange', this.onPointerLockChange);
    this.canvas.removeEventListener('mousedown', this.onMouseDown);
    window.removeEventListener('mouseup', this.onMouseUp);
    window.removeEventListener('mousemove', this.onMouseMove);
    this.canvas.removeEventListener('contextmenu', this.onContextMenu);

    this.elements.stick.removeEventListener('pointerdown', this.onStickDown);
    this.elements.stick.removeEventListener('pointermove', this.onStickMove);
    this.elements.stick.removeEventListener('pointerup', this.onStickUp);
    this.elements.stick.removeEventListener('pointercancel', this.onStickUp);
    this.elements.lookPad.removeEventListener('pointerdown', this.onLookDown);
    this.elements.lookPad.removeEventListener('pointermove', this.onLookMove);
    this.elements.lookPad.removeEventListener('pointerup', this.onLookUp);
    this.elements.lookPad.removeEventListener('pointercancel', this.onLookUp);

    for (const [element, handlers] of this.touchButtonHandlers) {
      element.removeEventListener('pointerdown', handlers.down);
      element.removeEventListener('pointerup', handlers.up);
      element.removeEventListener('pointercancel', handlers.up);
      element.removeEventListener('pointerleave', handlers.up);
    }
    this.touchButtonHandlers.clear();
  }

  private readonly touchButtonHandlers = new Map<
    HTMLElement,
    { down: (event: PointerEvent) => void; up: (event: PointerEvent) => void }
  >();

  private bindTouchButton(element: HTMLElement, action: ActionName): void {
    const down = (event: PointerEvent) => {
      event.preventDefault();
      this.markTouchActive();
      this.press(action);
      this.held.add(action);
    };
    const up = (event: PointerEvent) => {
      event.preventDefault();
      this.held.delete(action);
    };
    element.addEventListener('pointerdown', down);
    element.addEventListener('pointerup', up);
    element.addEventListener('pointercancel', up);
    element.addEventListener('pointerleave', up);
    this.touchButtonHandlers.set(element, { down, up });
  }

  private press(action: ActionName): void {
    this.pressed.add(action);
  }

  private markTouchActive(): void {
    this.touchActive = true;
    document.body.dataset.touch = 'true';
  }

  private readonly onKeyDown = (event: KeyboardEvent) => {
    if (event.code === 'Tab') event.preventDefault();
    if (event.repeat) return;
    this.keys.add(event.code);

    const slot = SLOT_KEYS[event.code];
    if (slot !== undefined) this.requestedSlot = slot;

    const action = KEY_ACTIONS[event.code];
    if (!action) return;
    this.press(action);
    this.held.add(action);
  };

  private readonly onKeyUp = (event: KeyboardEvent) => {
    this.keys.delete(event.code);
    const action = KEY_ACTIONS[event.code];
    if (action) this.held.delete(action);
  };

  private readonly onBlur = () => {
    this.keys.clear();
    this.held.clear();
    this.touchMove.set(0, 0);
    this.look.set(0, 0);
  };

  private readonly onPointerLockChange = () => {
    this.locked = document.pointerLockElement === this.canvas;
    if (!this.locked) {
      this.held.delete('fire');
      this.held.delete('ads');
    }
  };

  private readonly onMouseDown = (event: MouseEvent) => {
    if (!this.locked) return;
    event.preventDefault();
    if (event.button === 0) {
      this.press('fire');
      this.held.add('fire');
    } else if (event.button === 2) {
      this.press('ads');
      this.held.add('ads');
    }
  };

  private readonly onMouseUp = (event: MouseEvent) => {
    if (event.button === 0) this.held.delete('fire');
    if (event.button === 2) this.held.delete('ads');
  };

  private readonly onMouseMove = (event: MouseEvent) => {
    if (!this.locked) return;
    this.look.x += event.movementX;
    this.look.y += event.movementY;
  };

  private readonly onContextMenu = (event: Event) => {
    event.preventDefault();
  };

  private readonly onStickDown = (event: PointerEvent) => {
    event.preventDefault();
    this.markTouchActive();
    const rect = this.elements.stick.getBoundingClientRect();
    this.stickState.id = event.pointerId;
    this.stickState.centerX = rect.left + rect.width / 2;
    this.stickState.centerY = rect.top + rect.height / 2;
    this.stickState.radius = Math.max(1, rect.width * 0.42);
    try {
      this.elements.stick.setPointerCapture(event.pointerId);
    } catch {
      // Synthetic test events do not always carry a capturable pointer id.
    }
    this.updateStick(event.clientX, event.clientY);
  };

  private readonly onStickMove = (event: PointerEvent) => {
    if (event.pointerId !== this.stickState.id) return;
    event.preventDefault();
    this.updateStick(event.clientX, event.clientY);
  };

  private readonly onStickUp = (event: PointerEvent) => {
    if (event.pointerId !== this.stickState.id) return;
    event.preventDefault();
    this.stickState.id = -1;
    this.touchMove.set(0, 0);
    this.elements.knob.style.transform = 'translate(-50%, -50%)';
  };

  private updateStick(clientX: number, clientY: number): void {
    const dx = (clientX - this.stickState.centerX) / this.stickState.radius;
    const dy = (clientY - this.stickState.centerY) / this.stickState.radius;
    this.touchMove.set(dx, -dy);
    if (this.touchMove.lengthSq() > 1) this.touchMove.normalize();
    const offset = 34;
    this.elements.knob.style.transform = `translate(calc(-50% + ${this.touchMove.x * offset}px), calc(-50% + ${-this.touchMove.y * offset}px))`;
  }

  private readonly onLookDown = (event: PointerEvent) => {
    event.preventDefault();
    this.markTouchActive();
    this.lookTouch.id = event.pointerId;
    this.lookTouch.x = event.clientX;
    this.lookTouch.y = event.clientY;
    try {
      this.elements.lookPad.setPointerCapture(event.pointerId);
    } catch {
      // Synthetic test events do not always carry a capturable pointer id.
    }
  };

  private readonly onLookMove = (event: PointerEvent) => {
    if (event.pointerId !== this.lookTouch.id) return;
    event.preventDefault();
    this.look.x += event.clientX - this.lookTouch.x;
    this.look.y += event.clientY - this.lookTouch.y;
    this.lookTouch.x = event.clientX;
    this.lookTouch.y = event.clientY;
  };

  private readonly onLookUp = (event: PointerEvent) => {
    if (event.pointerId !== this.lookTouch.id) return;
    event.preventDefault();
    this.lookTouch.id = -1;
  };
}
