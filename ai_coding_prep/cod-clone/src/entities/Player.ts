import * as THREE from 'three';
import type { InputController } from '../core/InputController';
import type { CollisionSystem } from '../systems/CollisionSystem';

export const PLAYER_RADIUS = 0.36;
export const PLAYER_HEIGHT_STAND = 1.82;
export const PLAYER_HEIGHT_CROUCH = 1.22;
const EYE_OFFSET = 0.14;

const MAX_HEALTH = 100;
const REGEN_DELAY = 4;
const REGEN_RATE = 26;

const SPEED_WALK = 5.3;
const SPEED_SPRINT = 8.5;
const SPEED_CROUCH = 2.7;
const SPEED_ADS = 3.3;
const ACCEL_GROUND = 58;
const ACCEL_AIR = 9;
const GRAVITY = 23;
const JUMP_VELOCITY = 7.1;
const PITCH_LIMIT = Math.PI / 2 - 0.02;

export interface PlayerFrameState {
  readonly ads: boolean;
  readonly sprinting: boolean;
  readonly moveSpeedFraction: number;
  readonly landed: boolean;
  readonly stepped: boolean;
}

export class Player {
  readonly position = new THREE.Vector3(0, 0, 22);
  readonly velocity = new THREE.Vector3();
  yaw = Math.PI;
  pitch = 0;
  health = MAX_HEALTH;
  alive = true;
  /** Seconds of accumulated travel, used to phase the head bob. */
  bobPhase = 0;

  private crouchAmount = 0;
  private grounded = true;
  private sinceDamage = REGEN_DELAY;
  private stepAccumulator = 0;
  private readonly moveInput = new THREE.Vector2();
  private readonly lookInput = new THREE.Vector2();
  private readonly desiredVelocity = new THREE.Vector3();
  private readonly motion = new THREE.Vector3();

  get height(): number {
    return THREE.MathUtils.lerp(PLAYER_HEIGHT_STAND, PLAYER_HEIGHT_CROUCH, this.crouchAmount);
  }

  get eyeHeight(): number {
    return this.height - EYE_OFFSET;
  }

  get eyePosition(): THREE.Vector3 {
    return new THREE.Vector3(this.position.x, this.position.y + this.eyeHeight, this.position.z);
  }

  get isCrouched(): boolean {
    return this.crouchAmount > 0.5;
  }

  get healthFraction(): number {
    return Math.max(0, this.health) / MAX_HEALTH;
  }

  get isGrounded(): boolean {
    return this.grounded;
  }

  reset(spawn: THREE.Vector3, yaw: number): void {
    this.position.copy(spawn);
    this.velocity.set(0, 0, 0);
    this.yaw = yaw;
    this.pitch = 0;
    this.health = MAX_HEALTH;
    this.alive = true;
    this.crouchAmount = 0;
    this.grounded = true;
    this.sinceDamage = REGEN_DELAY;
    this.bobPhase = 0;
    this.stepAccumulator = 0;
  }

  applyLook(input: InputController, sensitivity: number): void {
    if (!this.alive) return;
    input.consumeLook(this.lookInput);
    this.yaw -= this.lookInput.x * sensitivity;
    this.pitch -= this.lookInput.y * sensitivity;
    this.pitch = THREE.MathUtils.clamp(this.pitch, -PITCH_LIMIT, PITCH_LIMIT);
    this.yaw = ((this.yaw + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI;
  }

  /** Adds camera kick from weapon recoil. Pitch rises, yaw drifts. */
  applyRecoil(pitch: number, yaw: number): void {
    this.pitch = THREE.MathUtils.clamp(this.pitch + pitch, -PITCH_LIMIT, PITCH_LIMIT);
    this.yaw += yaw;
  }

  forwardVector(target: THREE.Vector3): THREE.Vector3 {
    return target.set(
      -Math.sin(this.yaw) * Math.cos(this.pitch),
      Math.sin(this.pitch),
      -Math.cos(this.yaw) * Math.cos(this.pitch),
    );
  }

  update(
    delta: number,
    input: InputController,
    collision: CollisionSystem,
    canSprint: boolean,
    adsRequested: boolean,
  ): PlayerFrameState {
    if (!this.alive) {
      return { ads: false, sprinting: false, moveSpeedFraction: 0, landed: false, stepped: false };
    }

    input.readMovement(this.moveInput);
    const wantsCrouch = input.isHeld('crouch');
    const moving = this.moveInput.lengthSq() > 0.02;
    const wantsSprint =
      canSprint && input.isHeld('sprint') && moving && this.moveInput.y > 0.3 && !wantsCrouch;
    const ads = adsRequested && !wantsSprint;

    this.updateCrouch(delta, wantsCrouch, collision);

    const targetSpeed = wantsSprint
      ? SPEED_SPRINT
      : this.crouchAmount > 0.5
        ? SPEED_CROUCH
        : ads
          ? SPEED_ADS
          : SPEED_WALK;

    const sin = Math.sin(this.yaw);
    const cos = Math.cos(this.yaw);
    // Forward is -Z at yaw 0; strafe is its right-hand perpendicular.
    this.desiredVelocity.set(
      (-sin * this.moveInput.y + cos * this.moveInput.x) * targetSpeed,
      0,
      (-cos * this.moveInput.y - sin * this.moveInput.x) * targetSpeed,
    );

    const accel = (this.grounded ? ACCEL_GROUND : ACCEL_AIR) * delta;
    this.velocity.x = THREE.MathUtils.damp(this.velocity.x, this.desiredVelocity.x, accel, delta);
    this.velocity.z = THREE.MathUtils.damp(this.velocity.z, this.desiredVelocity.z, accel, delta);

    if (this.grounded && input.wasPressed('jump') && this.crouchAmount < 0.5) {
      this.velocity.y = JUMP_VELOCITY;
      this.grounded = false;
    }
    this.velocity.y -= GRAVITY * delta;

    const wasAirborne = !this.grounded;
    this.motion.copy(this.velocity).multiplyScalar(delta);
    this.grounded = collision.moveCharacter(
      this.position,
      this.motion,
      PLAYER_RADIUS,
      this.height,
      this.velocity,
    );

    const horizontalSpeed = Math.hypot(this.velocity.x, this.velocity.z);
    const moveSpeedFraction = Math.min(1, horizontalSpeed / SPEED_SPRINT);
    let stepped = false;
    if (this.grounded) {
      this.bobPhase += horizontalSpeed * delta;
      this.stepAccumulator += horizontalSpeed * delta;
      const stride = wantsSprint ? 2.1 : 1.65;
      if (this.stepAccumulator >= stride) {
        this.stepAccumulator -= stride;
        stepped = true;
      }
    }

    if (this.sinceDamage < REGEN_DELAY) {
      this.sinceDamage += delta;
    } else if (this.health < MAX_HEALTH) {
      this.health = Math.min(MAX_HEALTH, this.health + REGEN_RATE * delta);
    }

    return {
      ads,
      sprinting: wantsSprint,
      moveSpeedFraction,
      landed: wasAirborne && this.grounded,
      stepped,
    };
  }

  takeDamage(amount: number): boolean {
    if (!this.alive) return false;
    this.health -= amount;
    this.sinceDamage = 0;
    if (this.health <= 0) {
      this.health = 0;
      this.alive = false;
      return true;
    }
    return false;
  }

  private updateCrouch(delta: number, wantsCrouch: boolean, collision: CollisionSystem): void {
    const target = wantsCrouch ? 1 : 0;
    if (target === this.crouchAmount) return;
    const next = THREE.MathUtils.damp(this.crouchAmount, target, 14, delta);
    if (next < this.crouchAmount) {
      // Standing up must not push the head through a ceiling.
      const standingHeight = THREE.MathUtils.lerp(
        PLAYER_HEIGHT_STAND,
        PLAYER_HEIGHT_CROUCH,
        next,
      );
      if (!collision.isClear(this.position, PLAYER_RADIUS, standingHeight)) return;
    }
    this.crouchAmount = Math.abs(next - target) < 0.002 ? target : next;
  }
}
