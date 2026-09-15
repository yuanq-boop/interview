import * as THREE from 'three';
import type { Player } from '../entities/Player';

const TRAUMA_DECAY = 1.5;
const MAX_SHAKE_OFFSET = 0.13;
const MAX_SHAKE_ROLL = 0.035;

/** Deterministic value noise so shake replays identically under a fixed seed. */
function pseudoNoise(t: number, seed: number): number {
  const x = Math.sin(t * 12.9898 + seed * 78.233) * 43758.5453;
  return (x - Math.floor(x)) * 2 - 1;
}

export interface CameraFrameInput {
  readonly ads: boolean;
  readonly adsFraction: number;
  readonly sprinting: boolean;
  readonly moveFraction: number;
  readonly landed: boolean;
  readonly baseFov: number;
  readonly adsFov: number;
}

export class CameraRig {
  private trauma = 0;
  private time = 0;
  private landDip = 0;
  private damageRoll = 0;
  private fovPunch = 0;
  private currentFov: number;

  constructor(private readonly camera: THREE.PerspectiveCamera) {
    this.currentFov = camera.fov;
  }

  addTrauma(amount: number): void {
    this.trauma = Math.min(1, this.trauma + amount);
  }

  punchFov(degrees: number): void {
    this.fovPunch = Math.min(9, this.fovPunch + degrees);
  }

  /** Roll the view away from the hit so damage has a direction on screen. */
  addDamageRoll(amount: number): void {
    this.damageRoll = THREE.MathUtils.clamp(this.damageRoll + amount, -0.09, 0.09);
  }

  reset(): void {
    this.trauma = 0;
    this.landDip = 0;
    this.damageRoll = 0;
    this.fovPunch = 0;
  }

  update(delta: number, player: Player, input: CameraFrameInput): void {
    this.time += delta;
    this.trauma = Math.max(0, this.trauma - TRAUMA_DECAY * delta);
    this.damageRoll = THREE.MathUtils.damp(this.damageRoll, 0, 5, delta);
    this.fovPunch = THREE.MathUtils.damp(this.fovPunch, 0, 5.5, delta);

    if (input.landed) this.landDip = 0.09;
    this.landDip = THREE.MathUtils.damp(this.landDip, 0, 9, delta);

    // Bob is driven by distance travelled, so it stops the moment the player does.
    const bobStrength = input.moveFraction * (1 - input.adsFraction * 0.75) * (player.isGrounded ? 1 : 0);
    const bobY = Math.abs(Math.sin(player.bobPhase * 2.1)) * 0.038 * bobStrength;
    const bobX = Math.sin(player.bobPhase * 1.05) * 0.03 * bobStrength;
    const bobRoll = Math.sin(player.bobPhase * 1.05) * 0.011 * bobStrength;

    const eye = player.eyePosition;
    this.camera.position.set(
      eye.x + bobX * Math.cos(player.yaw),
      eye.y + bobY - this.landDip,
      eye.z - bobX * Math.sin(player.yaw),
    );
    this.camera.rotation.set(player.pitch, player.yaw, bobRoll + this.damageRoll, 'YXZ');

    if (this.trauma > 0) {
      const shake = this.trauma * this.trauma;
      const frequency = this.time * 34;
      this.camera.position.x += MAX_SHAKE_OFFSET * shake * pseudoNoise(frequency, 1);
      this.camera.position.y += MAX_SHAKE_OFFSET * shake * pseudoNoise(frequency, 2);
      this.camera.rotation.z += MAX_SHAKE_ROLL * shake * pseudoNoise(frequency, 3);
    }

    const targetFov =
      THREE.MathUtils.lerp(input.baseFov, input.adsFov, input.adsFraction) +
      (input.sprinting ? 4.5 : 0) +
      this.fovPunch;
    this.currentFov = THREE.MathUtils.damp(this.currentFov, targetFov, 12, delta);
    if (Math.abs(this.camera.fov - this.currentFov) > 0.01) {
      this.camera.fov = this.currentFov;
      this.camera.updateProjectionMatrix();
    }
  }

  /** Snaps the view to the player without any smoothing, for spawns and resets. */
  snapTo(player: Player, fov: number): void {
    this.reset();
    const eye = player.eyePosition;
    this.camera.position.copy(eye);
    this.camera.rotation.set(player.pitch, player.yaw, 0, 'YXZ');
    this.currentFov = fov;
    this.camera.fov = fov;
    this.camera.updateProjectionMatrix();
  }
}
