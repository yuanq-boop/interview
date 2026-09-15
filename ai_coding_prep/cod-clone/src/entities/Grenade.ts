import * as THREE from 'three';
import type { CollisionSystem } from '../systems/CollisionSystem';
import { disposeObject3D } from '../utils/dispose';

export const GRENADE_FUSE = 2.1;
export const GRENADE_RADIUS = 7.5;
export const GRENADE_DAMAGE = 190;
const GRENADE_BODY_RADIUS = 0.12;

export class Grenade {
  readonly group = new THREE.Group();
  readonly velocity = new THREE.Vector3();
  fuse = GRENADE_FUSE;
  exploded = false;

  private readonly motion = new THREE.Vector3();
  private readonly spin = new THREE.Vector3();

  constructor(origin: THREE.Vector3, direction: THREE.Vector3, speed: number, rng: () => number) {
    this.group.position.copy(origin);
    this.velocity.copy(direction).multiplyScalar(speed);
    this.velocity.y += 2.4;
    this.spin.set((rng() - 0.5) * 14, (rng() - 0.5) * 14, (rng() - 0.5) * 14);

    const body = new THREE.Mesh(
      new THREE.SphereGeometry(GRENADE_BODY_RADIUS, 12, 10),
      new THREE.MeshStandardMaterial({ color: '#59663c', roughness: 0.68, metalness: 0.32 }),
    );
    body.castShadow = true;
    this.group.add(body);

    const lever = new THREE.Mesh(
      new THREE.BoxGeometry(0.035, 0.11, 0.035),
      new THREE.MeshStandardMaterial({ color: '#8d9299', roughness: 0.4, metalness: 0.8 }),
    );
    lever.position.set(0.1, 0.02, 0);
    this.group.add(lever);

    const beacon = new THREE.PointLight('#ff6a2c', 2.2, 4, 2);
    this.group.add(beacon);
  }

  update(delta: number, collision: CollisionSystem): void {
    this.fuse -= delta;
    this.velocity.y -= 21 * delta;
    this.motion.copy(this.velocity).multiplyScalar(delta);

    const before = this.group.position.clone();
    collision.moveCharacter(
      this.group.position,
      this.motion,
      GRENADE_BODY_RADIUS,
      GRENADE_BODY_RADIUS * 2,
      this.velocity,
    );

    // A blocked axis means a surface was hit: bounce and bleed off energy.
    const moved = this.group.position.clone().sub(before);
    if (Math.abs(moved.x - this.motion.x) > 1e-4) this.velocity.x *= -0.42;
    if (Math.abs(moved.z - this.motion.z) > 1e-4) this.velocity.z *= -0.42;
    if (Math.abs(moved.y - this.motion.y) > 1e-4) {
      this.velocity.y *= -0.34;
      this.velocity.x *= 0.72;
      this.velocity.z *= 0.72;
      this.spin.multiplyScalar(0.6);
    }

    this.group.rotation.x += this.spin.x * delta;
    this.group.rotation.y += this.spin.y * delta;
    this.group.rotation.z += this.spin.z * delta;
  }

  dispose(): void {
    disposeObject3D(this.group);
  }
}

/** Falloff from full damage at the centre to zero at the blast edge. */
export function blastDamage(distance: number): number {
  if (distance >= GRENADE_RADIUS) return 0;
  const t = 1 - distance / GRENADE_RADIUS;
  return GRENADE_DAMAGE * t * t;
}
