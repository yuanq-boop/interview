import * as THREE from 'three';
import { disposeObject3D } from '../utils/dispose';

const RECHARGE_SECONDS = 16;
const PICKUP_RADIUS = 1.9;

/** A resupply point that recharges on a timer instead of vanishing. */
export class AmmoCrate {
  readonly group = new THREE.Group();

  private readonly lid: THREE.Mesh;
  private readonly glow: THREE.Mesh;
  private readonly light: THREE.PointLight;
  private cooldown = 0;
  private bobSeed: number;

  constructor(position: THREE.Vector3, seed: number) {
    this.group.position.copy(position);
    this.bobSeed = seed;

    const crate = new THREE.Mesh(
      new THREE.BoxGeometry(1.1, 0.62, 0.74),
      new THREE.MeshStandardMaterial({ color: '#4a5340', roughness: 0.82, metalness: 0.12 }),
    );
    crate.position.y = 0.31;
    crate.castShadow = true;
    crate.receiveShadow = true;
    this.group.add(crate);

    this.lid = new THREE.Mesh(
      new THREE.BoxGeometry(1.16, 0.12, 0.8),
      new THREE.MeshStandardMaterial({ color: '#39412f', roughness: 0.74, metalness: 0.2 }),
    );
    this.lid.position.y = 0.68;
    this.lid.castShadow = true;
    this.group.add(this.lid);

    const band = new THREE.Mesh(
      new THREE.BoxGeometry(1.14, 0.1, 0.1),
      new THREE.MeshStandardMaterial({ color: '#c2703a', roughness: 0.6, metalness: 0.3 }),
    );
    band.position.set(0, 0.36, 0.38);
    this.group.add(band);

    this.glow = new THREE.Mesh(
      new THREE.RingGeometry(0.85, 1.05, 28),
      new THREE.MeshBasicMaterial({
        color: '#d8a13a',
        transparent: true,
        opacity: 0.6,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    this.glow.rotation.x = -Math.PI / 2;
    this.glow.position.y = 0.03;
    this.group.add(this.glow);

    this.light = new THREE.PointLight('#e0a83f', 3.2, 6, 2);
    this.light.position.y = 1;
    this.group.add(this.light);
  }

  get isReady(): boolean {
    return this.cooldown <= 0;
  }

  reset(): void {
    this.cooldown = 0;
  }

  update(delta: number, elapsed: number): void {
    if (this.cooldown > 0) this.cooldown -= delta;

    const ready = this.isReady;
    const pulse = ready ? 0.45 + Math.sin(elapsed * 2.4 + this.bobSeed) * 0.22 : 0.08;
    (this.glow.material as THREE.MeshBasicMaterial).opacity = pulse;
    this.light.intensity = ready ? 2.4 + Math.sin(elapsed * 2.4 + this.bobSeed) * 1 : 0.3;
    this.lid.position.y = 0.68 + (ready ? Math.sin(elapsed * 1.6 + this.bobSeed) * 0.012 : 0);
  }

  /** Returns true when the player is close enough and the crate is charged. */
  tryConsume(playerPosition: THREE.Vector3): boolean {
    if (!this.isReady) return false;
    const dx = playerPosition.x - this.group.position.x;
    const dz = playerPosition.z - this.group.position.z;
    if (dx * dx + dz * dz > PICKUP_RADIUS * PICKUP_RADIUS) return false;
    this.cooldown = RECHARGE_SECONDS;
    return true;
  }

  dispose(): void {
    disposeObject3D(this.group);
  }
}
