import * as THREE from 'three';

const STEP_HEIGHT = 0.62;

/**
 * Axis-aligned box world. Characters are treated as an AABB of `radius` and
 * `height` and resolved one axis at a time, which keeps corners from tunnelling
 * and lets stairs be authored as ordinary boxes via the step-up allowance.
 */
export class CollisionSystem {
  private readonly boxes: THREE.Box3[] = [];
  private readonly probe = new THREE.Box3();

  add(box: THREE.Box3): void {
    this.boxes.push(box);
  }

  addFromCenterSize(center: THREE.Vector3, size: THREE.Vector3): THREE.Box3 {
    const half = size.clone().multiplyScalar(0.5);
    const box = new THREE.Box3(center.clone().sub(half), center.clone().add(half));
    this.boxes.push(box);
    return box;
  }

  clear(): void {
    this.boxes.length = 0;
  }

  /**
   * Applies `motion` to `position` (the character's feet) with per-axis
   * resolution. Returns whether the character ended the move on the ground.
   */
  moveCharacter(
    position: THREE.Vector3,
    motion: THREE.Vector3,
    radius: number,
    height: number,
    velocity: THREE.Vector3,
  ): boolean {
    position.x += motion.x;
    this.resolveAxis(position, radius, height, 'x', motion.x);
    position.z += motion.z;
    this.resolveAxis(position, radius, height, 'z', motion.z);

    position.y += motion.y;
    let grounded = false;
    if (position.y <= 0) {
      position.y = 0;
      grounded = true;
      if (velocity.y < 0) velocity.y = 0;
    }

    this.setProbe(position, radius, height);
    for (const box of this.boxes) {
      if (!box.intersectsBox(this.probe)) continue;
      if (motion.y <= 0) {
        // Landing: only snap up onto a surface the feet actually crossed.
        position.y = box.max.y;
        grounded = true;
        if (velocity.y < 0) velocity.y = 0;
      } else {
        position.y = box.min.y - height;
        if (velocity.y > 0) velocity.y = 0;
      }
      this.setProbe(position, radius, height);
    }

    return grounded;
  }

  /** True when the character box at this position overlaps nothing solid. */
  isClear(position: THREE.Vector3, radius: number, height: number): boolean {
    this.setProbe(position, radius, height);
    for (const box of this.boxes) {
      if (box.intersectsBox(this.probe)) return false;
    }
    return true;
  }

  /** Highest solid surface at or below `fromY` under this footprint. */
  groundHeightAt(x: number, z: number, fromY: number, radius: number): number {
    let best = 0;
    for (const box of this.boxes) {
      if (x + radius < box.min.x || x - radius > box.max.x) continue;
      if (z + radius < box.min.z || z - radius > box.max.z) continue;
      if (box.max.y > fromY + 0.01) continue;
      if (box.max.y > best) best = box.max.y;
    }
    return best;
  }

  private resolveAxis(
    position: THREE.Vector3,
    radius: number,
    height: number,
    axis: 'x' | 'z',
    delta: number,
  ): void {
    if (delta === 0) return;
    this.setProbe(position, radius, height);
    for (const box of this.boxes) {
      if (!box.intersectsBox(this.probe)) continue;

      const step = box.max.y - position.y;
      if (step > 0 && step <= STEP_HEIGHT) {
        const stepped = position.clone();
        stepped.y = box.max.y;
        if (this.isClear(stepped, radius, height)) {
          position.y = box.max.y;
          this.setProbe(position, radius, height);
          continue;
        }
      }

      position[axis] = delta > 0 ? box.min[axis] - radius : box.max[axis] + radius;
      this.setProbe(position, radius, height);
    }
  }

  private setProbe(position: THREE.Vector3, radius: number, height: number): void {
    this.probe.min.set(position.x - radius, position.y, position.z - radius);
    this.probe.max.set(position.x + radius, position.y + height, position.z + radius);
  }
}
