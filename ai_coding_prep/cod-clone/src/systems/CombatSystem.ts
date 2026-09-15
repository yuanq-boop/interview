import * as THREE from 'three';
import type { Enemy, HitZone } from '../entities/Enemy';

export interface ShotHit {
  readonly point: THREE.Vector3;
  readonly normal: THREE.Vector3;
  readonly distance: number;
  readonly enemy: Enemy | null;
  readonly zone: HitZone | null;
}

/**
 * Hitscan resolution. World geometry and enemy hitboxes are raycast together so
 * a shot is correctly blocked by cover the enemy is standing behind.
 */
export class CombatSystem {
  private readonly raycaster = new THREE.Raycaster();
  private readonly losRaycaster = new THREE.Raycaster();
  private readonly losDirection = new THREE.Vector3();
  private readonly hitPoint = new THREE.Vector3();
  private readonly hitNormal = new THREE.Vector3();
  private readonly normalMatrix = new THREE.Matrix3();

  resolveShot(
    origin: THREE.Vector3,
    direction: THREE.Vector3,
    range: number,
    worldTargets: readonly THREE.Object3D[],
    enemyTargets: readonly THREE.Object3D[],
  ): ShotHit | null {
    this.raycaster.set(origin, direction);
    this.raycaster.near = 0;
    this.raycaster.far = range;

    let best: THREE.Intersection | null = null;
    for (const list of [worldTargets, enemyTargets]) {
      const intersections = this.raycaster.intersectObjects(list as THREE.Object3D[], false);
      for (const intersection of intersections) {
        if (intersection.distance < 0.01) continue;
        const enemy = intersection.object.userData.enemy as Enemy | undefined;
        // Corpses keep their meshes for the death animation but stop taking hits.
        if (enemy && intersection.object.userData.enemyHittable !== true) continue;
        if (!best || intersection.distance < best.distance) best = intersection;
        break;
      }
    }

    if (!best) return null;

    this.hitPoint.copy(best.point);
    if (best.face) {
      this.normalMatrix.getNormalMatrix(best.object.matrixWorld);
      this.hitNormal.copy(best.face.normal).applyMatrix3(this.normalMatrix).normalize();
    } else {
      this.hitNormal.copy(direction).negate();
    }

    const enemy = (best.object.userData.enemy as Enemy | undefined) ?? null;
    return {
      point: this.hitPoint.clone(),
      normal: this.hitNormal.clone(),
      distance: best.distance,
      enemy,
      zone: enemy ? ((best.object.userData.hitZone as HitZone) ?? 'body') : null,
    };
  }

  hasLineOfSight(
    from: THREE.Vector3,
    to: THREE.Vector3,
    worldTargets: readonly THREE.Object3D[],
  ): boolean {
    this.losDirection.copy(to).sub(from);
    const distance = this.losDirection.length();
    if (distance < 0.001) return true;
    this.losDirection.divideScalar(distance);

    this.losRaycaster.set(from, this.losDirection);
    this.losRaycaster.near = 0;
    // Stop just short of the target so the target's own surface never blocks.
    this.losRaycaster.far = distance - 0.15;
    if (this.losRaycaster.far <= 0) return true;

    return this.losRaycaster.intersectObjects(worldTargets as THREE.Object3D[], false).length === 0;
  }
}

export function zoneMultiplier(zone: HitZone, headshotMultiplier: number): number {
  if (zone === 'head') return headshotMultiplier;
  if (zone === 'limb') return 0.75;
  return 1;
}
