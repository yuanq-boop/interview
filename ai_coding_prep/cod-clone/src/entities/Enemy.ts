import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import type { CollisionSystem } from '../systems/CollisionSystem';

/**
 * One shared material for every soldier on the field. Per-part colour is baked
 * into vertex colours instead, so a whole limb or torso cluster collapses into
 * a single draw call and archetypes still look different.
 */
const SOLDIER_MATERIAL = new THREE.MeshStandardMaterial({
  vertexColors: true,
  roughness: 0.78,
  metalness: 0.12,
});

interface Part {
  readonly geometry: THREE.BufferGeometry;
  readonly color: string;
  readonly position: readonly [number, number, number];
  readonly rotation?: readonly [number, number, number];
}

/** Bakes colour and transform into each part, then merges them into one mesh. */
function mergeParts(parts: readonly Part[]): THREE.BufferGeometry {
  const prepared = parts.map((part) => {
    const geometry = part.geometry;
    const color = new THREE.Color(part.color);
    const count = geometry.attributes.position.count;
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const matrix = new THREE.Matrix4();
    const rotation = part.rotation ?? [0, 0, 0];
    matrix.compose(
      new THREE.Vector3(part.position[0], part.position[1], part.position[2]),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(rotation[0], rotation[1], rotation[2])),
      new THREE.Vector3(1, 1, 1),
    );
    geometry.applyMatrix4(matrix);
    return geometry;
  });

  const merged = BufferGeometryUtils.mergeGeometries(prepared, false);
  for (const geometry of prepared) geometry.dispose();
  if (!merged) throw new Error('Failed to merge soldier geometry.');
  merged.computeBoundingSphere();
  return merged;
}

export type EnemyArchetypeId = 'rifleman' | 'rusher' | 'heavy';
export type HitZone = 'head' | 'body' | 'limb';

export interface EnemyArchetype {
  readonly id: EnemyArchetypeId;
  readonly name: string;
  readonly health: number;
  readonly speed: number;
  readonly damage: number;
  /** Seconds between shots inside a burst. */
  readonly shotInterval: number;
  readonly burstSize: number;
  readonly burstPause: number;
  /** Aim cone half-angle in radians; larger is less accurate. */
  readonly aimError: number;
  readonly preferredRange: number;
  readonly score: number;
  readonly uniform: string;
  readonly vest: string;
  readonly accent: string;
  readonly scale: number;
}

export const ENEMY_ARCHETYPES: Record<EnemyArchetypeId, EnemyArchetype> = {
  rifleman: {
    id: 'rifleman',
    name: 'Rifleman',
    health: 100,
    speed: 3.3,
    damage: 10,
    shotInterval: 0.16,
    burstSize: 3,
    burstPause: 1.35,
    aimError: 0.055,
    preferredRange: 17,
    score: 100,
    uniform: '#4b5138',
    vest: '#2c2f26',
    accent: '#8d3b2f',
    scale: 1,
  },
  rusher: {
    id: 'rusher',
    name: 'Rusher',
    health: 68,
    speed: 6.5,
    damage: 13,
    shotInterval: 0.1,
    burstSize: 5,
    burstPause: 1,
    aimError: 0.115,
    preferredRange: 4.5,
    score: 150,
    uniform: '#53392f',
    vest: '#312119',
    accent: '#d0762c',
    scale: 0.94,
  },
  heavy: {
    id: 'heavy',
    name: 'Heavy',
    health: 250,
    speed: 2.2,
    damage: 17,
    shotInterval: 0.2,
    burstSize: 4,
    burstPause: 1.7,
    aimError: 0.042,
    preferredRange: 13,
    score: 250,
    uniform: '#37414b',
    vest: '#1d232a',
    accent: '#3f7fa8',
    scale: 1.14,
  },
};

export const ENEMY_RADIUS = 0.42;
export const ENEMY_HEIGHT = 1.78;

type EnemyState = 'exit' | 'advance' | 'engage' | 'reposition' | 'dead';

export interface EnemyUpdateContext {
  readonly delta: number;
  readonly playerEye: THREE.Vector3;
  readonly playerPosition: THREE.Vector3;
  readonly playerAlive: boolean;
  readonly collision: CollisionSystem;
  readonly rng: () => number;
  readonly hasLineOfSight: (from: THREE.Vector3, to: THREE.Vector3) => boolean;
  readonly onFire: (enemy: Enemy, origin: THREE.Vector3, direction: THREE.Vector3) => void;
  readonly others: readonly Enemy[];
}

export class Enemy {
  readonly group = new THREE.Group();
  readonly hitMeshes: THREE.Mesh[] = [];
  readonly velocity = new THREE.Vector3();
  health: number;
  state: EnemyState = 'advance';
  alive = true;
  /** Last line-of-sight result, surfaced in diagnostics for AI debugging. */
  lastVisible = false;
  /** Set once the death animation has fully played out. */
  finished = false;

  private readonly archetypeRef: EnemyArchetype;
  private readonly legs: THREE.Group[] = [];
  private readonly arms: THREE.Group[] = [];
  private readonly muzzle = new THREE.Group();
  private readonly muzzleFlash: THREE.Mesh;
  private readonly torsoGroup = new THREE.Group();
  private readonly desired = new THREE.Vector3();
  private readonly toPlayer = new THREE.Vector3();
  private readonly motion = new THREE.Vector3();
  private readonly strafeDirection = new THREE.Vector3();
  private readonly eyeScratch = new THREE.Vector3();
  private readonly fireOrigin = new THREE.Vector3();
  private readonly fireDirection = new THREE.Vector3();
  private readonly repositionTarget = new THREE.Vector3();
  private readonly probeScratch = new THREE.Vector3();
  private readonly probeHeading = new THREE.Vector3();

  private walkPhase = 0;
  private burstRemaining = 0;
  private shotTimer = 0;
  private burstCooldown = 0;
  private strafeTimer = 0;
  private strafeSign = 1;
  private repositionTimer = 0;
  private deathTimer = 0;
  private flashTimer = 0;
  private stuckTimer = 0;
  private readonly lastPosition = new THREE.Vector3();
  private facingYaw = 0;
  private spawnFade = 0;

  /** Remaining exit waypoints; walked in order before free steering begins. */
  private readonly exitPath: THREE.Vector3[] = [];

  constructor(
    readonly archetype: EnemyArchetype,
    spawn: THREE.Vector3,
    rng: () => number,
    exitPath: readonly THREE.Vector3[] = [],
  ) {
    this.archetypeRef = archetype;
    this.health = archetype.health;
    this.group.position.copy(spawn);
    for (const point of exitPath) this.exitPath.push(point.clone());
    this.burstCooldown = 0.35 + rng() * 0.5;
    this.strafeSign = rng() > 0.5 ? 1 : -1;
    this.strafeTimer = 1 + rng() * 1.5;
    this.muzzleFlash = this.buildBody();
    this.group.scale.setScalar(0.001);
  }

  get eyePosition(): THREE.Vector3 {
    return this.eyeScratch.set(
      this.group.position.x,
      this.group.position.y + ENEMY_HEIGHT * this.archetypeRef.scale * 0.86,
      this.group.position.z,
    );
  }

  get healthFraction(): number {
    return Math.max(0, this.health) / this.archetypeRef.health;
  }

  /** Returns true when this hit killed the enemy. */
  applyDamage(amount: number): boolean {
    if (!this.alive) return false;
    this.health -= amount;
    if (this.health <= 0) {
      this.health = 0;
      this.kill();
      return true;
    }
    return false;
  }

  kill(): void {
    if (!this.alive) return;
    this.alive = false;
    this.state = 'dead';
    this.deathTimer = 0;
    for (const mesh of this.hitMeshes) mesh.userData.enemyHittable = false;
  }

  update(context: EnemyUpdateContext): void {
    const { delta } = context;
    this.spawnFade = Math.min(1, this.spawnFade + delta * 3.4);
    this.group.scale.setScalar(this.archetypeRef.scale * this.spawnFade);

    if (this.flashTimer > 0) {
      this.flashTimer -= delta;
      this.muzzleFlash.visible = this.flashTimer > 0;
    }

    if (!this.alive) {
      this.updateDeath(delta);
      return;
    }

    this.toPlayer.copy(context.playerPosition).sub(this.group.position);
    this.toPlayer.y = 0;
    const distance = this.toPlayer.length();
    if (distance > 0.001) this.toPlayer.divideScalar(distance);

    const visible =
      context.playerAlive && context.hasLineOfSight(this.eyePosition, context.playerEye);
    this.lastVisible = visible;

    if (this.advanceExitPath()) {
      this.state = 'exit';
      this.steerToExit(context, delta);
    } else {
      this.chooseState(context, distance, visible, delta);
      this.steer(context, distance, delta);
    }
    this.applyMotion(context, delta);
    this.detectStuck(delta);
    this.faceTarget(context, visible, delta);
    this.animateLocomotion(delta);

    if (visible && context.playerAlive) {
      this.updateFiring(context, distance, delta);
    } else {
      this.burstRemaining = 0;
    }
  }

  dispose(): void {
    // Geometry is per-enemy, but SOLDIER_MATERIAL is shared by every soldier on
    // the field — disposing it here would blank all the others.
    this.group.traverse((object) => {
      const mesh = object as THREE.Mesh;
      mesh.geometry?.dispose();
      const material = mesh.material;
      if (material && !Array.isArray(material) && material !== SOLDIER_MATERIAL) {
        material.dispose();
      }
    });
  }

  /** Drops waypoints as they are reached; true while a route remains. */
  private advanceExitPath(): boolean {
    while (this.exitPath.length > 0) {
      const target = this.exitPath[0];
      const dx = target.x - this.group.position.x;
      const dz = target.z - this.group.position.z;
      if (dx * dx + dz * dz > 2.6 * 2.6) return true;
      this.exitPath.shift();
    }
    return false;
  }

  private steerToExit(context: EnemyUpdateContext, delta: number): void {
    const target = this.exitPath[0];
    this.desired.set(target.x - this.group.position.x, 0, target.z - this.group.position.z);
    if (this.desired.lengthSq() > 0.001) {
      this.desired.normalize();
      // Interior cover sits across some exit lanes, so the route needs the same
      // avoidance as free steering or the enemy grinds into a sandbag forever.
      this.avoidObstacles(context, this.desired, this.archetypeRef.speed);
      this.desired.normalize().multiplyScalar(this.archetypeRef.speed);
    }
    const accel = 12 * delta;
    this.velocity.x = THREE.MathUtils.damp(this.velocity.x, this.desired.x, accel, delta);
    this.velocity.z = THREE.MathUtils.damp(this.velocity.z, this.desired.z, accel, delta);
  }

  private chooseState(
    context: EnemyUpdateContext,
    distance: number,
    visible: boolean,
    delta: number,
  ): void {
    this.repositionTimer -= delta;

    if (!visible) {
      if (this.state !== 'reposition' || this.repositionTimer <= 0) {
        this.state = 'reposition';
        this.repositionTimer = 1.4 + context.rng() * 1.2;
        // Aim for a point beside the player so the approach is not a straight line.
        const lateral = (context.rng() - 0.5) * 12;
        this.repositionTarget
          .copy(context.playerPosition)
          .addScaledVector(this.strafeDirection.set(-this.toPlayer.z, 0, this.toPlayer.x), lateral);
      }
      return;
    }

    this.state = distance > this.archetypeRef.preferredRange * 1.15 ? 'advance' : 'engage';
  }

  private steer(context: EnemyUpdateContext, distance: number, delta: number): void {
    const speed = this.archetypeRef.speed;
    this.desired.set(0, 0, 0);

    if (this.state === 'reposition') {
      const toTarget = this.repositionTarget.clone().sub(this.group.position);
      toTarget.y = 0;
      if (toTarget.lengthSq() > 0.25) {
        this.desired.copy(toTarget.normalize()).multiplyScalar(speed);
      }
    } else if (this.state === 'advance') {
      this.desired.copy(this.toPlayer).multiplyScalar(speed);
    } else {
      // Engaging: hold the preferred band and strafe across the player's view.
      this.strafeTimer -= delta;
      if (this.strafeTimer <= 0) {
        this.strafeTimer = 1.1 + context.rng() * 1.6;
        this.strafeSign *= -1;
      }
      const bandError = distance - this.archetypeRef.preferredRange;
      this.desired.copy(this.toPlayer).multiplyScalar(THREE.MathUtils.clamp(bandError, -1, 1) * speed * 0.6);
      this.strafeDirection.set(-this.toPlayer.z, 0, this.toPlayer.x).multiplyScalar(this.strafeSign);
      this.desired.addScaledVector(this.strafeDirection, speed * 0.75);
    }

    // Separation keeps a squad from stacking into one silhouette.
    for (const other of context.others) {
      if (other === this || !other.alive) continue;
      const dx = this.group.position.x - other.group.position.x;
      const dz = this.group.position.z - other.group.position.z;
      const distSq = dx * dx + dz * dz;
      const minDist = ENEMY_RADIUS * 2.6;
      if (distSq > minDist * minDist || distSq < 1e-6) continue;
      const push = (minDist - Math.sqrt(distSq)) / minDist;
      this.desired.x += (dx / Math.sqrt(distSq)) * push * speed * 1.4;
      this.desired.z += (dz / Math.sqrt(distSq)) * push * speed * 1.4;
    }

    if (this.desired.lengthSq() > 0.01) {
      this.probeScratch.copy(this.desired).normalize();
      this.avoidObstacles(context, this.probeHeading.copy(this.probeScratch), speed);
    }

    const accel = 12 * delta;
    this.velocity.x = THREE.MathUtils.damp(this.velocity.x, this.desired.x, accel, delta);
    this.velocity.z = THREE.MathUtils.damp(this.velocity.z, this.desired.z, accel, delta);
  }

  /**
   * Slides around an obstacle instead of grinding into it. Both sides are
   * probed so the enemy commits to whichever way is actually open.
   */
  private avoidObstacles(
    context: EnemyUpdateContext,
    heading: THREE.Vector3,
    speed: number,
  ): void {
    const probe = this.probeScratch;
    probe.copy(this.group.position).addScaledVector(heading, 1.5);
    if (context.collision.isClear(probe, ENEMY_RADIUS, ENEMY_HEIGHT)) return;

    this.strafeDirection.set(-heading.z, 0, heading.x).normalize();
    probe
      .copy(this.group.position)
      .addScaledVector(heading, 0.7)
      .addScaledVector(this.strafeDirection, 1.8);
    const leftOpen = context.collision.isClear(probe, ENEMY_RADIUS, ENEMY_HEIGHT);
    probe
      .copy(this.group.position)
      .addScaledVector(heading, 0.7)
      .addScaledVector(this.strafeDirection, -1.8);
    const rightOpen = context.collision.isClear(probe, ENEMY_RADIUS, ENEMY_HEIGHT);

    if (leftOpen && !rightOpen) this.strafeSign = 1;
    else if (rightOpen && !leftOpen) this.strafeSign = -1;

    this.desired.addScaledVector(this.strafeDirection, this.strafeSign * speed * 2.2);
  }

  private applyMotion(context: EnemyUpdateContext, delta: number): void {
    this.velocity.y -= 22 * delta;
    this.motion.copy(this.velocity).multiplyScalar(delta);
    context.collision.moveCharacter(
      this.group.position,
      this.motion,
      ENEMY_RADIUS,
      ENEMY_HEIGHT,
      this.velocity,
    );
  }

  /**
   * Safety net for an arena with no navmesh: an enemy that wants to move but
   * has not actually moved is wedged, so give up on the current waypoint and
   * try the other way around.
   */
  private detectStuck(delta: number): void {
    const wantsToMove = this.desired.lengthSq() > 0.5;
    const moved = this.group.position.distanceToSquared(this.lastPosition);
    this.lastPosition.copy(this.group.position);

    if (!wantsToMove || moved > 0.0009) {
      this.stuckTimer = 0;
      return;
    }

    this.stuckTimer += delta;
    if (this.stuckTimer < 1.4) return;
    this.stuckTimer = 0;
    this.strafeSign *= -1;
    if (this.exitPath.length > 0) this.exitPath.shift();
  }

  private faceTarget(context: EnemyUpdateContext, visible: boolean, delta: number): void {
    const facing = visible
      ? Math.atan2(context.playerPosition.x - this.group.position.x, context.playerPosition.z - this.group.position.z)
      : Math.atan2(this.velocity.x, this.velocity.z);
    if (!visible && this.velocity.lengthSq() < 0.05) return;
    let diff = facing - this.facingYaw;
    diff = ((diff + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI;
    this.facingYaw += diff * Math.min(1, delta * 8);
    this.group.rotation.y = this.facingYaw;
  }

  private animateLocomotion(delta: number): void {
    const speed = Math.hypot(this.velocity.x, this.velocity.z);
    this.walkPhase += speed * delta * 2.1;
    const swing = Math.sin(this.walkPhase) * Math.min(1, speed / 4) * 0.72;
    if (this.legs[0]) this.legs[0].rotation.x = swing;
    if (this.legs[1]) this.legs[1].rotation.x = -swing;
    if (this.arms[0]) this.arms[0].rotation.x = -swing * 0.35 - 0.95;
    if (this.arms[1]) this.arms[1].rotation.x = swing * 0.2 - 1.15;
    this.torsoGroup.position.y = Math.abs(Math.sin(this.walkPhase)) * 0.035 * Math.min(1, speed / 4);
  }

  private updateFiring(context: EnemyUpdateContext, distance: number, delta: number): void {
    if (this.burstRemaining > 0) {
      this.shotTimer -= delta;
      if (this.shotTimer <= 0) {
        this.shotTimer = this.archetypeRef.shotInterval;
        this.burstRemaining -= 1;
        this.fireShot(context, distance);
        if (this.burstRemaining <= 0) {
          this.burstCooldown = this.archetypeRef.burstPause * (0.75 + context.rng() * 0.5);
        }
      }
      return;
    }

    this.burstCooldown -= delta;
    if (this.burstCooldown <= 0) {
      this.burstRemaining = this.archetypeRef.burstSize;
      this.shotTimer = 0;
    }
  }

  private fireShot(context: EnemyUpdateContext, distance: number): void {
    this.muzzle.getWorldPosition(this.fireOrigin);
    this.fireDirection.copy(context.playerEye).sub(this.fireOrigin).normalize();

    // Aim error widens with range so distant enemies are a pressure source,
    // not a sniper wall.
    const error = this.archetypeRef.aimError * (0.6 + Math.min(1.4, distance / 28));
    this.fireDirection.x += (context.rng() - 0.5) * error * 2;
    this.fireDirection.y += (context.rng() - 0.5) * error * 2;
    this.fireDirection.z += (context.rng() - 0.5) * error * 2;
    this.fireDirection.normalize();

    this.flashTimer = 0.05;
    this.muzzleFlash.visible = true;
    context.onFire(this, this.fireOrigin, this.fireDirection);
  }

  private updateDeath(delta: number): void {
    this.deathTimer += delta;
    const t = Math.min(1, this.deathTimer / 0.75);
    this.group.rotation.x = -t * Math.PI * 0.46;
    this.group.position.y = Math.max(0, this.group.position.y - delta * 0.55);
    if (this.deathTimer > 2.6) {
      const fade = Math.min(1, (this.deathTimer - 2.6) / 0.8);
      this.group.position.y -= delta * 0.9;
      if (fade >= 1) this.finished = true;
    }
  }

  private buildBody(): THREE.Mesh {
    const a = this.archetypeRef;
    this.group.add(this.torsoGroup);

    const torso = new THREE.Mesh(
      mergeParts([
        { geometry: new THREE.BoxGeometry(0.42, 0.26, 0.26), color: a.uniform, position: [0, 0.88, 0] },
        { geometry: new THREE.BoxGeometry(0.5, 0.46, 0.3), color: a.uniform, position: [0, 1.24, 0] },
        { geometry: new THREE.BoxGeometry(0.46, 0.38, 0.12), color: a.vest, position: [0, 1.24, 0.14] },
        { geometry: new THREE.BoxGeometry(0.2, 0.1, 0.08), color: a.accent, position: [0.12, 1.05, 0.17] },
        { geometry: new THREE.CylinderGeometry(0.07, 0.08, 0.1, 8), color: '#8d6a4f', position: [0, 1.52, 0] },
      ]),
      SOLDIER_MATERIAL,
    );
    torso.castShadow = true;
    this.torsoGroup.add(torso);
    this.registerHitMesh(torso, 'body');

    const head = new THREE.Mesh(
      mergeParts([
        { geometry: new THREE.BoxGeometry(0.23, 0.26, 0.24), color: '#8d6a4f', position: [0, 1.67, 0] },
        {
          geometry: new THREE.SphereGeometry(0.155, 12, 10, 0, Math.PI * 2, 0, Math.PI * 0.62),
          color: a.vest,
          position: [0, 1.74, 0],
        },
        { geometry: new THREE.BoxGeometry(0.2, 0.05, 0.03), color: a.accent, position: [0, 1.7, 0.12] },
      ]),
      SOLDIER_MATERIAL,
    );
    head.castShadow = true;
    this.torsoGroup.add(head);
    this.registerHitMesh(head, 'head');

    for (let side = 0; side < 2; side += 1) {
      const sign = side === 0 ? -1 : 1;

      const armPivot = new THREE.Group();
      armPivot.position.set(sign * 0.32, 1.42, 0);
      this.torsoGroup.add(armPivot);
      this.arms.push(armPivot);

      const arm = new THREE.Mesh(
        mergeParts([
          { geometry: new THREE.BoxGeometry(0.13, 0.42, 0.14), color: a.uniform, position: [0, -0.2, 0] },
          {
            geometry: new THREE.BoxGeometry(0.115, 0.34, 0.125),
            color: a.uniform,
            position: [0, -0.42, 0.11],
            rotation: [0.55, 0, 0],
          },
        ]),
        SOLDIER_MATERIAL,
      );
      arm.castShadow = true;
      armPivot.add(arm);
      this.registerHitMesh(arm, 'limb');

      const legPivot = new THREE.Group();
      legPivot.position.set(sign * 0.13, 0.82, 0);
      this.group.add(legPivot);
      this.legs.push(legPivot);

      const leg = new THREE.Mesh(
        mergeParts([
          { geometry: new THREE.BoxGeometry(0.17, 0.44, 0.19), color: a.uniform, position: [0, -0.21, 0] },
          { geometry: new THREE.BoxGeometry(0.15, 0.4, 0.17), color: a.vest, position: [0, -0.62, 0] },
          { geometry: new THREE.BoxGeometry(0.16, 0.1, 0.25), color: a.vest, position: [0, -0.84, 0.04] },
        ]),
        SOLDIER_MATERIAL,
      );
      leg.castShadow = true;
      legPivot.add(leg);
      this.registerHitMesh(leg, 'limb');
    }

    // Rifle held in the right hand; the muzzle group is the tracer origin.
    const rifle = new THREE.Group();
    rifle.position.set(0.24, 1.26, 0.3);
    this.torsoGroup.add(rifle);

    const rifleMesh = new THREE.Mesh(
      mergeParts([
        { geometry: new THREE.BoxGeometry(0.07, 0.11, 0.6), color: '#1e2124', position: [0, 0, 0] },
        {
          geometry: new THREE.CylinderGeometry(0.018, 0.018, 0.3, 8),
          color: '#1e2124',
          position: [0, 0, 0.44],
          rotation: [Math.PI / 2, 0, 0],
        },
        { geometry: new THREE.BoxGeometry(0.05, 0.18, 0.06), color: '#1e2124', position: [0, -0.14, 0.02] },
      ]),
      SOLDIER_MATERIAL,
    );
    rifle.add(rifleMesh);

    this.muzzle.position.set(0, 0, 0.6);
    rifle.add(this.muzzle);

    const flash = new THREE.Mesh(
      new THREE.SphereGeometry(0.075, 8, 6),
      new THREE.MeshBasicMaterial({ color: '#ffd489', transparent: true, opacity: 0.9, depthWrite: false }),
    );
    flash.scale.z = 1.8;
    flash.visible = false;
    this.muzzle.add(flash);

    return flash;
  }

  private registerHitMesh(mesh: THREE.Mesh, zone: HitZone): void {
    mesh.userData.enemy = this;
    mesh.userData.hitZone = zone;
    mesh.userData.enemyHittable = true;
    this.hitMeshes.push(mesh);
  }
}
