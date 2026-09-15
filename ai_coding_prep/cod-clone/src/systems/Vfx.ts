import * as THREE from 'three';
import { createDecalTexture } from '../world/Textures';
import { disposeObject3D } from '../utils/dispose';

const TRACER_COUNT = 48;
const PARTICLE_COUNT = 900;
const DECAL_COUNT = 90;
const CASING_COUNT = 40;
const BLAST_COUNT = 6;

interface Tracer {
  readonly mesh: THREE.Mesh;
  life: number;
  maxLife: number;
}

interface Casing {
  readonly mesh: THREE.Mesh;
  readonly velocity: THREE.Vector3;
  readonly spin: THREE.Vector3;
  life: number;
}

interface Blast {
  readonly mesh: THREE.Mesh;
  readonly light: THREE.PointLight;
  life: number;
  radius: number;
}

/**
 * Pooled combat effects. Everything is allocated once at construction so the
 * firing path never allocates and never triggers a mid-fight GC pause.
 */
export class Vfx {
  readonly group = new THREE.Group();

  private readonly tracers: Tracer[] = [];
  private readonly decals: THREE.Mesh[] = [];
  private readonly casings: Casing[] = [];
  private readonly blasts: Blast[] = [];
  private decalCursor = 0;
  private casingCursor = 0;

  private readonly particlePositions = new Float32Array(PARTICLE_COUNT * 3);
  private readonly particleColors = new Float32Array(PARTICLE_COUNT * 3);
  private readonly particleBaseColors = new Float32Array(PARTICLE_COUNT * 3);
  private readonly particleVelocities = new Float32Array(PARTICLE_COUNT * 3);
  private readonly particleLife = new Float32Array(PARTICLE_COUNT);
  private readonly particleMaxLife = new Float32Array(PARTICLE_COUNT);
  private readonly particleGravity = new Float32Array(PARTICLE_COUNT);
  private readonly particles: THREE.Points;
  private particleCursor = 0;

  private readonly scratchA = new THREE.Vector3();
  private readonly scratchB = new THREE.Vector3();
  private readonly quaternion = new THREE.Quaternion();
  private readonly up = new THREE.Vector3(0, 1, 0);
  private readonly forward = new THREE.Vector3(0, 0, 1);

  private readonly impactDecal = createDecalTexture('rgba(18, 16, 14, 0.95)', 'rgba(30, 26, 22, 0.45)');
  private readonly bloodDecal = createDecalTexture('rgba(112, 16, 14, 0.92)', 'rgba(74, 10, 10, 0.4)');
  /** Replaced on seed() so effects stay reproducible under the test hooks. */
  private rng: () => number;

  constructor(rng: () => number) {
    this.rng = rng;
    const tracerGeometry = new THREE.CylinderGeometry(0.018, 0.018, 1, 5, 1, true);
    tracerGeometry.rotateX(Math.PI / 2);
    tracerGeometry.translate(0, 0, -0.5);
    for (let i = 0; i < TRACER_COUNT; i += 1) {
      const material = new THREE.MeshBasicMaterial({
        color: '#ffd98a',
        transparent: true,
        opacity: 0,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(tracerGeometry, material);
      mesh.visible = false;
      mesh.frustumCulled = false;
      this.group.add(mesh);
      this.tracers.push({ mesh, life: 0, maxLife: 0.055 });
    }

    const particleGeometry = new THREE.BufferGeometry();
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(this.particlePositions, 3));
    particleGeometry.setAttribute('color', new THREE.BufferAttribute(this.particleColors, 3));
    particleGeometry.setDrawRange(0, PARTICLE_COUNT);
    this.particles = new THREE.Points(
      particleGeometry,
      new THREE.PointsMaterial({
        size: 0.07,
        vertexColors: true,
        transparent: true,
        opacity: 0.95,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true,
      }),
    );
    this.particles.frustumCulled = false;
    // Parked far below the map until a burst claims them.
    for (let i = 0; i < PARTICLE_COUNT; i += 1) this.particlePositions[i * 3 + 1] = -1000;
    this.group.add(this.particles);

    const decalGeometry = new THREE.PlaneGeometry(1, 1);
    for (let i = 0; i < DECAL_COUNT; i += 1) {
      const mesh = new THREE.Mesh(
        decalGeometry,
        new THREE.MeshBasicMaterial({
          map: this.impactDecal,
          transparent: true,
          opacity: 0,
          depthWrite: false,
          polygonOffset: true,
          polygonOffsetFactor: -4,
        }),
      );
      mesh.visible = false;
      this.group.add(mesh);
      this.decals.push(mesh);
    }

    const casingGeometry = new THREE.CylinderGeometry(0.012, 0.012, 0.042, 6);
    const casingMaterial = new THREE.MeshStandardMaterial({
      color: '#c8a352',
      roughness: 0.34,
      metalness: 0.9,
    });
    for (let i = 0; i < CASING_COUNT; i += 1) {
      const mesh = new THREE.Mesh(casingGeometry, casingMaterial);
      mesh.visible = false;
      this.group.add(mesh);
      this.casings.push({
        mesh,
        velocity: new THREE.Vector3(),
        spin: new THREE.Vector3(),
        life: 0,
      });
    }

    const blastGeometry = new THREE.SphereGeometry(1, 16, 12);
    for (let i = 0; i < BLAST_COUNT; i += 1) {
      const mesh = new THREE.Mesh(
        blastGeometry,
        new THREE.MeshBasicMaterial({
          color: '#ffb04a',
          transparent: true,
          opacity: 0,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
        }),
      );
      mesh.visible = false;
      this.group.add(mesh);
      const light = new THREE.PointLight('#ffa23c', 0, 26, 2);
      light.visible = false;
      this.group.add(light);
      this.blasts.push({ mesh, light, life: 0, radius: 1 });
    }
  }

  setRandom(rng: () => number): void {
    this.rng = rng;
  }

  tracer(from: THREE.Vector3, to: THREE.Vector3, color: THREE.ColorRepresentation, width = 1): void {
    const tracer = this.tracers.find((candidate) => !candidate.mesh.visible) ?? this.tracers[0];
    const distance = this.scratchA.copy(to).sub(from).length();
    if (distance < 0.05) return;

    tracer.mesh.position.copy(from);
    this.quaternion.setFromUnitVectors(this.forward, this.scratchA.normalize());
    tracer.mesh.quaternion.copy(this.quaternion);
    // The cylinder is unit-length along -Z, so scaling Z stretches it to the hit.
    tracer.mesh.scale.set(width, width, -distance);
    tracer.mesh.visible = true;
    tracer.life = tracer.maxLife;
    const material = tracer.mesh.material as THREE.MeshBasicMaterial;
    material.color.set(color);
    material.opacity = 0.9;
  }

  impact(point: THREE.Vector3, normal: THREE.Vector3, color: THREE.ColorRepresentation, count = 9): void {
    const base = new THREE.Color(color);
    for (let i = 0; i < count; i += 1) {
      this.spawnParticle(point, normal, base, 3.2, 0.32, 9);
    }
    this.spawnDecal(point, normal, this.impactDecal, 0.16 + this.rng() * 0.06, 0.85);
  }

  blood(point: THREE.Vector3, normal: THREE.Vector3, count = 12): void {
    const base = new THREE.Color('#9d1d17');
    for (let i = 0; i < count; i += 1) {
      this.spawnParticle(point, normal, base, 2.4, 0.42, 7);
    }
  }

  bloodPool(point: THREE.Vector3): void {
    this.spawnDecal(point, this.up, this.bloodDecal, 1.1, 0.72);
  }

  casing(origin: THREE.Vector3, right: THREE.Vector3, up: THREE.Vector3): void {
    const casing = this.casings[this.casingCursor];
    this.casingCursor = (this.casingCursor + 1) % CASING_COUNT;
    casing.mesh.position.copy(origin);
    casing.mesh.visible = true;
    casing.life = 1.6;
    casing.velocity
      .copy(right)
      .multiplyScalar(2.1 + this.rng() * 1.4)
      .addScaledVector(up, 1.4 + this.rng() * 0.9);
    casing.spin.set((this.rng() - 0.5) * 26, (this.rng() - 0.5) * 26, (this.rng() - 0.5) * 26);
  }

  explosion(point: THREE.Vector3, radius: number): void {
    const blast = this.blasts.find((candidate) => candidate.life <= 0) ?? this.blasts[0];
    blast.mesh.position.copy(point);
    blast.mesh.visible = true;
    blast.mesh.scale.setScalar(0.4);
    blast.light.position.copy(point);
    blast.light.visible = true;
    blast.life = 0.52;
    blast.radius = radius;
    (blast.mesh.material as THREE.MeshBasicMaterial).opacity = 0.92;

    const smoke = new THREE.Color('#5a5045');
    const fire = new THREE.Color('#ffa338');
    for (let i = 0; i < 90; i += 1) {
      const color = i % 3 === 0 ? smoke : fire;
      this.spawnParticle(point, this.up, color, 11, 1.1, 5.5);
    }
  }

  update(delta: number): void {
    for (const tracer of this.tracers) {
      if (!tracer.mesh.visible) continue;
      tracer.life -= delta;
      const material = tracer.mesh.material as THREE.MeshBasicMaterial;
      if (tracer.life <= 0) {
        tracer.mesh.visible = false;
        material.opacity = 0;
        continue;
      }
      material.opacity = 0.9 * (tracer.life / tracer.maxLife);
    }

    this.updateParticles(delta);

    for (const decal of this.decals) {
      if (!decal.visible) continue;
      const material = decal.material as THREE.MeshBasicMaterial;
      const remaining = (decal.userData.life as number) - delta;
      decal.userData.life = remaining;
      if (remaining <= 0) {
        decal.visible = false;
        material.opacity = 0;
        continue;
      }
      // Hold full strength, then fade out over the final second.
      material.opacity = (decal.userData.peak as number) * Math.min(1, remaining);
    }

    for (const casing of this.casings) {
      if (!casing.mesh.visible) continue;
      casing.life -= delta;
      if (casing.life <= 0) {
        casing.mesh.visible = false;
        continue;
      }
      casing.velocity.y -= 16 * delta;
      casing.mesh.position.addScaledVector(casing.velocity, delta);
      casing.mesh.rotation.x += casing.spin.x * delta;
      casing.mesh.rotation.y += casing.spin.y * delta;
      casing.mesh.rotation.z += casing.spin.z * delta;
      if (casing.mesh.position.y < 0.02) {
        casing.mesh.position.y = 0.02;
        casing.velocity.multiplyScalar(0.3);
        casing.velocity.y = Math.abs(casing.velocity.y) * 0.35;
        casing.spin.multiplyScalar(0.4);
      }
    }

    for (const blast of this.blasts) {
      if (blast.life <= 0) continue;
      blast.life -= delta;
      const t = Math.max(0, blast.life / 0.52);
      const material = blast.mesh.material as THREE.MeshBasicMaterial;
      if (blast.life <= 0) {
        blast.mesh.visible = false;
        blast.light.visible = false;
        material.opacity = 0;
        blast.light.intensity = 0;
        continue;
      }
      blast.mesh.scale.setScalar(blast.radius * (1.25 - t * 0.95));
      material.opacity = 0.92 * t * t;
      blast.light.intensity = 40 * t;
    }
  }

  clearTransient(): void {
    for (const tracer of this.tracers) {
      tracer.mesh.visible = false;
      tracer.life = 0;
    }
    for (const decal of this.decals) {
      decal.visible = false;
      decal.userData.life = 0;
    }
    for (const casing of this.casings) {
      casing.mesh.visible = false;
      casing.life = 0;
    }
    for (const blast of this.blasts) {
      blast.mesh.visible = false;
      blast.light.visible = false;
      blast.light.intensity = 0;
      blast.life = 0;
    }
    for (let i = 0; i < PARTICLE_COUNT; i += 1) {
      this.particleLife[i] = 0;
      this.particlePositions[i * 3 + 1] = -1000;
    }
    this.particles.geometry.attributes.position.needsUpdate = true;
  }

  dispose(): void {
    disposeObject3D(this.group);
    this.impactDecal.dispose();
    this.bloodDecal.dispose();
  }

  private spawnParticle(
    origin: THREE.Vector3,
    normal: THREE.Vector3,
    color: THREE.Color,
    speed: number,
    life: number,
    gravity: number,
  ): void {
    const index = this.particleCursor;
    this.particleCursor = (this.particleCursor + 1) % PARTICLE_COUNT;
    const offset = index * 3;

    this.particlePositions[offset] = origin.x;
    this.particlePositions[offset + 1] = origin.y;
    this.particlePositions[offset + 2] = origin.z;

    // Scatter into the hemisphere around the surface normal.
    const spread = 0.85;
    this.scratchB.set(
      normal.x + (this.rng() - 0.5) * spread * 2,
      normal.y + (this.rng() - 0.5) * spread * 2,
      normal.z + (this.rng() - 0.5) * spread * 2,
    );
    if (this.scratchB.lengthSq() < 1e-5) this.scratchB.copy(normal);
    this.scratchB.normalize().multiplyScalar(speed * (0.35 + this.rng() * 0.9));

    this.particleVelocities[offset] = this.scratchB.x;
    this.particleVelocities[offset + 1] = this.scratchB.y;
    this.particleVelocities[offset + 2] = this.scratchB.z;

    this.particleColors[offset] = color.r;
    this.particleColors[offset + 1] = color.g;
    this.particleColors[offset + 2] = color.b;
    this.particleBaseColors[offset] = color.r;
    this.particleBaseColors[offset + 1] = color.g;
    this.particleBaseColors[offset + 2] = color.b;

    const total = life * (0.6 + this.rng() * 0.8);
    this.particleLife[index] = total;
    this.particleMaxLife[index] = total;
    this.particleGravity[index] = gravity;
  }

  private updateParticles(delta: number): void {
    let dirty = false;
    for (let i = 0; i < PARTICLE_COUNT; i += 1) {
      if (this.particleLife[i] <= 0) continue;
      dirty = true;
      this.particleLife[i] -= delta;
      const offset = i * 3;
      if (this.particleLife[i] <= 0) {
        this.particlePositions[offset + 1] = -1000;
        this.particleColors[offset] = 0;
        this.particleColors[offset + 1] = 0;
        this.particleColors[offset + 2] = 0;
        continue;
      }
      this.particleVelocities[offset + 1] -= this.particleGravity[i] * delta;
      this.particlePositions[offset] += this.particleVelocities[offset] * delta;
      this.particlePositions[offset + 1] += this.particleVelocities[offset + 1] * delta;
      this.particlePositions[offset + 2] += this.particleVelocities[offset + 2] * delta;

      // Additive points fade by dimming their vertex colour toward black.
      const fade = this.particleLife[i] / this.particleMaxLife[i];
      this.particleColors[offset] = this.particleBaseColors[offset] * fade;
      this.particleColors[offset + 1] = this.particleBaseColors[offset + 1] * fade;
      this.particleColors[offset + 2] = this.particleBaseColors[offset + 2] * fade;
    }
    if (dirty) {
      this.particles.geometry.attributes.position.needsUpdate = true;
      this.particles.geometry.attributes.color.needsUpdate = true;
    }
  }

  private spawnDecal(
    point: THREE.Vector3,
    normal: THREE.Vector3,
    map: THREE.Texture,
    size: number,
    peak: number,
  ): void {
    const decal = this.decals[this.decalCursor];
    this.decalCursor = (this.decalCursor + 1) % DECAL_COUNT;
    decal.position.copy(point).addScaledVector(normal, 0.015);
    this.scratchA.copy(point).addScaledVector(normal, 1);
    decal.lookAt(this.scratchA);
    decal.scale.setScalar(size);
    decal.visible = true;
    const material = decal.material as THREE.MeshBasicMaterial;
    material.map = map;
    material.opacity = peak;
    material.needsUpdate = true;
    decal.userData.life = 12;
    decal.userData.peak = peak;
  }
}
