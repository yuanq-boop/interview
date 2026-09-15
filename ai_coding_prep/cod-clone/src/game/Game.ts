import * as THREE from 'three';
import { InputController } from '../core/InputController';
import { Loop } from '../core/Loop';
import { createRenderer, resizeRenderer } from '../core/Renderer';
import { AmmoCrate } from '../entities/AmmoCrate';
import { Enemy, ENEMY_ARCHETYPES, type EnemyArchetypeId } from '../entities/Enemy';
import { blastDamage, Grenade, GRENADE_RADIUS } from '../entities/Grenade';
import { Player } from '../entities/Player';
import { AudioSystem } from '../systems/AudioSystem';
import { CameraRig } from '../systems/CameraRig';
import { CollisionSystem } from '../systems/CollisionSystem';
import { CombatSystem, zoneMultiplier } from '../systems/CombatSystem';
import { DebugTools, type GameTuning } from '../systems/DebugTools';
import { Hud, streakMultiplier } from '../systems/Hud';
import { Vfx } from '../systems/Vfx';
import { WaveDirector } from '../systems/WaveDirector';
import { createSeededRandom } from '../utils/random';
import { Weapon } from '../weapons/Weapon';
import { damageAtDistance, WEAPON_DEFS } from '../weapons/WeaponDefs';
import { ViewModel } from '../weapons/ViewModel';
import { buildCompound, type CompoundBuild, type EnemySpawn } from '../world/Compound';
import { createSkyTexture } from '../world/Textures';

type Phase = 'menu' | 'playing' | 'paused' | 'dead';

interface RunStats {
  score: number;
  kills: number;
  headshots: number;
  shotsFired: number;
  shotsHit: number;
  streak: number;
  bestStreak: number;
  grenades: number;
}

interface ScriptedInput {
  move: THREE.Vector2;
  look: THREE.Vector2;
  fire: boolean;
  ads: boolean;
  reload: boolean;
  active: boolean;
}

const MAX_GRENADES = 2;
const LOS_REFRESH = 0.12;
const WORLD_UP = new THREE.Vector3(0, 1, 0);

export class Game {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(92, 1, 0.08, 260);
  private readonly input: InputController;
  private readonly player = new Player();
  private readonly cameraRig = new CameraRig(this.camera);
  private readonly collision = new CollisionSystem();
  private readonly combat = new CombatSystem();
  private readonly viewModel = new ViewModel();
  private readonly waves = new WaveDirector();
  private readonly hud = new Hud();

  private rng = createSeededRandom(1);
  private readonly vfx = new Vfx(() => this.rng());
  private readonly audio = new AudioSystem(() => this.rng());

  private readonly weapons = WEAPON_DEFS.map((def) => new Weapon(def));
  private activeWeaponIndex = 0;
  private readonly enemies: Enemy[] = [];
  private readonly grenades: Grenade[] = [];
  private readonly ammoCrates: AmmoCrate[] = [];
  private readonly enemyMeshes: THREE.Mesh[] = [];
  private readonly losCache = new Map<Enemy, { value: boolean; timer: number }>();
  private compound!: CompoundBuild;
  private sun!: THREE.DirectionalLight;

  private readonly tuning: GameTuning = {
    sensitivity: 0.0022,
    baseFov: 92,
    exposure: 1.08,
    maxDpr: 2,
    shadows: true,
    godMode: false,
  };
  private readonly debugTools: DebugTools;

  private readonly loop = new Loop(
    (delta) => this.update(delta),
    () => this.render(),
  );

  private phase: Phase = 'menu';
  private stats: RunStats = createStats();
  private frame = 0;
  private elapsed = 0;
  private adsFraction = 0;
  private timeScale = 1;
  private hitstopRemaining = 0;
  private pausedForScreenshot = false;
  private reducedMotion = false;
  private captureMode = false;
  private spawnCursor = 0;
  private lastFrameState = { ads: false, sprinting: false, moveFraction: 0 };

  private readonly scripted: ScriptedInput = {
    move: new THREE.Vector2(),
    look: new THREE.Vector2(),
    fire: false,
    ads: false,
    reload: false,
    active: false,
  };

  // Scratch vectors; the combat path runs many times per second and must not allocate.
  private readonly forward = new THREE.Vector3();
  private readonly right = new THREE.Vector3();
  private readonly up = new THREE.Vector3();
  private readonly muzzle = new THREE.Vector3();
  private readonly shotDirection = new THREE.Vector3();
  private readonly scratch = new THREE.Vector3();
  private readonly scratchB = new THREE.Vector3();
  private readonly lookDelta = new THREE.Vector2();

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.renderer = createRenderer(canvas);
    this.renderer.toneMappingExposure = this.tuning.exposure;

    this.input = new InputController(canvas, {
      stick: this.element('#touch-stick'),
      knob: this.element('#touch-knob'),
      lookPad: this.element('#touch-look'),
      fire: this.element('#touch-fire'),
      ads: this.element('#touch-ads'),
      reload: this.element('#touch-reload'),
      jump: this.element('#touch-jump'),
      crouch: this.element('#touch-crouch'),
      swap: this.element('#touch-swap'),
      grenade: this.element('#touch-grenade'),
    });

    this.debugTools = new DebugTools(this.tuning, () => {
      this.renderer.toneMappingExposure = this.tuning.exposure;
      this.renderer.shadowMap.enabled = this.tuning.shadows;
      this.sun.castShadow = this.tuning.shadows;
      resizeRenderer(this.renderer, this.camera, this.tuning.maxDpr);
    });

    this.createScene();
    this.viewModel.setWeapon(this.activeWeapon.def);
    this.hud.buildWeaponSlots(this.weapons);
    this.hud.setActiveWeapon(this.activeWeaponIndex);
    this.hud.onAction(() => this.onOverlayAction());
    this.showMenu();

    resizeRenderer(this.renderer, this.camera, this.tuning.maxDpr);
    this.canvas.addEventListener('mousedown', this.onCanvasMouseDown);
    document.addEventListener('pointerlockchange', this.onPointerLockChange);
    window.addEventListener('keydown', this.onGlobalKeyDown);

    this.player.reset(this.compound.playerSpawn, this.compound.playerSpawnYaw);
    this.cameraRig.snapTo(this.player, this.tuning.baseFov);
    this.installTestHooks();
    this.publishDiagnostics();
  }

  start(): void {
    this.loop.start();
  }

  dispose(): void {
    this.loop.stop();
    this.canvas.removeEventListener('mousedown', this.onCanvasMouseDown);
    document.removeEventListener('pointerlockchange', this.onPointerLockChange);
    window.removeEventListener('keydown', this.onGlobalKeyDown);
    this.input.dispose();
    this.audio.dispose();
    this.debugTools.dispose();
    this.viewModel.dispose();
    this.vfx.dispose();
    for (const enemy of this.enemies) enemy.dispose();
    for (const grenade of this.grenades) grenade.dispose();
    for (const crate of this.ammoCrates) crate.dispose();
    this.renderer.dispose();
    window.__THREE_GAME_DIAGNOSTICS__ = undefined;
    window.__THREE_GAME_TEST_HOOKS__ = undefined;
  }

  private get activeWeapon(): Weapon {
    return this.weapons[this.activeWeaponIndex];
  }

  // ----------------------------------------------------------------- update

  private update(delta: number): void {
    this.frame += 1;
    resizeRenderer(this.renderer, this.camera, this.tuning.maxDpr);
    this.viewModel.setAspect(this.camera.aspect);

    if (this.pausedForScreenshot) {
      this.publishDiagnostics();
      return;
    }

    if (this.phase === 'playing') {
      this.simulate(delta);
    } else {
      // Menus still breathe: ambient crate pulses keep the scene from looking frozen.
      const ambientDelta = this.reducedMotion ? 0 : delta;
      for (const crate of this.ammoCrates) crate.update(ambientDelta, this.elapsed);
      this.vfx.update(ambientDelta);
    }

    this.input.endFrame();
    this.publishDiagnostics();
  }

  private simulate(delta: number): void {
    if (this.hitstopRemaining > 0) {
      this.hitstopRemaining -= delta;
      if (this.hitstopRemaining <= 0) {
        this.timeScale = 1;
        this.audio.setDuck(1);
      }
    }
    const gameplayDelta = delta * this.timeScale;
    this.elapsed += gameplayDelta;

    this.readScriptedLook();
    this.player.applyLook(this.input, this.tuning.sensitivity);

    const weapon = this.activeWeapon;
    const adsHeld = this.scripted.active ? this.scripted.ads : this.input.isHeld('ads');
    const state = this.player.update(
      gameplayDelta,
      this.input,
      this.collision,
      !weapon.isReloading,
      adsHeld,
    );
    this.lastFrameState = {
      ads: state.ads,
      sprinting: state.sprinting,
      moveFraction: state.moveSpeedFraction,
    };
    if (state.stepped) this.audio.footstep(state.sprinting);
    if (state.landed) this.cameraRig.addTrauma(0.12);

    this.adsFraction = THREE.MathUtils.damp(
      this.adsFraction,
      state.ads ? 1 : 0,
      1 / Math.max(0.02, weapon.def.adsSeconds),
      gameplayDelta,
    );

    this.updateWeapons(gameplayDelta);
    this.updateEnemies(gameplayDelta);
    this.updateGrenades(gameplayDelta);
    this.updateWaves(gameplayDelta);

    for (const crate of this.ammoCrates) {
      crate.update(this.reducedMotion ? 0 : gameplayDelta, this.elapsed);
      if (crate.tryConsume(this.player.position)) this.resupply();
    }

    this.cameraRig.update(delta, this.player, {
      ads: state.ads,
      adsFraction: this.adsFraction,
      sprinting: state.sprinting,
      moveFraction: state.moveSpeedFraction,
      landed: state.landed,
      baseFov: this.tuning.baseFov,
      adsFov: weapon.def.adsFov,
    });

    this.viewModel.update(delta, {
      adsFraction: this.adsFraction,
      moveFraction: state.moveSpeedFraction,
      bobPhase: this.player.bobPhase,
      sprinting: state.sprinting,
      reloading: weapon.isReloading,
      reloadProgress: weapon.reloadProgress,
      lookDelta: this.lookDelta,
      grounded: this.player.isGrounded,
    });

    this.vfx.update(delta);
    this.updateHud();
  }

  private readScriptedLook(): void {
    if (!this.scripted.active) return;
    this.input.setScriptedMovement(this.scripted.move);
    // Bot input feeds the same yaw/pitch path a mouse would.
    this.player.yaw -= this.scripted.look.x * this.tuning.sensitivity;
    this.player.pitch = THREE.MathUtils.clamp(
      this.player.pitch - this.scripted.look.y * this.tuning.sensitivity,
      -1.5,
      1.5,
    );
    this.scripted.look.set(0, 0);
  }

  private updateWeapons(delta: number): void {
    const weapon = this.activeWeapon;
    const reloadFinished = weapon.update(delta);
    if (reloadFinished) this.audio.reload('in');

    const slotRequest = this.input.consumeSlotRequest();
    if (slotRequest !== null && slotRequest < this.weapons.length) {
      this.selectWeapon(slotRequest);
    } else if (this.input.wasPressed('swap')) {
      this.selectWeapon((this.activeWeaponIndex + 1) % this.weapons.length);
    }

    if (this.input.wasPressed('reload') || (this.scripted.active && this.scripted.reload)) {
      this.scripted.reload = false;
      if (this.activeWeapon.startReload()) this.audio.reload('out');
    }

    if (this.input.wasPressed('grenade')) this.throwGrenade();

    const firing = this.scripted.active ? this.scripted.fire : this.input.isHeld('fire');
    if (!firing) {
      this.activeWeapon.releaseTrigger();
      return;
    }
    if (this.lastFrameState.sprinting) return;

    const current = this.activeWeapon;
    if (current.isReloading) return;
    if (current.isEmpty) {
      if (this.input.wasPressed('fire') || this.scripted.active) {
        this.audio.dryFire();
        if (current.startReload()) this.audio.reload('out');
      }
      return;
    }
    if (current.tryFire()) this.fireShot(current);
  }

  private fireShot(weapon: Weapon): void {
    const def = weapon.def;
    this.stats.shotsFired += 1;

    // Aim comes from the player's own yaw/pitch, not the camera matrix: the
    // camera is not rewritten until cameraRig.update() later this frame, so
    // reading it here would fire along the previous frame's direction.
    this.aimBasis();

    const eye = this.player.eyePosition;
    this.viewModel.muzzleWorldPosition(
      this.muzzle,
      eye,
      this.forward,
      this.right,
      this.up,
      this.adsFraction,
    );

    const spread = weapon.spread(this.adsFraction > 0.6, this.lastFrameState.moveFraction);
    let registeredHit = false;

    for (let pellet = 0; pellet < def.pellets; pellet += 1) {
      this.shotDirection.copy(this.forward);
      if (spread > 0) {
        // Uniform disc sample mapped onto the aim cone.
        const angle = this.rng() * Math.PI * 2;
        const radius = Math.sqrt(this.rng()) * spread;
        this.shotDirection
          .addScaledVector(this.right, Math.cos(angle) * radius)
          .addScaledVector(this.up, Math.sin(angle) * radius)
          .normalize();
      }

      const hit = this.combat.resolveShot(
        eye,
        this.shotDirection,
        def.range,
        this.compound.raycastTargets,
        this.enemyMeshes,
      );

      const endPoint = hit
        ? hit.point
        : this.scratch.copy(eye).addScaledVector(this.shotDirection, def.range);
      this.vfx.tracer(this.muzzle, endPoint, '#ffe6a8', 1);

      if (!hit) continue;

      if (hit.enemy && hit.zone) {
        registeredHit = true;
        this.applyBulletDamage(hit.enemy, hit.zone, hit.distance, weapon, hit.point, hit.normal);
      } else {
        this.vfx.impact(hit.point, hit.normal, '#ffd9a0', 7);
        this.audio.impact();
      }
    }

    if (registeredHit) this.stats.shotsHit += 1;

    this.player.applyRecoil(
      def.recoilPitch * (this.adsFraction > 0.6 ? 0.62 : 1),
      (this.rng() - 0.5) * 2 * def.recoilYaw,
    );
    this.cameraRig.addTrauma(def.mode === 'pump' ? 0.3 : 0.11);
    this.cameraRig.punchFov(def.mode === 'pump' ? 3.4 : 1.1);
    this.viewModel.fire(def.recoilPitch);
    this.audio.playerShot(def.audio);
    this.vfx.casing(
      this.scratchB.copy(this.muzzle).addScaledVector(this.forward, -0.25),
      this.right,
      this.up,
    );
  }

  private applyBulletDamage(
    enemy: Enemy,
    zone: 'head' | 'body' | 'limb',
    distance: number,
    weapon: Weapon,
    point: THREE.Vector3,
    normal: THREE.Vector3,
  ): void {
    const base = damageAtDistance(weapon.def, distance);
    const damage = base * zoneMultiplier(zone, weapon.def.headshotMultiplier);
    const headshot = zone === 'head';
    const killed = enemy.applyDamage(damage);

    this.vfx.blood(point, normal, headshot ? 16 : 10);
    if (killed) {
      this.onEnemyKilled(enemy, headshot, point);
    } else {
      this.hud.showHitMarker(headshot ? 'head' : 'body');
      this.audio.hitMarker(headshot);
    }
  }

  private onEnemyKilled(enemy: Enemy, headshot: boolean, point: THREE.Vector3): void {
    this.stats.kills += 1;
    this.stats.streak += 1;
    this.stats.bestStreak = Math.max(this.stats.bestStreak, this.stats.streak);
    if (headshot) this.stats.headshots += 1;

    const multiplier = streakMultiplier(this.stats.streak);
    const points = Math.round(enemy.archetype.score * (headshot ? 1.5 : 1) * multiplier);
    this.stats.score += points;

    this.hud.showHitMarker('kill');
    this.hud.addKill(enemy.archetype.name, headshot ? 'headshot' : undefined);
    this.audio.enemyDown();
    this.vfx.bloodPool(this.scratch.set(enemy.group.position.x, 0.02, enemy.group.position.z));
    this.vfx.blood(point, this.up.set(0, 1, 0), 20);
    this.hitstop(headshot ? 0.075 : 0.045);
    this.cameraRig.addTrauma(0.16);
    this.removeEnemyMeshes(enemy);
  }

  private updateEnemies(delta: number): void {
    const playerEye = this.player.eyePosition;

    for (let i = this.enemies.length - 1; i >= 0; i -= 1) {
      const enemy = this.enemies[i];
      enemy.update({
        delta,
        playerEye,
        playerPosition: this.player.position,
        playerAlive: this.player.alive,
        collision: this.collision,
        rng: () => this.rng(),
        hasLineOfSight: (from, to) => this.cachedLineOfSight(enemy, from, to, delta),
        onFire: (source, origin, direction) => this.onEnemyFire(source, origin, direction),
        others: this.enemies,
      });

      if (enemy.finished) {
        this.scene.remove(enemy.group);
        enemy.dispose();
        this.losCache.delete(enemy);
        this.enemies.splice(i, 1);
      }
    }
  }

  private cachedLineOfSight(
    enemy: Enemy,
    from: THREE.Vector3,
    to: THREE.Vector3,
    delta: number,
  ): boolean {
    // A full raycast per enemy per frame is wasted work; refresh on a stagger.
    let entry = this.losCache.get(enemy);
    if (!entry) {
      entry = { value: false, timer: this.rng() * LOS_REFRESH };
      this.losCache.set(enemy, entry);
    }
    entry.timer -= delta;
    if (entry.timer <= 0) {
      entry.timer = LOS_REFRESH;
      entry.value = this.combat.hasLineOfSight(from, to, this.compound.raycastTargets);
    }
    return entry.value;
  }

  private onEnemyFire(enemy: Enemy, origin: THREE.Vector3, direction: THREE.Vector3): void {
    const worldHit = this.combat.resolveShot(
      origin,
      direction,
      120,
      this.compound.raycastTargets,
      [],
    );
    const maxDistance = worldHit ? worldHit.distance : 120;

    const hitPlayer = this.rayHitsPlayer(origin, direction, maxDistance);
    const endPoint = hitPlayer
      ? this.scratch.copy(origin).addScaledVector(direction, hitPlayer)
      : worldHit
        ? worldHit.point
        : this.scratch.copy(origin).addScaledVector(direction, 120);

    this.vfx.tracer(origin, endPoint, '#ff9a4c', 0.85);
    this.audio.enemyShot(origin.distanceTo(this.player.position));

    if (hitPlayer !== null) {
      this.damagePlayer(enemy.archetype.damage, enemy.group.position, enemy.archetype.name);
    } else if (worldHit) {
      this.vfx.impact(worldHit.point, worldHit.normal, '#ffcf9a', 5);
    }
  }

  /** Distance along the ray at which it strikes the player, or null. */
  private rayHitsPlayer(origin: THREE.Vector3, direction: THREE.Vector3, maxDistance: number): number | null {
    if (!this.player.alive) return null;
    const hitRadius = 0.42;
    const feetY = this.player.position.y;
    const headY = feetY + this.player.height;

    this.scratchB.copy(this.player.position).sub(origin);
    const along = this.scratchB.dot(direction);
    if (along <= 0 || along > maxDistance) return null;

    this.scratch.copy(origin).addScaledVector(direction, along);
    // Clamp the closest point to the player's vertical extent, then measure.
    const clampedY = THREE.MathUtils.clamp(this.scratch.y, feetY, headY);
    const dx = this.scratch.x - this.player.position.x;
    const dz = this.scratch.z - this.player.position.z;
    const dy = this.scratch.y - clampedY;
    const distanceSq = dx * dx + dz * dz + dy * dy;
    return distanceSq <= hitRadius * hitRadius ? along : null;
  }

  private damagePlayer(amount: number, source: THREE.Vector3, sourceName: string): void {
    if (this.tuning.godMode || !this.player.alive) return;

    const died = this.player.takeDamage(amount);
    if (died && this.captureMode) {
      // Scripted capture states need a live firefight, not a death card. Damage
      // still lands and still shows on the HUD; it just cannot finish the job.
      this.player.health = 18;
      this.player.alive = true;
      return;
    }
    this.audio.playerHurt();
    this.cameraRig.addTrauma(0.26);

    // Bearing of the shooter relative to where the player is looking.
    const angle =
      Math.atan2(source.x - this.player.position.x, source.z - this.player.position.z) -
      this.player.yaw +
      Math.PI;
    this.hud.showDamageFrom(angle);
    this.cameraRig.addDamageRoll(Math.sin(angle) * 0.045);
    this.stats.streak = 0;

    if (died) this.onPlayerDeath(sourceName);
  }

  private updateGrenades(delta: number): void {
    for (let i = this.grenades.length - 1; i >= 0; i -= 1) {
      const grenade = this.grenades[i];
      grenade.update(delta, this.collision);
      if (grenade.fuse > 0) continue;

      this.detonate(grenade.group.position);
      this.scene.remove(grenade.group);
      grenade.dispose();
      this.grenades.splice(i, 1);
    }
  }

  private detonate(point: THREE.Vector3): void {
    this.vfx.explosion(point, GRENADE_RADIUS * 0.34);
    this.audio.explosion();
    this.cameraRig.addTrauma(0.7);
    this.cameraRig.punchFov(7);
    this.hitstop(0.09);

    for (const enemy of this.enemies) {
      if (!enemy.alive) continue;
      const distance = enemy.group.position.distanceTo(point);
      if (distance > GRENADE_RADIUS) continue;
      // Cover blocks the blast, so a grenade cannot kill through a wall.
      if (!this.combat.hasLineOfSight(point, enemy.eyePosition, this.compound.raycastTargets)) continue;
      const damage = blastDamage(distance);
      if (damage <= 0) continue;
      if (enemy.applyDamage(damage)) {
        this.onEnemyKilled(enemy, false, enemy.eyePosition.clone());
      }
    }

    const playerDistance = this.player.position.distanceTo(point);
    if (playerDistance < GRENADE_RADIUS) {
      const selfDamage = blastDamage(playerDistance) * 0.45;
      if (selfDamage > 1) this.damagePlayer(selfDamage, point, 'own grenade');
    }
  }

  /** Writes the current forward/right/up aim basis from the player's view angles. */
  private aimBasis(): void {
    this.player.forwardVector(this.forward);
    this.right.crossVectors(this.forward, WORLD_UP).normalize();
    this.up.crossVectors(this.right, this.forward).normalize();
  }

  private throwGrenade(): void {
    if (this.stats.grenades <= 0) return;
    this.stats.grenades -= 1;

    this.aimBasis();
    const origin = this.player.eyePosition.addScaledVector(this.forward, 0.5);
    const grenade = new Grenade(origin, this.forward, 17, () => this.rng());
    this.grenades.push(grenade);
    this.scene.add(grenade.group);
    this.hud.updateGrenades(this.stats.grenades);
    this.hud.showToast('Frag out');
  }

  private updateWaves(delta: number): void {
    const aliveCount = this.enemies.filter((enemy) => enemy.alive).length;
    const result = this.waves.update(delta, aliveCount, (archetype) => this.spawnEnemy(archetype));

    if (result === 'wave-cleared') {
      this.hud.showWaveBanner(`Wave ${this.waves.wave} cleared`);
      this.hud.showToast('Resupply and hold');
      this.stats.grenades = Math.min(MAX_GRENADES, this.stats.grenades + 1);
      this.hud.updateGrenades(this.stats.grenades);
      for (const crate of this.ammoCrates) crate.reset();
    } else if (result === 'wave-started') {
      this.hud.showWaveBanner(`Wave ${this.waves.wave}`);
      this.audio.waveStart(this.waves.wave);
    }
  }

  private spawnEnemy(archetypeId: EnemyArchetypeId): void {
    const spawn = this.pickSpawnPoint();
    const enemy = new Enemy(
      ENEMY_ARCHETYPES[archetypeId],
      spawn.position,
      () => this.rng(),
      spawn.exitPath,
    );
    this.enemies.push(enemy);
    this.scene.add(enemy.group);
    for (const mesh of enemy.hitMeshes) this.enemyMeshes.push(mesh);
  }

  /**
   * Picks the closest spawn the player cannot currently see and that is not
   * right on top of them. Closest-first matters: spawning the first wave at the
   * far perimeter leaves the opening minute with no pressure at all.
   */
  private pickSpawnPoint(): EnemySpawn {
    const points = this.compound.enemySpawns;
    const eye = this.player.eyePosition;

    const ranked = points
      .map((candidate) => ({
        candidate,
        distance: candidate.position.distanceTo(this.player.position),
      }))
      .filter((entry) => entry.distance >= 14)
      .sort((a, b) => a.distance - b.distance);

    if (ranked.length === 0) return points[this.spawnCursor++ % points.length];

    // Rotate through the hidden candidates so waves do not all arrive down one lane.
    for (let attempt = 0; attempt < ranked.length; attempt += 1) {
      const entry = ranked[(this.spawnCursor + attempt) % ranked.length];
      this.scratch.set(
        entry.candidate.position.x,
        entry.candidate.position.y + 1.5,
        entry.candidate.position.z,
      );
      const hidden = !this.combat.hasLineOfSight(eye, this.scratch, this.compound.raycastTargets);
      // Popping into view at 22 m+ reads as an enemy rounding a corner; closer
      // than that it looks like a cheat, so only hidden spawns qualify.
      if (hidden || entry.distance >= 22) {
        this.spawnCursor = (this.spawnCursor + attempt + 1) % ranked.length;
        return entry.candidate;
      }
    }

    this.spawnCursor = (this.spawnCursor + 1) % ranked.length;
    return ranked[0].candidate;
  }

  private removeEnemyMeshes(enemy: Enemy): void {
    for (const mesh of enemy.hitMeshes) {
      const index = this.enemyMeshes.indexOf(mesh);
      if (index >= 0) this.enemyMeshes.splice(index, 1);
    }
  }

  private hitstop(seconds: number): void {
    this.hitstopRemaining = Math.max(this.hitstopRemaining, seconds);
    this.timeScale = 0.06;
    this.audio.setDuck(0.55);
  }

  private resupply(): void {
    let gained = 0;
    for (const weapon of this.weapons) {
      gained += weapon.addReserve(Math.ceil(weapon.def.reserveMax * 0.5));
    }
    this.stats.grenades = Math.min(MAX_GRENADES, this.stats.grenades + 1);
    this.hud.updateGrenades(this.stats.grenades);
    this.audio.pickup();
    this.hud.showToast(gained > 0 ? `Resupplied +${gained}` : 'Grenade resupplied');
  }

  private selectWeapon(index: number): void {
    if (index === this.activeWeaponIndex) return;
    this.activeWeapon.cancelReload();
    this.activeWeaponIndex = index;
    this.viewModel.setWeapon(this.activeWeapon.def);
    this.hud.setActiveWeapon(index);
    this.hud.updateAmmo(this.activeWeapon);
    this.audio.reload('in');
  }

  private updateHud(): void {
    const weapon = this.activeWeapon;
    this.hud.updateVitals(this.player.health, this.player.healthFraction);
    this.hud.updateAmmo(weapon);
    this.hud.updateScore(this.stats.score, this.stats.streak);
    this.hud.updateWave(
      Math.max(1, this.waves.wave),
      this.waves.remaining + this.enemies.filter((enemy) => enemy.alive).length,
    );

    const spread = weapon.spread(this.adsFraction > 0.6, this.lastFrameState.moveFraction);
    const pixels = spreadToPixels(spread, this.camera.fov, this.canvas.clientHeight);
    this.hud.setCrosshairSpread(THREE.MathUtils.clamp(pixels, 3, 90));
    this.hud.setCrosshairHidden(this.adsFraction > 0.72 || this.lastFrameState.sprinting);
  }

  // ------------------------------------------------------------ run control

  private startRun(): void {
    this.resetRun();
    this.phase = 'playing';
    this.hud.hideOverlay();
    this.hud.setVisible(true);
    this.waves.beginNextWave();
    this.hud.showWaveBanner('Wave 1');
    this.audio.waveStart(1);
    void this.audio.unlock();
    if (!this.input.isTouchActive) this.input.requestPointerLock();
  }

  private resetRun(): void {
    for (const enemy of this.enemies) {
      this.scene.remove(enemy.group);
      enemy.dispose();
    }
    this.enemies.length = 0;
    this.enemyMeshes.length = 0;
    this.losCache.clear();

    for (const grenade of this.grenades) {
      this.scene.remove(grenade.group);
      grenade.dispose();
    }
    this.grenades.length = 0;

    for (const weapon of this.weapons) weapon.reset();
    for (const crate of this.ammoCrates) crate.reset();

    this.activeWeaponIndex = 0;
    this.viewModel.setWeapon(this.activeWeapon.def);
    this.stats = createStats();
    this.waves.reset();
    this.elapsed = 0;
    this.adsFraction = 0;
    this.timeScale = 1;
    this.hitstopRemaining = 0;
    this.spawnCursor = 0;
    this.vfx.clearTransient();

    this.player.reset(this.compound.playerSpawn, this.compound.playerSpawnYaw);
    this.cameraRig.snapTo(this.player, this.tuning.baseFov);

    this.hud.invalidate();
    this.hud.setActiveWeapon(this.activeWeaponIndex);
    this.hud.updateGrenades(this.stats.grenades);
    this.hud.updateVitals(this.player.health, this.player.healthFraction);
    this.hud.updateAmmo(this.activeWeapon);
    this.hud.updateScore(this.stats.score, this.stats.streak);
    this.hud.updateWave(1, 0);
  }

  private onPlayerDeath(killedBy: string): void {
    this.phase = 'dead';
    this.audio.setDuck(1);
    this.timeScale = 1;
    this.hitstopRemaining = 0;
    this.input.releasePointerLock();
    this.hud.setVisible(false);
    this.hud.showOverlay({
      kicker: 'Killed in action',
      title: `Wave ${Math.max(1, this.waves.wave)}`,
      body: `Taken down by a ${killedBy.toLowerCase()}. The compound is still hot.`,
      action: 'Redeploy',
      stats: this.buildStatCards(),
      showControls: false,
    });
  }

  private showMenu(): void {
    this.phase = 'menu';
    this.hud.setVisible(false);
    this.hud.showOverlay({
      kicker: 'Operation',
      title: 'Blackline Ops',
      body:
        'Hostiles are pushing the compound from the buildings and the north catwalk. ' +
        'Clear every wave, use cover, and watch your ammo — crates resupply on a timer.',
      action: 'Click to deploy',
    });
  }

  private pauseRun(): void {
    if (this.phase !== 'playing') return;
    this.phase = 'paused';
    // Pausing from the keyboard must also free the cursor, otherwise the
    // pause menu is on screen but nothing on it can be clicked.
    this.input.releasePointerLock();
    this.hud.setVisible(false);
    this.hud.showOverlay({
      kicker: 'Paused',
      title: `Wave ${Math.max(1, this.waves.wave)}`,
      body: 'The compound is holding. Resume when you are ready.',
      action: 'Resume',
      stats: this.buildStatCards(),
    });
  }

  private resumeRun(): void {
    this.phase = 'playing';
    this.hud.hideOverlay();
    this.hud.setVisible(true);
    if (!this.input.isTouchActive) this.input.requestPointerLock();
  }

  private onOverlayAction(): void {
    if (this.phase === 'paused') this.resumeRun();
    else this.startRun();
  }

  private buildStatCards(): Array<{ label: string; value: string }> {
    const accuracy =
      this.stats.shotsFired === 0
        ? 0
        : Math.round((this.stats.shotsHit / this.stats.shotsFired) * 100);
    return [
      { label: 'Score', value: this.stats.score.toLocaleString('en-US') },
      { label: 'Kills', value: String(this.stats.kills) },
      { label: 'Headshots', value: String(this.stats.headshots) },
      { label: 'Accuracy', value: `${accuracy}%` },
      { label: 'Best streak', value: String(this.stats.bestStreak) },
    ];
  }

  private readonly onCanvasMouseDown = () => {
    if (this.phase === 'playing' && !this.input.isPointerLocked && !this.input.isTouchActive) {
      this.input.requestPointerLock();
    }
  };

  private readonly onPointerLockChange = () => {
    // Escape releases the pointer, which is the player asking for a pause.
    if (document.pointerLockElement !== this.canvas && this.phase === 'playing') {
      this.pauseRun();
    }
  };

  private readonly onGlobalKeyDown = (event: KeyboardEvent) => {
    if (event.code === 'KeyM') {
      const muted = this.audio.toggleMute();
      this.hud.showToast(muted ? 'Audio muted' : 'Audio on');
      return;
    }
    if (event.code === 'Enter' && this.hud.isOverlayVisible) {
      event.preventDefault();
      this.onOverlayAction();
      return;
    }
    if (event.code === 'KeyP' && this.phase === 'playing') this.pauseRun();
  };

  // ---------------------------------------------------------------- scene

  private createScene(): void {
    const horizon = new THREE.Color('#b9a98c');
    this.scene.background = horizon;
    this.scene.fog = new THREE.Fog(horizon.getHex(), 42, 190);

    const sky = new THREE.Mesh(
      new THREE.SphereGeometry(230, 32, 20),
      new THREE.MeshBasicMaterial({
        map: createSkyTexture(),
        side: THREE.BackSide,
        fog: false,
        depthWrite: false,
      }),
    );
    this.scene.add(sky);

    const hemisphere = new THREE.HemisphereLight('#cfe0f2', '#6b6353', 1.25);
    this.scene.add(hemisphere);

    this.sun = new THREE.DirectionalLight('#ffe7bd', 3.1);
    this.sun.position.set(38, 46, 26);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    this.sun.shadow.camera.near = 1;
    this.sun.shadow.camera.far = 150;
    this.sun.shadow.camera.left = -55;
    this.sun.shadow.camera.right = 55;
    this.sun.shadow.camera.top = 55;
    this.sun.shadow.camera.bottom = -55;
    this.sun.shadow.bias = -0.0009;
    this.sun.shadow.normalBias = 0.022;
    this.scene.add(this.sun);
    this.scene.add(this.sun.target);

    // A cool bounce light keeps shadowed faces from going flat black.
    const bounce = new THREE.DirectionalLight('#8fb3d9', 0.65);
    bounce.position.set(-30, 18, -34);
    this.scene.add(bounce);

    this.compound = buildCompound(this.collision);
    this.scene.add(this.compound.group);
    this.scene.add(this.vfx.group);

    this.compound.ammoStations.forEach((position, index) => {
      const crate = new AmmoCrate(position, index * 1.7);
      this.ammoCrates.push(crate);
      this.scene.add(crate.group);
    });
  }

  private render(): void {
    // Three resets render stats on every render() call, so with a second pass
    // the diagnostics would only ever report the view model. Reset by hand and
    // let both passes accumulate into one honest frame total.
    this.renderer.info.autoReset = false;
    this.renderer.info.reset();
    this.renderer.render(this.scene, this.camera);

    // The view model draws in its own space after a depth clear so it can never
    // intersect world geometry when the player backs into a wall.
    if (this.phase === 'playing' || this.phase === 'paused') {
      this.renderer.autoClear = false;
      this.renderer.clearDepth();
      this.renderer.render(this.viewModel.scene, this.viewModel.camera);
      this.renderer.autoClear = true;
    }
  }

  // ------------------------------------------------------------ test hooks

  private installTestHooks(): void {
    window.__THREE_GAME_TEST_HOOKS__ = {
      seed: (value: number) => {
        this.rng = createSeededRandom(value);
        this.vfx.setRandom(() => this.rng());
        this.audio.setRandom(() => this.rng());
      },
      setState: (name: string) => {
        switch (name) {
          case 'menu':
            this.resetRun();
            this.showMenu();
            break;
          case 'active-play':
            this.enterScriptedCombat(1, 11);
            break;
          case 'heavy-combat':
            this.enterScriptedCombat(6, 17);
            break;
          case 'ads-aiming':
            // Hold the aim through the real input path so the view model,
            // FOV and crosshair all settle the way they do in play.
            this.scripted.active = true;
            this.scripted.ads = true;
            this.enterScriptedCombat(2, 5);
            this.scripted.active = false;
            this.scripted.ads = false;
            this.input.setScriptedMovement(null);
            break;
          case 'death':
            this.enterScriptedCombat(3, 0.6);
            this.player.health = 1;
            this.damagePlayer(50, new THREE.Vector3(0, 1, -10), 'Rifleman');
            break;
          default:
            throw new Error(`Unknown test state: ${name}`);
        }
        this.render();
        this.publishDiagnostics();
        return { state: name };
      },
      setPausedForScreenshot: (paused: boolean) => {
        this.pausedForScreenshot = paused;
      },
      setReducedMotion: (enabled: boolean) => {
        this.reducedMotion = enabled;
        this.render();
        this.publishDiagnostics();
      },
      hideDebugUi: (hidden: boolean) => {
        this.debugTools.setHidden(hidden);
      },
      botInput: (command) => {
        this.scripted.active = true;
        if (command.move) this.scripted.move.set(command.move.x, command.move.y);
        if (command.look) this.scripted.look.set(command.look.x, command.look.y);
        if (command.fire !== undefined) this.scripted.fire = command.fire;
        if (command.ads !== undefined) this.scripted.ads = command.ads;
        if (command.reload) this.scripted.reload = true;
      },
    };
  }

  /**
   * Deterministically drops the player into a live firefight: spawn the wave,
   * then advance the simulation in fixed steps so enemies are mid-approach.
   */
  private enterScriptedCombat(wave: number, simulateSeconds: number): void {
    this.resetRun();
    this.phase = 'playing';
    this.hud.hideOverlay();
    this.hud.setVisible(true);

    for (let i = 1; i < wave; i += 1) this.waves.beginNextWave();
    this.waves.beginNextWave();

    const step = 1 / 60;
    this.captureMode = true;
    for (let time = 0; time < simulateSeconds; time += step) {
      this.simulate(step);
    }
    this.captureMode = false;
  }

  private publishDiagnostics(): void {
    const info = this.renderer.info;
    const aliveEnemies = this.enemies.filter((enemy) => enemy.alive).length;
    const weapon = this.activeWeapon;

    // The nearest live target, so a scripted playtest can aim the same way a
    // player does instead of being handed a magic "kill" hook.
    let nearest: ThreeGameDiagnostics['nearestEnemy'] = null;
    let nearestDistance = Infinity;
    for (const enemy of this.enemies) {
      if (!enemy.alive) continue;
      const distance = enemy.group.position.distanceTo(this.player.position);
      if (distance >= nearestDistance) continue;
      nearestDistance = distance;
      const eye = enemy.eyePosition;
      nearest = {
        x: eye.x,
        y: eye.y,
        z: eye.z,
        distance,
        archetype: enemy.archetype.id,
        visible: enemy.lastVisible,
      };
    }

    window.__THREE_GAME_DIAGNOSTICS__ = {
      frame: this.frame,
      elapsed: this.elapsed,
      score: this.stats.score,
      targetScore: 0,
      complete: false,
      phase: this.phase,
      wave: this.waves.wave,
      enemiesAlive: aliveEnemies,
      enemiesRemaining: this.waves.remaining + aliveEnemies,
      kills: this.stats.kills,
      headshots: this.stats.headshots,
      accuracy: this.stats.shotsFired === 0 ? 0 : this.stats.shotsHit / this.stats.shotsFired,
      weapon: weapon.def.id,
      ammo: { mag: weapon.ammo, reserve: weapon.reserve },
      nearestEnemy: nearest,
      enemyStates: this.enemies
        .filter((enemy) => enemy.alive)
        .map((enemy) => ({
          state: enemy.state,
          visible: enemy.lastVisible,
          distance: Number(enemy.group.position.distanceTo(this.player.position).toFixed(1)),
        })),
      player: {
        position: {
          x: this.player.position.x,
          y: this.player.position.y,
          z: this.player.position.z,
        },
        speed: Math.hypot(this.player.velocity.x, this.player.velocity.z),
        health: this.player.health,
        alive: this.player.alive,
        yaw: this.player.yaw,
        pitch: this.player.pitch,
      },
      renderer: {
        calls: info.render.calls,
        triangles: info.render.triangles,
        geometries: info.memory.geometries,
        textures: info.memory.textures,
        programs: info.programs?.length ?? 0,
      },
      canvas: {
        clientWidth: this.canvas.clientWidth,
        clientHeight: this.canvas.clientHeight,
        width: this.canvas.width,
        height: this.canvas.height,
        dpr: Math.min(window.devicePixelRatio || 1, this.tuning.maxDpr),
      },
    };
  }

  private element(selector: string): HTMLElement {
    const found = document.querySelector<HTMLElement>(selector);
    if (!found) throw new Error(`Missing element: ${selector}`);
    return found;
  }
}

function createStats(): RunStats {
  return {
    score: 0,
    kills: 0,
    headshots: 0,
    shotsFired: 0,
    shotsHit: 0,
    streak: 0,
    bestStreak: 0,
    grenades: MAX_GRENADES,
  };
}

/** Converts an aim-cone half-angle into on-screen crosshair pixels. */
function spreadToPixels(spread: number, fovDegrees: number, viewportHeight: number): number {
  const halfFov = THREE.MathUtils.degToRad(fovDegrees) / 2;
  return (Math.tan(spread) / Math.tan(halfFov)) * (viewportHeight / 2);
}
