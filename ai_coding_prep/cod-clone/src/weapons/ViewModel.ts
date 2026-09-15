import * as THREE from 'three';
import { disposeObject3D } from '../utils/dispose';
import type { WeaponDef } from './WeaponDefs';

export interface ViewModelState {
  readonly adsFraction: number;
  readonly moveFraction: number;
  readonly bobPhase: number;
  readonly sprinting: boolean;
  readonly reloading: boolean;
  readonly reloadProgress: number;
  readonly lookDelta: THREE.Vector2;
  readonly grounded: boolean;
}

const SWAY_LIMIT = 0.05;
/**
 * Weapons are modelled at real-world scale for readability, then shrunk for the
 * view model. Without this a 0.9 m rifle held 0.4 m from the eye fills the frame.
 */
const VIEW_MODEL_SCALE = 0.5;

/**
 * The first-person weapon lives in its own scene and camera, drawn after a
 * depth clear. That is what stops the gun from intersecting world geometry
 * when the player backs into a wall.
 */
export class ViewModel {
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(55, 1, 0.01, 6);

  private readonly root = new THREE.Group();
  private readonly weaponGroup = new THREE.Group();
  private readonly muzzleFlash: THREE.Group;
  private readonly basePosition = new THREE.Vector3();
  private readonly sway = new THREE.Vector2();
  private def: WeaponDef | null = null;
  private weaponMesh: THREE.Group | null = null;
  private recoilOffset = 0;
  private recoilPitch = 0;
  private flashTimer = 0;
  private swapDip = 0;

  constructor() {
    this.scene.add(this.root);
    this.root.add(this.weaponGroup);
    this.weaponGroup.scale.setScalar(VIEW_MODEL_SCALE);

    const key = new THREE.DirectionalLight('#fff2da', 3.6);
    key.position.set(-0.7, 1.2, 1.1);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight('#9dc0e8', 1.9);
    this.scene.add(fill);
    fill.position.set(1.1, -0.2, 0.7);
    const rim = new THREE.DirectionalLight('#ffd9a8', 1.5);
    rim.position.set(0.4, 0.5, -1.2);
    this.scene.add(rim);
    this.scene.add(new THREE.AmbientLight('#8996a8', 2.4));

    this.muzzleFlash = this.createMuzzleFlash();
    this.muzzleFlash.visible = false;
    this.weaponGroup.add(this.muzzleFlash);
  }

  setWeapon(def: WeaponDef): void {
    if (this.def?.id === def.id) return;
    if (this.weaponMesh) {
      this.weaponGroup.remove(this.weaponMesh);
      disposeObject3D(this.weaponMesh);
    }
    this.def = def;
    this.weaponMesh = this.buildWeapon(def);
    this.weaponGroup.add(this.weaponMesh);
    this.muzzleFlash.position.set(0, 0, -(def.viewModel.bodyLength / 2 + def.viewModel.barrelLength));
    this.swapDip = 1;
  }

  setAspect(aspect: number): void {
    if (this.camera.aspect === aspect) return;
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();
  }

  fire(recoilKick: number): void {
    this.recoilOffset = Math.min(0.13, this.recoilOffset + recoilKick * 2.6);
    this.recoilPitch = Math.min(0.3, this.recoilPitch + recoilKick * 5.5);
    this.flashTimer = 0.045;
    this.muzzleFlash.rotation.z = this.flashTimer * 140;
  }

  update(delta: number, state: ViewModelState): void {
    if (!this.def) return;
    const spec = this.def.viewModel;

    this.basePosition.set(
      THREE.MathUtils.lerp(spec.hipPosition[0], spec.adsPosition[0], state.adsFraction),
      THREE.MathUtils.lerp(spec.hipPosition[1], spec.adsPosition[1], state.adsFraction),
      THREE.MathUtils.lerp(spec.hipPosition[2], spec.adsPosition[2], state.adsFraction),
    );

    // Sway trails the look input, then eases back to centre.
    const swayScale = (1 - state.adsFraction) * 0.9 + 0.1;
    this.sway.x = THREE.MathUtils.clamp(
      THREE.MathUtils.damp(this.sway.x - state.lookDelta.x * 0.0016 * swayScale, 0, 6, delta),
      -SWAY_LIMIT,
      SWAY_LIMIT,
    );
    this.sway.y = THREE.MathUtils.clamp(
      THREE.MathUtils.damp(this.sway.y - state.lookDelta.y * 0.0016 * swayScale, 0, 6, delta),
      -SWAY_LIMIT,
      SWAY_LIMIT,
    );

    const bobScale = (1 - state.adsFraction * 0.82) * state.moveFraction * (state.grounded ? 1 : 0.2);
    const bobX = Math.sin(state.bobPhase * 2.1) * 0.017 * bobScale;
    const bobY = (Math.cos(state.bobPhase * 4.2) - 1) * 0.009 * bobScale;

    this.recoilOffset = THREE.MathUtils.damp(this.recoilOffset, 0, 11, delta);
    this.recoilPitch = THREE.MathUtils.damp(this.recoilPitch, 0, 11, delta);
    this.swapDip = THREE.MathUtils.damp(this.swapDip, 0, 8, delta);

    let reloadDip = 0;
    let reloadRoll = 0;
    if (state.reloading) {
      // Down on the way out, back up on the way in.
      const arc = Math.sin(Math.PI * Math.min(1, state.reloadProgress));
      reloadDip = arc * 0.14;
      reloadRoll = arc * 0.5;
    }

    const sprintBlend = state.sprinting ? 1 : 0;
    const sprintLerp = THREE.MathUtils.damp(
      this.weaponGroup.userData.sprintBlend ?? 0,
      sprintBlend,
      9,
      delta,
    );
    this.weaponGroup.userData.sprintBlend = sprintLerp;

    this.root.position.set(
      this.basePosition.x + this.sway.x + bobX + sprintLerp * 0.06,
      this.basePosition.y + this.sway.y + bobY - reloadDip - this.swapDip * 0.22 - sprintLerp * 0.07,
      this.basePosition.z + this.recoilOffset + sprintLerp * 0.05,
    );
    this.root.rotation.set(
      this.recoilPitch + reloadRoll * 0.6 + this.swapDip * 0.5,
      -this.sway.x * 3.2 + sprintLerp * 0.62,
      reloadRoll * 0.35 + this.sway.x * 1.4 + sprintLerp * 0.3,
    );

    if (this.flashTimer > 0) {
      this.flashTimer -= delta;
      this.muzzleFlash.visible = this.flashTimer > 0;
      const scale = 0.7 + Math.max(0, this.flashTimer) * 9;
      this.muzzleFlash.scale.setScalar(scale);
    }
  }

  /** Muzzle position in the main scene, derived from the camera basis. */
  muzzleWorldPosition(
    target: THREE.Vector3,
    eye: THREE.Vector3,
    forward: THREE.Vector3,
    right: THREE.Vector3,
    up: THREE.Vector3,
    adsFraction: number,
  ): THREE.Vector3 {
    const lateral = THREE.MathUtils.lerp(0.19, 0.0, adsFraction);
    const drop = THREE.MathUtils.lerp(0.15, 0.07, adsFraction);
    return target
      .copy(eye)
      .addScaledVector(forward, 0.62)
      .addScaledVector(right, lateral)
      .addScaledVector(up, -drop);
  }

  dispose(): void {
    disposeObject3D(this.scene);
  }

  private buildWeapon(def: WeaponDef): THREE.Group {
    const spec = def.viewModel;
    const group = new THREE.Group();

    const bodyMaterial = new THREE.MeshStandardMaterial({
      color: spec.bodyColor,
      roughness: 0.42,
      metalness: 0.55,
    });
    const accentMaterial = new THREE.MeshStandardMaterial({
      color: spec.accentColor,
      roughness: 0.62,
      metalness: 0.2,
    });
    const darkMaterial = new THREE.MeshStandardMaterial({
      color: '#2a2f34',
      roughness: 0.38,
      metalness: 0.7,
    });

    const receiver = new THREE.Mesh(
      new THREE.BoxGeometry(spec.bodyWidth, spec.bodyHeight, spec.bodyLength),
      bodyMaterial,
    );
    group.add(receiver);

    const barrel = new THREE.Mesh(
      new THREE.CylinderGeometry(spec.barrelRadius, spec.barrelRadius * 0.94, spec.barrelLength, 12),
      darkMaterial,
    );
    barrel.rotation.x = Math.PI / 2;
    barrel.position.set(0, spec.bodyHeight * 0.06, -(spec.bodyLength / 2 + spec.barrelLength / 2));
    group.add(barrel);

    const handguard = new THREE.Mesh(
      new THREE.BoxGeometry(spec.bodyWidth * 0.82, spec.bodyHeight * 0.68, spec.barrelLength * 0.68),
      accentMaterial,
    );
    handguard.position.set(0, 0, -(spec.bodyLength / 2 + spec.barrelLength * 0.3));
    group.add(handguard);

    const grip = new THREE.Mesh(
      new THREE.BoxGeometry(spec.bodyWidth * 0.86, 0.13, 0.06),
      darkMaterial,
    );
    grip.position.set(0, -0.092, spec.bodyLength * 0.24);
    grip.rotation.x = -0.28;
    group.add(grip);

    if (spec.magazineDepth > 0) {
      const magazine = new THREE.Mesh(
        new THREE.BoxGeometry(spec.bodyWidth * 0.72, spec.magazineDepth, 0.055),
        darkMaterial,
      );
      magazine.position.set(0, -spec.magazineDepth / 2 - 0.03, -spec.bodyLength * 0.06);
      magazine.rotation.x = 0.12;
      group.add(magazine);
    }

    if (spec.foregrip) {
      const foregrip = new THREE.Mesh(
        new THREE.BoxGeometry(spec.bodyWidth * 0.6, 0.085, 0.045),
        darkMaterial,
      );
      foregrip.position.set(0, -0.075, -(spec.bodyLength / 2 + spec.barrelLength * 0.42));
      group.add(foregrip);
    }

    if (spec.stock) {
      const stock = new THREE.Mesh(
        new THREE.BoxGeometry(spec.bodyWidth * 0.8, spec.bodyHeight * 0.92, 0.17),
        accentMaterial,
      );
      stock.position.set(0, -0.012, spec.bodyLength / 2 + 0.082);
      group.add(stock);
    }

    // Iron sights: a rear notch and a front post the ADS pose lines up on.
    const rearSight = new THREE.Mesh(
      new THREE.BoxGeometry(spec.bodyWidth * 0.5, 0.026, 0.02),
      darkMaterial,
    );
    rearSight.position.set(0, spec.bodyHeight / 2 + 0.013, spec.bodyLength * 0.28);
    group.add(rearSight);

    const frontPost = new THREE.Mesh(
      new THREE.BoxGeometry(0.008, 0.032, 0.014),
      darkMaterial,
    );
    frontPost.position.set(0, spec.bodyHeight / 2 + 0.016, -(spec.bodyLength / 2 + spec.barrelLength * 0.82));
    group.add(frontPost);

    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(0.0055, 8, 8),
      new THREE.MeshBasicMaterial({ color: '#7dff9f' }),
    );
    dot.position.set(0, spec.bodyHeight / 2 + 0.026, spec.bodyLength * 0.28);
    group.add(dot);

    return group;
  }

  private createMuzzleFlash(): THREE.Group {
    const group = new THREE.Group();
    const core = new THREE.Mesh(
      new THREE.SphereGeometry(0.035, 10, 8),
      new THREE.MeshBasicMaterial({ color: '#fff3c4', transparent: true, opacity: 0.95 }),
    );
    core.scale.z = 1.7;
    group.add(core);

    const petals = new THREE.Mesh(
      new THREE.ConeGeometry(0.052, 0.13, 6, 1, true),
      new THREE.MeshBasicMaterial({
        color: '#ffb340',
        transparent: true,
        opacity: 0.68,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    );
    petals.rotation.x = -Math.PI / 2;
    petals.position.z = -0.05;
    group.add(petals);

    return group;
  }
}
