import type { WeaponDef } from './WeaponDefs';

/** Runtime state for one carried weapon: ammo, cadence, reload and bloom. */
export class Weapon {
  ammo: number;
  reserve: number;
  private cooldown = 0;
  private reloadRemaining = 0;
  private bloom = 0;
  /** Semi/pump weapons require the trigger to be released between shots. */
  private triggerLatched = false;

  constructor(readonly def: WeaponDef) {
    this.ammo = def.magSize;
    this.reserve = def.reserveMax;
  }

  get isReloading(): boolean {
    return this.reloadRemaining > 0;
  }

  get reloadProgress(): number {
    if (this.reloadRemaining <= 0) return 1;
    return 1 - this.reloadRemaining / this.def.reloadSeconds;
  }

  get isEmpty(): boolean {
    return this.ammo <= 0;
  }

  get canRefill(): boolean {
    return this.reserve > 0 && this.ammo < this.def.magSize;
  }

  reset(): void {
    this.ammo = this.def.magSize;
    this.reserve = this.def.reserveMax;
    this.cooldown = 0;
    this.reloadRemaining = 0;
    this.bloom = 0;
    this.triggerLatched = false;
  }

  update(delta: number): boolean {
    if (this.cooldown > 0) this.cooldown = Math.max(0, this.cooldown - delta);
    if (this.bloom > 0) {
      this.bloom = Math.max(0, this.bloom - this.def.bloomRecovery * delta);
    }
    if (this.reloadRemaining > 0) {
      this.reloadRemaining -= delta;
      if (this.reloadRemaining <= 0) {
        this.reloadRemaining = 0;
        this.finishReload();
        return true;
      }
    }
    return false;
  }

  releaseTrigger(): void {
    this.triggerLatched = false;
  }

  /** Consumes one round when the weapon is ready; returns false otherwise. */
  tryFire(): boolean {
    if (this.reloadRemaining > 0 || this.cooldown > 0 || this.ammo <= 0) return false;
    if (this.def.mode !== 'auto') {
      if (this.triggerLatched) return false;
      this.triggerLatched = true;
    }
    this.ammo -= 1;
    this.cooldown = 60 / this.def.rpm;
    this.bloom = Math.min(this.def.bloomMax, this.bloom + this.def.bloomPerShot);
    return true;
  }

  startReload(): boolean {
    if (this.reloadRemaining > 0 || !this.canRefill) return false;
    this.reloadRemaining = this.def.reloadSeconds;
    return true;
  }

  cancelReload(): void {
    this.reloadRemaining = 0;
  }

  /** Total cone half-angle for the next shot. `movement` is 0..1 of top speed. */
  spread(ads: boolean, movement: number): number {
    const base = ads ? this.def.spreadAds : this.def.spreadHip;
    return base + this.def.spreadMoving * movement + this.bloom;
  }

  addReserve(rounds: number): number {
    const before = this.reserve;
    this.reserve = Math.min(this.def.reserveMax, this.reserve + rounds);
    return this.reserve - before;
  }

  private finishReload(): void {
    const needed = this.def.magSize - this.ammo;
    const loaded = Math.min(needed, this.reserve);
    this.ammo += loaded;
    this.reserve -= loaded;
  }
}
