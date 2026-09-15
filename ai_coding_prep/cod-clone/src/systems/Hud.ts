import type { Weapon } from '../weapons/Weapon';

export type HitMarkerKind = 'body' | 'head' | 'kill';

export interface OverlayStat {
  readonly label: string;
  readonly value: string;
}

export interface OverlayConfig {
  readonly kicker: string;
  readonly title: string;
  readonly body: string;
  readonly action: string;
  readonly stats?: readonly OverlayStat[];
  readonly showControls?: boolean;
}

const DAMAGE_ARROW_COUNT = 6;
const KILLFEED_LIMIT = 5;

export class Hud {
  private readonly root = this.element('#hud');
  private readonly crosshair = this.element('#crosshair');
  private readonly hitmarker = this.element('#hitmarker');
  private readonly damageIndicators = this.element('#damage-indicators');
  private readonly damageVignette = this.element('#damage-vignette');
  private readonly lowHealthPulse = this.element('#low-health-pulse');
  private readonly scoreValue = this.element('#score-value');
  private readonly streakChip = this.element('#streak-chip');
  private readonly streakValue = this.element('#streak-value');
  private readonly waveValue = this.element('#wave-value');
  private readonly enemyRemaining = this.element('#enemy-remaining');
  private readonly killfeed = this.element('#killfeed');
  private readonly healthValue = this.element('#health-value');
  private readonly healthFill = this.element('#health-fill');
  private readonly grenadeCount = this.element('#grenade-count');
  private readonly weaponName = this.element('#weapon-name');
  private readonly ammoMag = this.element('#ammo-mag');
  private readonly ammoReserve = this.element('#ammo-reserve');
  private readonly reloadBar = this.element('#reload-bar');
  private readonly reloadFill = this.element('#reload-fill');
  private readonly weaponSlots = this.element('#weapon-slots');
  private readonly waveBanner = this.element('#wave-banner');
  private readonly waveBannerText = this.element('#wave-banner-text');
  private readonly toast = this.element('#hud-toast');
  private readonly overlay = this.element('#overlay');
  private readonly overlayKicker = this.element('#overlay-kicker');
  private readonly overlayTitle = this.element('#overlay-title');
  private readonly overlayBody = this.element('#overlay-body');
  private readonly overlayStats = this.element('#overlay-stats');
  private readonly overlayAction = this.element('#overlay-action') as HTMLButtonElement;
  private readonly overlayControls = this.element('#overlay-controls');

  private readonly damageArrows: HTMLElement[] = [];
  private damageArrowCursor = 0;
  private lastScore = -1;
  private lastAmmo = -1;
  private lastHealthBucket = -1;

  constructor() {
    for (let i = 0; i < DAMAGE_ARROW_COUNT; i += 1) {
      const arrow = document.createElement('div');
      arrow.className = 'damage-arrow';
      this.damageIndicators.appendChild(arrow);
      this.damageArrows.push(arrow);
    }
  }

  setVisible(visible: boolean): void {
    this.root.dataset.visible = visible ? 'true' : 'false';
  }

  onAction(handler: () => void): void {
    this.overlayAction.addEventListener('click', handler);
  }

  updateVitals(health: number, fraction: number): void {
    const rounded = Math.ceil(health);
    const bucket = Math.ceil(fraction * 20);
    if (bucket !== this.lastHealthBucket) {
      this.lastHealthBucket = bucket;
      this.healthValue.textContent = String(rounded);
      this.healthFill.style.width = `${Math.max(0, fraction) * 100}%`;
      this.healthFill.style.backgroundColor =
        fraction > 0.6 ? 'var(--hud-good)' : fraction > 0.3 ? 'var(--hud-accent)' : 'var(--hud-danger)';
      this.lowHealthPulse.style.opacity = fraction <= 0.3 && fraction > 0 ? '' : '0';
      this.lowHealthPulse.style.display = fraction <= 0.3 && fraction > 0 ? 'block' : 'none';
    }
    // The vignette tracks health continuously so recovery reads as it happens.
    this.damageVignette.style.opacity = String(Math.max(0, 1 - fraction) * 0.9);
  }

  updateAmmo(weapon: Weapon): void {
    if (weapon.ammo !== this.lastAmmo) {
      this.lastAmmo = weapon.ammo;
      this.ammoMag.textContent = String(weapon.ammo);
      this.ammoMag.dataset.low = weapon.ammo <= Math.ceil(weapon.def.magSize * 0.25) ? 'true' : 'false';
    }
    this.ammoReserve.textContent = `/ ${weapon.reserve}`;
    this.weaponName.textContent = weapon.def.name;

    const reloading = weapon.isReloading;
    this.reloadBar.dataset.active = reloading ? 'true' : 'false';
    if (reloading) this.reloadFill.style.width = `${weapon.reloadProgress * 100}%`;
  }

  buildWeaponSlots(weapons: readonly Weapon[]): void {
    this.weaponSlots.replaceChildren();
    weapons.forEach((weapon, index) => {
      const item = document.createElement('li');
      const key = document.createElement('span');
      key.className = 'slot-key';
      key.textContent = String(index + 1);
      const tag = document.createElement('span');
      tag.textContent = weapon.def.tag;
      item.append(key, tag);
      this.weaponSlots.appendChild(item);
    });
  }

  setActiveWeapon(index: number): void {
    Array.from(this.weaponSlots.children).forEach((child, childIndex) => {
      (child as HTMLElement).dataset.active = childIndex === index ? 'true' : 'false';
    });
    this.lastAmmo = -1;
  }

  updateScore(score: number, streak: number): void {
    if (score !== this.lastScore) {
      this.lastScore = score;
      this.scoreValue.textContent = score.toLocaleString('en-US');
      this.scoreValue.animate(
        [{ transform: 'scale(1.16)' }, { transform: 'scale(1)' }],
        { duration: 130, easing: 'ease-out' },
      );
    }
    const multiplier = streakMultiplier(streak);
    this.streakChip.dataset.active = multiplier > 1 ? 'true' : 'false';
    this.streakValue.textContent = `x${multiplier}`;
  }

  updateWave(wave: number, remaining: number): void {
    this.waveValue.textContent = String(wave);
    this.enemyRemaining.textContent = String(remaining);
  }

  updateGrenades(count: number): void {
    this.grenadeCount.textContent = String(count);
  }

  setCrosshairSpread(pixels: number): void {
    this.crosshair.style.setProperty('--spread', `${pixels.toFixed(1)}px`);
  }

  setCrosshairHidden(hidden: boolean): void {
    this.crosshair.dataset.hidden = hidden ? 'true' : 'false';
  }

  showHitMarker(kind: HitMarkerKind): void {
    this.hitmarker.dataset.kind = kind;
    this.hitmarker.animate(
      [
        { opacity: 1, transform: 'scale(0.78)' },
        { opacity: 1, transform: 'scale(1)' },
        { opacity: 0, transform: 'scale(1.1)' },
      ],
      { duration: kind === 'kill' ? 320 : 220, easing: 'ease-out' },
    );
  }

  addKill(target: string, tag?: string): void {
    const item = document.createElement('li');
    if (tag) {
      const tagElement = document.createElement('span');
      tagElement.className = 'feed-tag';
      tagElement.textContent = tag;
      item.appendChild(tagElement);
    }
    const label = document.createElement('span');
    label.textContent = target;
    item.appendChild(label);
    this.killfeed.prepend(item);

    while (this.killfeed.childElementCount > KILLFEED_LIMIT) {
      this.killfeed.lastElementChild?.remove();
    }
    window.setTimeout(() => item.remove(), 4200);
  }

  /** `angle` is the bearing of the shooter relative to where the player looks. */
  showDamageFrom(angle: number): void {
    const arrow = this.damageArrows[this.damageArrowCursor];
    this.damageArrowCursor = (this.damageArrowCursor + 1) % DAMAGE_ARROW_COUNT;
    arrow.style.transform = `rotate(${angle}rad)`;
    arrow.animate([{ opacity: 0.95 }, { opacity: 0.95 }, { opacity: 0 }], {
      duration: 900,
      easing: 'ease-out',
    });
  }

  showWaveBanner(text: string): void {
    this.waveBannerText.textContent = text;
    this.waveBanner.animate(
      [
        { opacity: 0, transform: 'translateX(-50%) translateY(10px)' },
        { opacity: 1, transform: 'translateX(-50%) translateY(0)' },
        { opacity: 1, transform: 'translateX(-50%) translateY(0)' },
        { opacity: 0, transform: 'translateX(-50%) translateY(-8px)' },
      ],
      { duration: 2100, easing: 'ease-out' },
    );
  }

  showToast(text: string): void {
    this.toast.textContent = text;
    this.toast.animate([{ opacity: 0 }, { opacity: 1 }, { opacity: 1 }, { opacity: 0 }], {
      duration: 1600,
      easing: 'ease-out',
    });
  }

  showOverlay(config: OverlayConfig): void {
    this.overlayKicker.textContent = config.kicker;
    this.overlayTitle.textContent = config.title;
    this.overlayBody.textContent = config.body;
    this.overlayAction.textContent = config.action;

    this.overlayStats.replaceChildren();
    for (const stat of config.stats ?? []) {
      const wrapper = document.createElement('div');
      wrapper.className = 'stat';
      const term = document.createElement('dt');
      term.textContent = stat.label;
      const value = document.createElement('dd');
      value.textContent = stat.value;
      wrapper.append(term, value);
      this.overlayStats.appendChild(wrapper);
    }

    this.overlayControls.style.display = config.showControls === false ? 'none' : '';
    this.overlay.dataset.hidden = 'false';
    this.overlayAction.focus({ preventScroll: true });
  }

  hideOverlay(): void {
    this.overlay.dataset.hidden = 'true';
  }

  get isOverlayVisible(): boolean {
    return this.overlay.dataset.hidden !== 'true';
  }

  /** Forces the next vitals/ammo write to repaint after a run reset. */
  invalidate(): void {
    this.lastScore = -1;
    this.lastAmmo = -1;
    this.lastHealthBucket = -1;
    this.killfeed.replaceChildren();
  }

  private element(selector: string): HTMLElement {
    const found = document.querySelector<HTMLElement>(selector);
    if (!found) throw new Error(`Missing HUD element: ${selector}`);
    return found;
  }
}

export function streakMultiplier(streak: number): number {
  if (streak >= 15) return 5;
  if (streak >= 10) return 4;
  if (streak >= 6) return 3;
  if (streak >= 3) return 2;
  return 1;
}
