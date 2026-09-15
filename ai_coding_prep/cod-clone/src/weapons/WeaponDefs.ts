export type FireMode = 'auto' | 'semi' | 'pump';

export interface WeaponDef {
  readonly id: string;
  readonly name: string;
  /** Short label for the HUD ammo block. */
  readonly tag: string;
  readonly mode: FireMode;
  readonly damage: number;
  readonly headshotMultiplier: number;
  readonly pellets: number;
  readonly rpm: number;
  readonly magSize: number;
  readonly reserveMax: number;
  readonly reloadSeconds: number;
  /** Cone half-angle in radians before per-shot recoil bloom. */
  readonly spreadHip: number;
  readonly spreadAds: number;
  /** Extra cone added at full sprint-speed movement. */
  readonly spreadMoving: number;
  /** Cone added per shot fired, decaying at `bloomRecovery` per second. */
  readonly bloomPerShot: number;
  readonly bloomMax: number;
  readonly bloomRecovery: number;
  /** Camera kick in radians per shot. */
  readonly recoilPitch: number;
  readonly recoilYaw: number;
  /** Fraction of accumulated kick pulled back per second once firing stops. */
  readonly recoilRecovery: number;
  readonly range: number;
  /** Damage is full to `falloffStart`, scaling to `falloffFloor` at `falloffEnd`. */
  readonly falloffStart: number;
  readonly falloffEnd: number;
  readonly falloffFloor: number;
  readonly adsFov: number;
  readonly adsSeconds: number;
  readonly viewModel: ViewModelSpec;
  readonly audio: WeaponAudioSpec;
}

export interface ViewModelSpec {
  readonly bodyLength: number;
  readonly bodyHeight: number;
  readonly bodyWidth: number;
  readonly barrelLength: number;
  readonly barrelRadius: number;
  readonly stock: boolean;
  readonly magazineDepth: number;
  readonly foregrip: boolean;
  readonly bodyColor: string;
  readonly accentColor: string;
  /** Hip-fire rest position of the view model, in camera space. */
  readonly hipPosition: readonly [number, number, number];
  readonly adsPosition: readonly [number, number, number];
}

export interface WeaponAudioSpec {
  /** Base oscillator frequency of the shot transient, in Hz. */
  readonly bodyHz: number;
  readonly decaySeconds: number;
  readonly noiseGain: number;
  readonly gain: number;
}

export const WEAPON_DEFS: readonly WeaponDef[] = [
  {
    id: 'mk4',
    name: 'MK-4 Carbine',
    tag: 'AR',
    mode: 'auto',
    damage: 34,
    headshotMultiplier: 2.6,
    pellets: 1,
    rpm: 720,
    magSize: 30,
    reserveMax: 210,
    reloadSeconds: 1.85,
    spreadHip: 0.036,
    spreadAds: 0.004,
    spreadMoving: 0.03,
    bloomPerShot: 0.0045,
    bloomMax: 0.05,
    bloomRecovery: 0.075,
    recoilPitch: 0.0135,
    recoilYaw: 0.0052,
    recoilRecovery: 7.5,
    range: 140,
    falloffStart: 34,
    falloffEnd: 78,
    falloffFloor: 0.55,
    adsFov: 52,
    adsSeconds: 0.17,
    viewModel: {
      bodyLength: 0.52,
      bodyHeight: 0.1,
      bodyWidth: 0.072,
      barrelLength: 0.34,
      barrelRadius: 0.019,
      stock: true,
      magazineDepth: 0.17,
      foregrip: true,
      bodyColor: '#2f3338',
      accentColor: '#8c7a4e',
      hipPosition: [0.16, -0.16, -0.62],
      adsPosition: [0, -0.0315, -0.92],
    },
    audio: { bodyHz: 165, decaySeconds: 0.2, noiseGain: 0.55, gain: 0.6 },
  },
  {
    id: 'vx9',
    name: 'VX-9 Vector',
    tag: 'SMG',
    mode: 'auto',
    damage: 22,
    headshotMultiplier: 2.2,
    pellets: 1,
    rpm: 960,
    magSize: 36,
    reserveMax: 252,
    reloadSeconds: 1.55,
    spreadHip: 0.028,
    spreadAds: 0.009,
    spreadMoving: 0.018,
    bloomPerShot: 0.0038,
    bloomMax: 0.058,
    bloomRecovery: 0.09,
    recoilPitch: 0.0092,
    recoilYaw: 0.0061,
    recoilRecovery: 9,
    range: 90,
    falloffStart: 18,
    falloffEnd: 44,
    falloffFloor: 0.42,
    adsFov: 58,
    adsSeconds: 0.12,
    viewModel: {
      bodyLength: 0.36,
      bodyHeight: 0.105,
      bodyWidth: 0.066,
      barrelLength: 0.16,
      barrelRadius: 0.016,
      stock: false,
      magazineDepth: 0.21,
      foregrip: true,
      bodyColor: '#33373d',
      accentColor: '#5d6b7a',
      hipPosition: [0.15, -0.15, -0.55],
      adsPosition: [0, -0.0328, -0.85],
    },
    audio: { bodyHz: 210, decaySeconds: 0.14, noiseGain: 0.48, gain: 0.5 },
  },
  {
    id: 'breacher',
    name: 'Breacher 12',
    tag: 'SG',
    mode: 'pump',
    damage: 15,
    headshotMultiplier: 1.6,
    pellets: 9,
    rpm: 78,
    magSize: 6,
    reserveMax: 48,
    reloadSeconds: 2.6,
    spreadHip: 0.062,
    spreadAds: 0.042,
    spreadMoving: 0.014,
    bloomPerShot: 0,
    bloomMax: 0,
    bloomRecovery: 1,
    recoilPitch: 0.042,
    recoilYaw: 0.008,
    recoilRecovery: 6,
    range: 46,
    falloffStart: 9,
    falloffEnd: 26,
    falloffFloor: 0.18,
    adsFov: 60,
    adsSeconds: 0.2,
    viewModel: {
      bodyLength: 0.48,
      bodyHeight: 0.112,
      bodyWidth: 0.078,
      barrelLength: 0.42,
      barrelRadius: 0.026,
      stock: true,
      magazineDepth: 0,
      foregrip: true,
      bodyColor: '#3a3128',
      accentColor: '#6d5436',
      hipPosition: [0.17, -0.17, -0.64],
      adsPosition: [0, -0.0345, -0.95],
    },
    audio: { bodyHz: 96, decaySeconds: 0.32, noiseGain: 0.75, gain: 0.78 },
  },
];

export function damageAtDistance(def: WeaponDef, distance: number): number {
  if (distance <= def.falloffStart) return def.damage;
  if (distance >= def.falloffEnd) return def.damage * def.falloffFloor;
  const t = (distance - def.falloffStart) / (def.falloffEnd - def.falloffStart);
  return def.damage * (1 - t * (1 - def.falloffFloor));
}
