import type { WeaponAudioSpec } from '../weapons/WeaponDefs';

/**
 * Fully synthesised combat audio — no sample files. Every sound is a shaped
 * noise burst plus a tonal body, which keeps the build asset-free while still
 * giving each weapon its own character.
 */
export class AudioSystem {
  private context: AudioContext | null = null;
  private master: GainNode | null = null;
  private noiseBuffer: AudioBuffer | null = null;
  private unlocked = false;
  private muted = false;
  private duck = 1;
  private rng: () => number;

  private readonly unlockListener = () => {
    void this.unlock();
    this.removeUnlockListeners();
  };

  constructor(rng: () => number) {
    this.rng = rng;
    window.addEventListener('pointerdown', this.unlockListener, { once: true });
    window.addEventListener('keydown', this.unlockListener, { once: true });
  }

  get isMuted(): boolean {
    return this.muted;
  }

  setRandom(rng: () => number): void {
    this.rng = rng;
  }

  async unlock(): Promise<void> {
    if (this.unlocked) return;
    const AudioContextClass =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextClass) return;

    this.context = new AudioContextClass();
    this.master = this.context.createGain();
    this.master.gain.value = this.muted ? 0 : 0.6;
    this.master.connect(this.context.destination);
    this.noiseBuffer = this.createNoiseBuffer(this.context);
    await this.context.resume();
    this.unlocked = true;
  }

  toggleMute(): boolean {
    this.muted = !this.muted;
    if (this.master && this.context) {
      this.master.gain.setTargetAtTime(this.muted ? 0 : 0.6, this.context.currentTime, 0.02);
    }
    return this.muted;
  }

  /** Lowers everything but the impact itself while hitstop holds. */
  setDuck(value: number): void {
    this.duck = value;
  }

  playerShot(spec: WeaponAudioSpec): void {
    const ctx = this.ready();
    if (!ctx) return;
    const now = ctx.currentTime;
    const pitch = 1 + (this.rng() - 0.5) * 0.09;

    this.noiseBurst(now, {
      gain: spec.gain * spec.noiseGain * this.duck,
      attack: 0.001,
      decay: spec.decaySeconds,
      filterType: 'bandpass',
      filterHz: 1700 * pitch,
      q: 0.8,
    });
    this.tone(now, {
      frequency: spec.bodyHz * pitch,
      endFrequency: spec.bodyHz * 0.42 * pitch,
      gain: spec.gain * 0.85 * this.duck,
      decay: spec.decaySeconds * 0.85,
      type: 'triangle',
    });
    // Mechanical click of the action cycling.
    this.noiseBurst(now + 0.028, {
      gain: spec.gain * 0.14 * this.duck,
      attack: 0.001,
      decay: 0.04,
      filterType: 'highpass',
      filterHz: 3800,
      q: 1,
    });
  }

  enemyShot(distance: number): void {
    const ctx = this.ready();
    if (!ctx) return;
    const now = ctx.currentTime;
    // Distant fire loses its high end and arrives quieter.
    const attenuation = Math.max(0.06, 1 - distance / 60);
    this.noiseBurst(now, {
      gain: 0.34 * attenuation * this.duck,
      attack: 0.002,
      decay: 0.16 + distance * 0.004,
      filterType: 'lowpass',
      filterHz: 900 + 2600 * attenuation,
      q: 0.7,
    });
    this.tone(now, {
      frequency: 120 * (1 + (this.rng() - 0.5) * 0.12),
      endFrequency: 58,
      gain: 0.3 * attenuation * this.duck,
      decay: 0.18,
      type: 'triangle',
    });
  }

  impact(): void {
    const ctx = this.ready();
    if (!ctx) return;
    this.noiseBurst(ctx.currentTime, {
      gain: 0.16 * this.duck,
      attack: 0.001,
      decay: 0.07,
      filterType: 'bandpass',
      filterHz: 2600 + this.rng() * 1800,
      q: 2.4,
    });
  }

  hitMarker(headshot: boolean): void {
    const ctx = this.ready();
    if (!ctx) return;
    this.tone(ctx.currentTime, {
      frequency: headshot ? 1560 : 1040,
      endFrequency: headshot ? 1180 : 880,
      gain: 0.2,
      decay: headshot ? 0.13 : 0.07,
      type: 'square',
    });
  }

  enemyDown(): void {
    const ctx = this.ready();
    if (!ctx) return;
    const now = ctx.currentTime;
    this.tone(now, { frequency: 420, endFrequency: 150, gain: 0.22, decay: 0.3, type: 'sawtooth' });
    this.noiseBurst(now + 0.02, {
      gain: 0.16,
      attack: 0.004,
      decay: 0.26,
      filterType: 'lowpass',
      filterHz: 700,
      q: 0.6,
    });
  }

  playerHurt(): void {
    const ctx = this.ready();
    if (!ctx) return;
    const now = ctx.currentTime;
    this.tone(now, { frequency: 180, endFrequency: 74, gain: 0.34, decay: 0.34, type: 'sawtooth' });
    this.noiseBurst(now, {
      gain: 0.2,
      attack: 0.002,
      decay: 0.3,
      filterType: 'lowpass',
      filterHz: 420,
      q: 0.8,
    });
  }

  reload(stage: 'out' | 'in'): void {
    const ctx = this.ready();
    if (!ctx) return;
    const now = ctx.currentTime;
    this.noiseBurst(now, {
      gain: 0.2 * this.duck,
      attack: 0.001,
      decay: stage === 'out' ? 0.07 : 0.05,
      filterType: 'bandpass',
      filterHz: stage === 'out' ? 1500 : 2700,
      q: 3,
    });
    this.tone(now, {
      frequency: stage === 'out' ? 300 : 520,
      endFrequency: stage === 'out' ? 180 : 320,
      gain: 0.12 * this.duck,
      decay: 0.06,
      type: 'square',
    });
  }

  dryFire(): void {
    const ctx = this.ready();
    if (!ctx) return;
    this.noiseBurst(ctx.currentTime, {
      gain: 0.13,
      attack: 0.001,
      decay: 0.035,
      filterType: 'highpass',
      filterHz: 4200,
      q: 1.6,
    });
  }

  footstep(sprinting: boolean): void {
    const ctx = this.ready();
    if (!ctx) return;
    this.noiseBurst(ctx.currentTime, {
      gain: (sprinting ? 0.11 : 0.06) * this.duck,
      attack: 0.003,
      decay: sprinting ? 0.1 : 0.08,
      filterType: 'lowpass',
      filterHz: 620 + this.rng() * 260,
      q: 0.9,
    });
  }

  explosion(): void {
    const ctx = this.ready();
    if (!ctx) return;
    const now = ctx.currentTime;
    this.noiseBurst(now, {
      gain: 0.85,
      attack: 0.002,
      decay: 0.9,
      filterType: 'lowpass',
      filterHz: 1400,
      q: 0.5,
    });
    this.tone(now, { frequency: 78, endFrequency: 28, gain: 0.7, decay: 0.75, type: 'sine' });
  }

  waveStart(wave: number): void {
    const ctx = this.ready();
    if (!ctx) return;
    const now = ctx.currentTime;
    const root = 220 + Math.min(6, wave) * 12;
    this.tone(now, { frequency: root, endFrequency: root, gain: 0.16, decay: 0.42, type: 'sawtooth' });
    this.tone(now + 0.16, {
      frequency: root * 1.5,
      endFrequency: root * 1.5,
      gain: 0.16,
      decay: 0.5,
      type: 'sawtooth',
    });
  }

  pickup(): void {
    const ctx = this.ready();
    if (!ctx) return;
    const now = ctx.currentTime;
    this.tone(now, { frequency: 620, endFrequency: 880, gain: 0.16, decay: 0.16, type: 'triangle' });
  }

  dispose(): void {
    this.removeUnlockListeners();
    void this.context?.close();
    this.context = null;
    this.master = null;
    this.noiseBuffer = null;
    this.unlocked = false;
  }

  private removeUnlockListeners(): void {
    window.removeEventListener('pointerdown', this.unlockListener);
    window.removeEventListener('keydown', this.unlockListener);
  }

  private ready(): AudioContext | null {
    if (!this.context || !this.master || this.context.state !== 'running' || this.muted) return null;
    return this.context;
  }

  private noiseBurst(
    startTime: number,
    options: {
      gain: number;
      attack: number;
      decay: number;
      filterType: BiquadFilterType;
      filterHz: number;
      q: number;
    },
  ): void {
    const ctx = this.context;
    if (!ctx || !this.master || !this.noiseBuffer) return;

    const source = ctx.createBufferSource();
    source.buffer = this.noiseBuffer;
    source.loop = true;
    // A random offset stops repeated shots from replaying identical noise.
    const offset = this.rng() * (this.noiseBuffer.duration - options.decay - 0.05);

    const filter = ctx.createBiquadFilter();
    filter.type = options.filterType;
    filter.frequency.value = options.filterHz;
    filter.Q.value = options.q;

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, startTime);
    gain.gain.exponentialRampToValueAtTime(Math.max(0.0002, options.gain), startTime + options.attack);
    gain.gain.exponentialRampToValueAtTime(0.0001, startTime + options.attack + options.decay);

    source.connect(filter).connect(gain).connect(this.master);
    source.start(startTime, Math.max(0, offset));
    source.stop(startTime + options.attack + options.decay + 0.02);
  }

  private tone(
    startTime: number,
    options: {
      frequency: number;
      endFrequency: number;
      gain: number;
      decay: number;
      type: OscillatorType;
    },
  ): void {
    const ctx = this.context;
    if (!ctx || !this.master) return;

    const oscillator = ctx.createOscillator();
    oscillator.type = options.type;
    oscillator.frequency.setValueAtTime(options.frequency, startTime);
    oscillator.frequency.exponentialRampToValueAtTime(
      Math.max(20, options.endFrequency),
      startTime + options.decay,
    );

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.0001, startTime);
    gain.gain.exponentialRampToValueAtTime(Math.max(0.0002, options.gain), startTime + 0.004);
    gain.gain.exponentialRampToValueAtTime(0.0001, startTime + options.decay);

    oscillator.connect(gain).connect(this.master);
    oscillator.start(startTime);
    oscillator.stop(startTime + options.decay + 0.02);
  }

  private createNoiseBuffer(ctx: AudioContext): AudioBuffer {
    const length = Math.floor(ctx.sampleRate * 2);
    const buffer = ctx.createBuffer(1, length, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < length; i += 1) {
      data[i] = this.rng() * 2 - 1;
    }
    return buffer;
  }
}
