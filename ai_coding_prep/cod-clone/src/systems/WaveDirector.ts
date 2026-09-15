import type { EnemyArchetypeId } from '../entities/Enemy';

const MAX_CONCURRENT = 9;
const SPAWN_INTERVAL = 0.85;
const INTERMISSION = 4;

export type WavePhase = 'idle' | 'intermission' | 'spawning' | 'clearing';

/**
 * Owns wave composition and pacing. It never touches the scene; the game asks
 * it what to spawn and reports how many enemies are still standing.
 */
export class WaveDirector {
  wave = 0;
  phase: WavePhase = 'idle';

  private queue: EnemyArchetypeId[] = [];
  private spawnTimer = 0;
  private intermissionTimer = 0;

  get remaining(): number {
    return this.queue.length;
  }

  get intermissionRemaining(): number {
    return Math.max(0, this.intermissionTimer);
  }

  reset(): void {
    this.wave = 0;
    this.phase = 'idle';
    this.queue = [];
    this.spawnTimer = 0;
    this.intermissionTimer = 0;
  }

  /** Queues the next wave and returns its number. */
  beginNextWave(): number {
    this.wave += 1;
    this.queue = composeWave(this.wave);
    this.spawnTimer = 0;
    this.phase = 'spawning';
    return this.wave;
  }

  startIntermission(seconds = INTERMISSION): void {
    this.phase = 'intermission';
    this.intermissionTimer = seconds;
  }

  /**
   * Advances pacing. Calls `spawn` when a slot frees up, and returns
   * `'wave-cleared'` on the frame the last enemy of a wave dies.
   */
  update(
    delta: number,
    aliveCount: number,
    spawn: (archetype: EnemyArchetypeId) => void,
  ): 'none' | 'wave-cleared' | 'wave-started' {
    if (this.phase === 'intermission') {
      this.intermissionTimer -= delta;
      if (this.intermissionTimer <= 0) {
        this.beginNextWave();
        return 'wave-started';
      }
      return 'none';
    }

    if (this.phase === 'spawning') {
      this.spawnTimer -= delta;
      if (this.queue.length > 0 && aliveCount < MAX_CONCURRENT && this.spawnTimer <= 0) {
        const archetype = this.queue.shift();
        if (archetype) {
          spawn(archetype);
          // Later waves feed in faster so pressure keeps climbing.
          this.spawnTimer = Math.max(0.3, SPAWN_INTERVAL - this.wave * 0.04);
        }
      }
      if (this.queue.length === 0) this.phase = 'clearing';
      return 'none';
    }

    if (this.phase === 'clearing' && aliveCount === 0) {
      this.startIntermission();
      return 'wave-cleared';
    }

    return 'none';
  }
}

export function composeWave(wave: number): EnemyArchetypeId[] {
  const riflemen = 3 + Math.floor(wave * 0.85);
  const rushers = wave < 2 ? 0 : Math.min(6, Math.floor((wave - 1) * 0.9));
  const heavies = wave < 4 ? 0 : Math.min(4, Math.floor((wave - 1) / 3));

  const queue: EnemyArchetypeId[] = [];
  // Interleave so a wave does not arrive as three separate single-type blocks.
  const pools: Array<[EnemyArchetypeId, number]> = [
    ['rifleman', riflemen],
    ['rusher', rushers],
    ['heavy', heavies],
  ];
  let added = true;
  while (added) {
    added = false;
    for (const pool of pools) {
      if (pool[1] > 0) {
        queue.push(pool[0]);
        pool[1] -= 1;
        added = true;
      }
    }
  }
  return queue;
}
