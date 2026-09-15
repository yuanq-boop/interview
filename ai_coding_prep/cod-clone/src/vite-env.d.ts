/// <reference types="vite/client" />

interface ThreeGameDiagnostics {
  frame: number;
  elapsed: number;
  score: number;
  targetScore: number;
  complete: boolean;
  /** Current run state: menu, playing, paused or dead. */
  phase: string;
  wave: number;
  enemiesAlive: number;
  enemiesRemaining: number;
  kills: number;
  headshots: number;
  accuracy: number;
  weapon: string;
  ammo: { mag: number; reserve: number };
  /** Eye position of the closest living enemy, for scripted aiming. */
  nearestEnemy: {
    x: number;
    y: number;
    z: number;
    distance: number;
    archetype: string;
    visible: boolean;
  } | null;
  enemyStates: Array<{ state: string; visible: boolean; distance: number }>;
  player: {
    position: { x: number; y: number; z: number };
    speed: number;
    health: number;
    alive: boolean;
    yaw: number;
    pitch: number;
  };
  renderer: {
    calls: number;
    triangles: number;
    geometries: number;
    textures: number;
    programs: number;
  };
  canvas: {
    clientWidth: number;
    clientHeight: number;
    width: number;
    height: number;
    dpr: number;
  };
}

interface ThreeGameTestHooks {
  /** Re-seed the game RNG; all gameplay randomness must flow through it. */
  seed(value: number): void | Promise<void>;
  /** Acknowledge after setup/assets are ready; throw for unknown states. */
  setState(name: string): { state: string } | Promise<{ state: string }>;
  /** Stop simulation/state transitions immediately; keep rendering. Await optional synchronization. */
  setPausedForScreenshot(paused: boolean): void | Promise<void>;
  /** Stabilize ambient/idle visuals without requiring an unpaused simulation tick. */
  setReducedMotion(enabled: boolean): void | Promise<void>;
  /** Hide debug UI (lil-gui) before capturing. */
  hideDebugUi(hidden: boolean): void | Promise<void>;
  /** Scripted input for bot playtests: drive movement, aim and the trigger. */
  botInput?(command: {
    move?: { x: number; y: number };
    look?: { x: number; y: number };
    fire?: boolean;
    ads?: boolean;
    reload?: boolean;
    jump?: boolean;
  }): void;
}

interface Window {
  __THREE_GAME_DIAGNOSTICS__?: ThreeGameDiagnostics;
  __THREE_GAME_TEST_HOOKS__?: ThreeGameTestHooks;
}
