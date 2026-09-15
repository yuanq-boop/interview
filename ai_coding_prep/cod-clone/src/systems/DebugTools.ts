import GUI from 'lil-gui';

export interface GameTuning {
  sensitivity: number;
  baseFov: number;
  exposure: number;
  maxDpr: number;
  shadows: boolean;
  godMode: boolean;
}

export class DebugTools {
  private gui: GUI | null = null;

  constructor(tuning: GameTuning, onChange: () => void) {
    const enabled = new URLSearchParams(window.location.search).has('debug');
    if (!enabled) return;

    this.gui = new GUI({ title: 'Blackline Ops — tuning' });
    this.gui.add(tuning, 'sensitivity', 0.0004, 0.006, 0.0001).name('look sensitivity');
    this.gui.add(tuning, 'baseFov', 60, 110, 1).name('field of view');
    this.gui.add(tuning, 'exposure', 0.5, 2, 0.01).onChange(onChange);
    this.gui.add(tuning, 'maxDpr', 1, 2, 0.25).onChange(onChange);
    this.gui.add(tuning, 'shadows').name('shadows').onChange(onChange);
    this.gui.add(tuning, 'godMode').name('god mode');
  }

  setHidden(hidden: boolean): void {
    if (!this.gui) return;
    if (hidden) this.gui.hide();
    else this.gui.show();
  }

  dispose(): void {
    this.gui?.destroy();
    this.gui = null;
  }
}
