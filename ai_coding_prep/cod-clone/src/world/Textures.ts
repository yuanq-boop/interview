import * as THREE from 'three';
import { createSeededRandom } from '../utils/random';

/**
 * Canvas-drawn surface textures. Every generator uses a fixed seed so the
 * compound looks identical across reloads and screenshot baselines stay stable.
 */
function createCanvas(size: number): { canvas: HTMLCanvasElement; context: CanvasRenderingContext2D } {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('2D canvas context unavailable for texture generation.');
  return { canvas, context };
}

function finish(canvas: HTMLCanvasElement): THREE.CanvasTexture {
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.anisotropy = 4;
  return texture;
}

export function createConcreteTexture(light: string, dark: string): THREE.CanvasTexture {
  const size = 256;
  const { canvas, context } = createCanvas(size);
  const rng = createSeededRandom(0x5ea1);

  context.fillStyle = light;
  context.fillRect(0, 0, size, size);

  // Blotchy aggregate.
  for (let i = 0; i < 900; i += 1) {
    const radius = 1 + rng() * 5;
    context.fillStyle = `rgba(0, 0, 0, ${0.02 + rng() * 0.07})`;
    context.beginPath();
    context.arc(rng() * size, rng() * size, radius, 0, Math.PI * 2);
    context.fill();
  }
  for (let i = 0; i < 260; i += 1) {
    context.fillStyle = `rgba(255, 255, 255, ${0.02 + rng() * 0.05})`;
    context.beginPath();
    context.arc(rng() * size, rng() * size, 1 + rng() * 3, 0, Math.PI * 2);
    context.fill();
  }

  // Panel seams give the eye a repeating scale reference.
  context.strokeStyle = dark;
  context.globalAlpha = 0.5;
  context.lineWidth = 2;
  for (const offset of [0, size / 2]) {
    context.beginPath();
    context.moveTo(0, offset);
    context.lineTo(size, offset);
    context.moveTo(offset, 0);
    context.lineTo(offset, size);
    context.stroke();
  }

  // Streaks of weathering running down from the seams.
  context.globalAlpha = 0.16;
  context.lineWidth = 1;
  for (let i = 0; i < 40; i += 1) {
    const x = rng() * size;
    const top = rng() * size;
    context.strokeStyle = dark;
    context.beginPath();
    context.moveTo(x, top);
    context.lineTo(x + (rng() - 0.5) * 6, top + 20 + rng() * 60);
    context.stroke();
  }
  context.globalAlpha = 1;

  return finish(canvas);
}

export function createMetalTexture(light: string, dark: string): THREE.CanvasTexture {
  const size = 256;
  const { canvas, context } = createCanvas(size);
  const rng = createSeededRandom(0x7b12);

  context.fillStyle = light;
  context.fillRect(0, 0, size, size);

  // Corrugation: the vertical ribs that read as a shipping container.
  const ribWidth = size / 16;
  for (let i = 0; i < 16; i += 1) {
    const gradient = context.createLinearGradient(i * ribWidth, 0, (i + 1) * ribWidth, 0);
    gradient.addColorStop(0, 'rgba(0, 0, 0, 0.28)');
    gradient.addColorStop(0.45, 'rgba(255, 255, 255, 0.12)');
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0.24)');
    context.fillStyle = gradient;
    context.fillRect(i * ribWidth, 0, ribWidth, size);
  }

  // Rust bloom and scratches.
  for (let i = 0; i < 180; i += 1) {
    context.fillStyle = `rgba(${90 + rng() * 60 | 0}, ${45 + rng() * 30 | 0}, 25, ${0.05 + rng() * 0.2})`;
    context.beginPath();
    context.arc(rng() * size, rng() * size, 2 + rng() * 9, 0, Math.PI * 2);
    context.fill();
  }
  context.strokeStyle = dark;
  context.globalAlpha = 0.3;
  for (let i = 0; i < 60; i += 1) {
    const x = rng() * size;
    const y = rng() * size;
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(x + (rng() - 0.5) * 30, y + (rng() - 0.5) * 12);
    context.stroke();
  }
  context.globalAlpha = 1;

  return finish(canvas);
}

export function createGroundTexture(): THREE.CanvasTexture {
  const size = 256;
  const { canvas, context } = createCanvas(size);
  const rng = createSeededRandom(0x31f7);

  context.fillStyle = '#8f8876';
  context.fillRect(0, 0, size, size);

  // Grit.
  for (let i = 0; i < 2600; i += 1) {
    const shade = rng();
    context.fillStyle =
      shade > 0.72
        ? `rgba(60, 54, 44, ${0.12 + rng() * 0.3})`
        : `rgba(214, 206, 186, ${0.06 + rng() * 0.18})`;
    context.fillRect(rng() * size, rng() * size, 1 + rng() * 2.4, 1 + rng() * 2.4);
  }

  // Loose gravel.
  for (let i = 0; i < 160; i += 1) {
    context.fillStyle = `rgba(${70 + rng() * 60 | 0}, ${66 + rng() * 54 | 0}, ${56 + rng() * 44 | 0}, 0.55)`;
    context.beginPath();
    context.ellipse(rng() * size, rng() * size, 1.5 + rng() * 3, 1 + rng() * 2, rng() * Math.PI, 0, Math.PI * 2);
    context.fill();
  }

  // Dried cracks.
  context.strokeStyle = 'rgba(52, 46, 37, 0.3)';
  context.lineWidth = 1.2;
  for (let i = 0; i < 14; i += 1) {
    let x = rng() * size;
    let y = rng() * size;
    context.beginPath();
    context.moveTo(x, y);
    for (let step = 0; step < 6; step += 1) {
      x += (rng() - 0.5) * 34;
      y += (rng() - 0.5) * 34;
      context.lineTo(x, y);
    }
    context.stroke();
  }

  return finish(canvas);
}

export function createSandbagTexture(): THREE.CanvasTexture {
  const size = 256;
  const { canvas, context } = createCanvas(size);
  const rng = createSeededRandom(0x9c44);

  context.fillStyle = '#a2966f';
  context.fillRect(0, 0, size, size);

  // Stacked sacks: overlapping rounded lozenges in two offset courses.
  const rows = 4;
  const rowHeight = size / rows;
  for (let row = 0; row < rows; row += 1) {
    const offset = row % 2 === 0 ? 0 : size / 6;
    for (let i = -1; i < 4; i += 1) {
      const x = offset + i * (size / 3);
      const y = row * rowHeight + rowHeight / 2;
      const shade = 150 + rng() * 40;
      context.fillStyle = `rgb(${shade | 0}, ${(shade * 0.92) | 0}, ${(shade * 0.68) | 0})`;
      context.beginPath();
      context.ellipse(x, y, size / 6.4, rowHeight * 0.42, 0, 0, Math.PI * 2);
      context.fill();
      context.strokeStyle = 'rgba(70, 62, 44, 0.42)';
      context.lineWidth = 2;
      context.stroke();
    }
  }

  // Weave grain and dirt.
  for (let i = 0; i < 2200; i += 1) {
    context.fillStyle = `rgba(${60 + rng() * 50 | 0}, ${54 + rng() * 44 | 0}, 38, ${0.05 + rng() * 0.12})`;
    context.fillRect(rng() * size, rng() * size, 1.6, 1.1);
  }

  return finish(canvas);
}

export function createWoodTexture(): THREE.CanvasTexture {
  const size = 256;
  const { canvas, context } = createCanvas(size);
  const rng = createSeededRandom(0x4d17);

  context.fillStyle = '#8a6a40';
  context.fillRect(0, 0, size, size);

  // Planks with dark seams between them.
  const plankCount = 5;
  const plankHeight = size / plankCount;
  for (let i = 0; i < plankCount; i += 1) {
    const shade = 118 + rng() * 44;
    context.fillStyle = `rgb(${shade | 0}, ${(shade * 0.76) | 0}, ${(shade * 0.48) | 0})`;
    context.fillRect(0, i * plankHeight, size, plankHeight - 2);
    context.fillStyle = 'rgba(42, 30, 18, 0.55)';
    context.fillRect(0, i * plankHeight + plankHeight - 2, size, 2);

    // Grain lines running along the plank.
    context.strokeStyle = 'rgba(58, 40, 22, 0.24)';
    context.lineWidth = 1;
    for (let g = 0; g < 7; g += 1) {
      const y = i * plankHeight + 3 + rng() * (plankHeight - 8);
      context.beginPath();
      context.moveTo(0, y);
      for (let x = 0; x <= size; x += 32) {
        context.lineTo(x, y + (rng() - 0.5) * 3);
      }
      context.stroke();
    }
  }

  // Corner banding hints at a crate rather than a plain board.
  context.strokeStyle = 'rgba(70, 52, 30, 0.5)';
  context.lineWidth = 6;
  context.strokeRect(3, 3, size - 6, size - 6);

  return finish(canvas);
}

/** Radial soft-edged disc used for bullet holes and blood decals. */
export function createDecalTexture(inner: string, outer: string): THREE.CanvasTexture {
  const size = 64;
  const { canvas, context } = createCanvas(size);
  const gradient = context.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  gradient.addColorStop(0, inner);
  gradient.addColorStop(0.55, outer);
  gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/** Vertical gradient sky dome texture: horizon haze up to deep zenith. */
export function createSkyTexture(): THREE.CanvasTexture {
  const width = 16;
  const height = 256;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('2D canvas context unavailable for sky generation.');

  const gradient = context.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, '#20344f');
  gradient.addColorStop(0.42, '#5a6f86');
  gradient.addColorStop(0.72, '#a8a08c');
  gradient.addColorStop(0.88, '#d8c19a');
  gradient.addColorStop(1, '#e8d2ab');
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  return texture;
}
