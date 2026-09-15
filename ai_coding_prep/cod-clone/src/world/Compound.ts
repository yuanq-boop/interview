import * as THREE from 'three';
import * as BufferGeometryUtils from 'three/addons/utils/BufferGeometryUtils.js';
import type { CollisionSystem } from '../systems/CollisionSystem';
import {
  createConcreteTexture,
  createGroundTexture,
  createMetalTexture,
  createSandbagTexture,
  createWoodTexture,
} from './Textures';

export interface EnemySpawn {
  readonly position: THREE.Vector3;
  /**
   * Waypoints leading from the spawn into the open courtyard. There is no
   * navmesh, so spawns inside buildings hand the enemy an exit route; once the
   * route is walked the enemy steers at the player directly.
   */
  readonly exitPath: readonly THREE.Vector3[];
}

export interface CompoundBuild {
  readonly group: THREE.Group;
  /** Solid meshes used for bullet impacts and line-of-sight tests. */
  readonly raycastTargets: THREE.Mesh[];
  readonly enemySpawns: EnemySpawn[];
  readonly playerSpawn: THREE.Vector3;
  readonly playerSpawnYaw: number;
  readonly ammoStations: THREE.Vector3[];
  readonly bounds: number;
}

type SurfaceKey = 'concrete' | 'wall' | 'metal' | 'crate' | 'sandbag' | 'accent' | 'rust';

interface BoxSpec {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly w: number;
  readonly h: number;
  readonly d: number;
  readonly surface: SurfaceKey;
  readonly rotationY?: number;
  readonly solid?: boolean;
}

const HALF = 37;

/**
 * Builds the compound as data, then merges every box into one mesh per surface
 * so a ~200-box level costs a handful of draw calls instead of two hundred.
 */
export function buildCompound(collision: CollisionSystem): CompoundBuild {
  const group = new THREE.Group();
  const boxes: BoxSpec[] = [];

  addPerimeter(boxes);
  addCourtyardCover(boxes);
  addBuilding(boxes, -23, -8, 'west');
  addBuilding(boxes, 23, -8, 'east');
  addCatwalk(boxes);
  addCommsTower(boxes);
  addContainerStacks(boxes);

  const materials = createMaterials();
  const geometriesBySurface = new Map<SurfaceKey, THREE.BufferGeometry[]>();
  const matrix = new THREE.Matrix4();
  const quaternion = new THREE.Quaternion();
  const position = new THREE.Vector3();
  const scale = new THREE.Vector3(1, 1, 1);
  const euler = new THREE.Euler();

  for (const box of boxes) {
    const geometry = new THREE.BoxGeometry(box.w, box.h, box.d);
    euler.set(0, box.rotationY ?? 0, 0);
    quaternion.setFromEuler(euler);
    position.set(box.x, box.y + box.h / 2, box.z);
    matrix.compose(position, quaternion, scale);
    geometry.applyMatrix4(matrix);

    const list = geometriesBySurface.get(box.surface) ?? [];
    list.push(geometry);
    geometriesBySurface.set(box.surface, list);

    if (box.solid !== false) {
      if (box.rotationY) {
        // Rotated props still collide as their axis-aligned envelope.
        const cos = Math.abs(Math.cos(box.rotationY));
        const sin = Math.abs(Math.sin(box.rotationY));
        collision.addFromCenterSize(
          position.clone(),
          new THREE.Vector3(box.w * cos + box.d * sin, box.h, box.w * sin + box.d * cos),
        );
      } else {
        collision.addFromCenterSize(position.clone(), new THREE.Vector3(box.w, box.h, box.d));
      }
    }
  }

  const raycastTargets: THREE.Mesh[] = [];
  for (const [surface, geometries] of geometriesBySurface) {
    const merged = BufferGeometryUtils.mergeGeometries(geometries, false);
    for (const geometry of geometries) geometry.dispose();
    if (!merged) continue;
    merged.computeBoundingSphere();
    const mesh = new THREE.Mesh(merged, materials[surface]);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.userData.worldSolid = true;
    group.add(mesh);
    raycastTargets.push(mesh);
  }

  const ground = createGround();
  group.add(ground);
  raycastTargets.push(ground);

  return {
    group,
    raycastTargets,
    enemySpawns: [
      // Inside the west building: out through the south gap, then into the open.
      spawn(-23, -13, [
        [-23, -4],
        [-23, 4],
        [-19, 10],
      ]),
      // Inside the east building, mirrored.
      spawn(23, -13, [
        [23, -4],
        [23, 4],
        [19, 10],
      ]),
      spawn(-29, 6, [[-24, 12]]),
      spawn(29, 6, [[24, 12]]),
      // Under the north catwalk.
      spawn(-9, -27, [[-9, -20]]),
      spawn(9, -27, [[9, -20]]),
      spawn(0, -24, []),
      spawn(-31, -25, [
        [-27, -20],
        [-25, -8],
      ]),
      spawn(31, -25, [
        [27, -20],
        [25, -8],
      ]),
      spawn(-20, 20, []),
      spawn(20, 20, []),
    ],
    playerSpawn: new THREE.Vector3(0, 0, 24),
    playerSpawnYaw: 0,
    ammoStations: [
      new THREE.Vector3(-20, 0, -4),
      new THREE.Vector3(20, 0, -4),
      new THREE.Vector3(0, 0, 16),
    ],
    bounds: HALF,
  };
}

function spawn(x: number, z: number, path: ReadonlyArray<readonly [number, number]>): EnemySpawn {
  return {
    position: new THREE.Vector3(x, 0, z),
    exitPath: path.map(([px, pz]) => new THREE.Vector3(px, 0, pz)),
  };
}

function addPerimeter(boxes: BoxSpec[]): void {
  const thickness = 1.1;
  const height = 6;
  boxes.push({ x: 0, y: 0, z: -HALF, w: HALF * 2, h: height, d: thickness, surface: 'wall' });
  boxes.push({ x: 0, y: 0, z: HALF, w: HALF * 2, h: height, d: thickness, surface: 'wall' });
  boxes.push({ x: -HALF, y: 0, z: 0, w: thickness, h: height, d: HALF * 2, surface: 'wall' });
  boxes.push({ x: HALF, y: 0, z: 0, w: thickness, h: height, d: HALF * 2, surface: 'wall' });

  // Buttresses break the flat wall silhouette and give the eye a scale cue.
  for (let i = -3; i <= 3; i += 1) {
    const offset = i * 10;
    boxes.push({ x: offset, y: 0, z: -HALF + 0.9, w: 1.4, h: height + 0.5, d: 0.9, surface: 'concrete' });
    boxes.push({ x: offset, y: 0, z: HALF - 0.9, w: 1.4, h: height + 0.5, d: 0.9, surface: 'concrete' });
    boxes.push({ x: -HALF + 0.9, y: 0, z: offset, w: 0.9, h: height + 0.5, d: 1.4, surface: 'concrete' });
    boxes.push({ x: HALF - 0.9, y: 0, z: offset, w: 0.9, h: height + 0.5, d: 1.4, surface: 'concrete' });
  }
}

function addCourtyardCover(boxes: BoxSpec[]): void {
  // Sandbag line the player starts behind.
  for (let i = -3; i <= 3; i += 1) {
    if (i === 0) continue;
    boxes.push({ x: i * 3.1, y: 0, z: 18, w: 2.6, h: 1.05, d: 0.95, surface: 'sandbag' });
    boxes.push({ x: i * 3.1, y: 1.05, z: 18, w: 2.2, h: 0.45, d: 0.8, surface: 'sandbag' });
  }

  // Staggered hard cover across the middle so no single angle holds everything.
  const covers: Array<[number, number, number]> = [
    [-11, 9, 0],
    [-4, 4, 0.3],
    [5, 6, -0.2],
    [12, 10, 0],
    [-15, -2, 0.5],
    [16, -1, -0.4],
    [-7, -6, 0],
    [7, -7, 0.25],
    [0, 0, 0],
  ];
  for (const [x, z, rotationY] of covers) {
    boxes.push({ x, y: 0, z, w: 3.4, h: 1.25, d: 1.2, surface: 'concrete', rotationY });
    boxes.push({ x, y: 1.25, z, w: 3, h: 0.22, d: 1, surface: 'accent', rotationY });
  }

  // Loose crates for silhouette variety and short cover.
  const crates: Array<[number, number, number, number]> = [
    [-19, 12, 0.3, 1.1],
    [19, 13, -0.5, 1.2],
    [-2, -12, 0.2, 1.05],
    [10, -14, 0.7, 1.3],
    [-12, -15, -0.3, 1],
    [22, 8, 0.4, 1.15],
    [-24, 16, 0.1, 1.1],
  ];
  for (const [x, z, rotationY, size] of crates) {
    boxes.push({ x, y: 0, z, w: size, h: size, d: size, surface: 'crate', rotationY });
    if (size > 1.1) {
      boxes.push({ x: x + 0.15, y: size, z: z + 0.1, w: size * 0.8, h: size * 0.8, d: size * 0.8, surface: 'crate', rotationY: rotationY + 0.4 });
    }
  }

  // Low barriers that shape lanes without blocking sightlines.
  for (let i = 0; i < 4; i += 1) {
    boxes.push({ x: -28 + i * 1.8, y: 0, z: 8 + i * 0.4, w: 1.6, h: 0.85, d: 0.7, surface: 'accent', rotationY: 0.5 });
    boxes.push({ x: 28 - i * 1.8, y: 0, z: 8 + i * 0.4, w: 1.6, h: 0.85, d: 0.7, surface: 'accent', rotationY: -0.5 });
  }
}

function addBuilding(boxes: BoxSpec[], centerX: number, centerZ: number, side: 'west' | 'east'): void {
  const halfW = 8;
  const halfD = 10;
  const height = 4.6;
  const thickness = 0.7;
  const inner = side === 'west' ? 1 : -1;

  // Long walls run unbroken; the short walls carry the doorways.
  boxes.push({
    x: centerX,
    y: 0,
    z: centerZ - halfD,
    w: halfW * 2,
    h: height,
    d: thickness,
    surface: 'wall',
  });
  boxes.push({
    x: centerX - inner * halfW,
    y: 0,
    z: centerZ,
    w: thickness,
    h: height,
    d: halfD * 2,
    surface: 'wall',
  });

  // Courtyard-facing wall with a central doorway.
  const doorHalf = 1.6;
  for (const sign of [-1, 1]) {
    boxes.push({
      x: centerX + inner * halfW,
      y: 0,
      z: centerZ + sign * (halfD / 2 + doorHalf / 2),
      w: thickness,
      h: height,
      d: halfD - doorHalf,
      surface: 'wall',
    });
  }
  boxes.push({
    x: centerX + inner * halfW,
    y: 2.6,
    z: centerZ,
    w: thickness,
    h: height - 2.6,
    d: doorHalf * 2,
    surface: 'wall',
  });

  // South wall with a second entrance so the building is not a dead end.
  boxes.push({
    x: centerX - halfW / 2 - 0.8,
    y: 0,
    z: centerZ + halfD,
    w: halfW - 1.6,
    h: height,
    d: thickness,
    surface: 'wall',
  });
  boxes.push({
    x: centerX + halfW / 2 + 0.8,
    y: 0,
    z: centerZ + halfD,
    w: halfW - 1.6,
    h: height,
    d: thickness,
    surface: 'wall',
  });

  // Roof and parapet read the building as a solid volume from the courtyard.
  boxes.push({
    x: centerX,
    y: height,
    z: centerZ,
    w: halfW * 2 + 0.6,
    h: 0.45,
    d: halfD * 2 + 0.6,
    surface: 'concrete',
  });
  for (const sign of [-1, 1]) {
    boxes.push({ x: centerX + sign * (halfW + 0.2), y: height + 0.45, z: centerZ, w: 0.4, h: 0.7, d: halfD * 2, surface: 'concrete' });
    boxes.push({ x: centerX, y: height + 0.45, z: centerZ + sign * (halfD + 0.2), w: halfW * 2, h: 0.7, d: 0.4, surface: 'concrete' });
  }

  // Interior cover and window slits.
  boxes.push({ x: centerX - inner * 3, y: 0, z: centerZ - 4, w: 2.6, h: 1.1, d: 1, surface: 'crate' });
  boxes.push({ x: centerX + inner * 2.5, y: 0, z: centerZ + 3, w: 1.2, h: 1.2, d: 1.2, surface: 'crate', rotationY: 0.4 });
  // Kept off the centre line so the doorway lane out of the building stays walkable.
  boxes.push({ x: centerX - inner * 3.4, y: 0, z: centerZ - 1, w: 3.2, h: 0.9, d: 0.8, surface: 'sandbag' });
  boxes.push({ x: centerX - inner * halfW, y: 2.9, z: centerZ - 5, w: 0.9, h: 0.9, d: 2.2, surface: 'rust', solid: false });
  boxes.push({ x: centerX - inner * halfW, y: 2.9, z: centerZ + 5, w: 0.9, h: 0.9, d: 2.2, surface: 'rust', solid: false });
}

function addCatwalk(boxes: BoxSpec[]): void {
  const deckY = 3.4;
  const z = -30;

  boxes.push({ x: 0, y: deckY, z, w: 44, h: 0.4, d: 3.6, surface: 'metal' });
  for (const sign of [-1, 1]) {
    boxes.push({ x: sign * 21, y: 0, z, w: 1, h: deckY, d: 3.6, surface: 'concrete' });
  }
  boxes.push({ x: 0, y: 0, z, w: 1, h: deckY, d: 3.6, surface: 'concrete' });

  // Railings: non-solid so the player can drop off the front edge deliberately.
  boxes.push({ x: 0, y: deckY + 0.4, z: z - 1.7, w: 44, h: 1.05, d: 0.16, surface: 'rust', solid: false });
  for (let i = -10; i <= 10; i += 1) {
    boxes.push({ x: i * 2.1, y: deckY + 0.4, z: z - 1.7, w: 0.12, h: 1.05, d: 0.12, surface: 'rust', solid: false });
  }

  // Stairs on both flanks, authored as boxes the step-up allowance can climb.
  for (const sign of [-1, 1]) {
    for (let step = 0; step < 7; step += 1) {
      boxes.push({
        x: sign * 14,
        y: 0,
        z: z + 2.6 + step * 1.1,
        w: 2.4,
        h: deckY - step * (deckY / 7),
        d: 1.1,
        surface: 'metal',
      });
    }
  }

  // Crates on the deck give the elevated angle its own cover.
  boxes.push({ x: -6, y: deckY + 0.4, z, w: 1.1, h: 1.1, d: 1.1, surface: 'crate' });
  boxes.push({ x: 7, y: deckY + 0.4, z, w: 1.1, h: 1.1, d: 1.1, surface: 'crate', rotationY: 0.3 });
}

function addCommsTower(boxes: BoxSpec[]): void {
  const x = 30;
  const z = -31;
  for (let level = 0; level < 5; level += 1) {
    const inset = level * 0.22;
    boxes.push({
      x,
      y: level * 2.4,
      z,
      w: 2.6 - inset,
      h: 2.4,
      d: 2.6 - inset,
      surface: level % 2 === 0 ? 'rust' : 'metal',
    });
  }
  boxes.push({ x, y: 12, z, w: 0.3, h: 4.5, d: 0.3, surface: 'rust', solid: false });
  boxes.push({ x, y: 13.4, z, w: 3.2, h: 0.18, d: 0.18, surface: 'accent', solid: false });
  boxes.push({ x, y: 14.6, z, w: 2.4, h: 0.18, d: 0.18, surface: 'accent', solid: false });
}

function addContainerStacks(boxes: BoxSpec[]): void {
  const stacks: Array<[number, number, number]> = [
    [-29, -18, 0.18],
    [29, 16, -0.22],
    [-16, 26, 0.5],
    [17, -24, 0.12],
  ];
  for (const [x, z, rotationY] of stacks) {
    boxes.push({ x, y: 0, z, w: 6.1, h: 2.6, d: 2.5, surface: 'metal', rotationY });
    boxes.push({ x: x + 0.4, y: 2.6, z: z + 0.2, w: 6.1, h: 2.6, d: 2.5, surface: 'rust', rotationY: rotationY + 0.08 });
    // Corner posts read as container hardware at a glance.
    for (const sx of [-1, 1]) {
      for (const sz of [-1, 1]) {
        boxes.push({
          x: x + sx * 2.9 * Math.cos(rotationY),
          y: 0,
          z: z + sz * 1.15,
          w: 0.32,
          h: 2.6,
          d: 0.32,
          surface: 'accent',
          solid: false,
        });
      }
    }
  }
}

function createMaterials(): Record<SurfaceKey, THREE.MeshStandardMaterial> {
  const concreteMap = createConcreteTexture('#8d8578', '#6d6559');
  concreteMap.repeat.set(0.22, 0.22);
  const wallMap = createConcreteTexture('#9a9382', '#726b5d');
  wallMap.repeat.set(0.18, 0.18);
  const metalMap = createMetalTexture('#5d6b74', '#3e4a52');
  metalMap.repeat.set(0.3, 0.3);
  const rustMap = createMetalTexture('#8a5233', '#5a3320');
  rustMap.repeat.set(0.3, 0.3);
  const sandbagMap = createSandbagTexture();
  sandbagMap.repeat.set(0.55, 0.9);
  const woodMap = createWoodTexture();
  woodMap.repeat.set(0.8, 0.8);

  return {
    concrete: new THREE.MeshStandardMaterial({ map: concreteMap, color: '#b9b2a4', roughness: 0.92, metalness: 0.04 }),
    wall: new THREE.MeshStandardMaterial({ map: wallMap, color: '#a9a294', roughness: 0.95, metalness: 0.03 }),
    metal: new THREE.MeshStandardMaterial({ map: metalMap, color: '#8d9aa4', roughness: 0.56, metalness: 0.62 }),
    rust: new THREE.MeshStandardMaterial({ map: rustMap, color: '#b0724a', roughness: 0.82, metalness: 0.32 }),
    crate: new THREE.MeshStandardMaterial({ map: woodMap, color: '#b08a56', roughness: 0.88, metalness: 0.05 }),
    sandbag: new THREE.MeshStandardMaterial({ map: sandbagMap, color: '#b4a884', roughness: 0.96, metalness: 0.02 }),
    accent: new THREE.MeshStandardMaterial({ color: '#c2703a', roughness: 0.7, metalness: 0.18 }),
  };
}

function createGround(): THREE.Mesh {
  const map = createGroundTexture();
  map.repeat.set(24, 24);
  const geometry = new THREE.PlaneGeometry(HALF * 2 + 6, HALF * 2 + 6);
  geometry.rotateX(-Math.PI / 2);
  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({ map, color: '#9c9585', roughness: 0.97, metalness: 0.02 }),
  );
  mesh.receiveShadow = true;
  mesh.userData.worldSolid = true;
  return mesh;
}
