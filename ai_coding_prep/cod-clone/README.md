# Blackline Ops

A Call of Duty-style first-person shooter that runs in the browser. Three.js + TypeScript
+ Vite, no external assets — every mesh, texture and sound is generated at runtime.

Clear escalating waves of hostiles in a walled compound: hold the sandbag line, push the
buildings, watch your flanks, and keep your magazine full.

## Run it

```bash
npm install
npm run dev        # http://127.0.0.1:5173
```

Click **Deploy** to lock the pointer and start. Production build:

```bash
npm run build && npm run preview
```

## Controls

| Input | Action |
| --- | --- |
| `W` `A` `S` `D` | Move |
| Mouse | Look |
| Left mouse | Fire |
| Right mouse | Aim down sights |
| `R` | Reload |
| `Shift` | Sprint |
| `Ctrl` / `C` | Crouch |
| `Space` | Jump |
| `1` `2` `3` / `Q` | Switch weapon |
| `G` | Throw grenade |
| `M` | Mute audio |
| `Esc` / `P` | Pause |

On a touch device the left stick moves, the right half of the screen looks, and the
on-screen cluster covers fire, aim, reload, jump, crouch, swap and grenade.

## What's in it

- **Three weapons** with distinct handling: MK-4 Carbine (assault rifle), VX-9 Vector
  (SMG), Breacher 12 (pump shotgun). Each has its own recoil, bloom, spread, damage
  falloff, reload timing, ADS speed and synthesised report.
- **Hitscan combat** with head / body / limb hit zones — headshots do 2.6× on the carbine —
  range-based damage falloff, and cover that genuinely blocks shots.
- **Three enemy archetypes**: riflemen hold their range and burst-fire, rushers close fast,
  heavies soak 250 damage. They take cover-aware routes out of their spawn buildings,
  strafe while engaging, and lose track of you when you break line of sight.
- **Wave director** that escalates count and composition, with a resupply lull between waves.
- **COD-style HUD**: dynamic crosshair that blooms with your spread, hitmarkers (body /
  headshot / kill), killfeed, directional damage indicators, streak multiplier, health
  regeneration with a damage vignette, and full menu / pause / death states.
- **Game feel**: hitstop on kills, trauma-based screenshake, FOV punch, recoil kick,
  view-model sway and bob, tracers, impact sparks, bullet decals, blood, shell casings,
  and grenades with line-of-sight-blocked blast damage.

## Verifying it

```bash
npm run build                 # tsc + vite
npm run inspect:canvas -- --url http://127.0.0.1:5173 --state active-play --run-id pass-1
node scripts/playtest.mjs --url http://127.0.0.1:5173 --seconds 120
```

`scripts/playtest.mjs` drives a scripted bot through the real input path and reports
kills, headshots, accuracy, wave reached and any console errors.

Test hook states for deterministic captures: `menu`, `active-play`, `heavy-combat`,
`ads-aiming`, `death`.

If your Chromium build number does not match the pinned `@playwright/test`, point both
scripts at a browser explicitly:

```bash
CHROMIUM_EXECUTABLE_PATH=/path/to/chromium node scripts/playtest.mjs
```

## Notes

All art is procedural because no image, model or audio generation keys were available.
The look is deliberately stylised low-poly rather than photoreal — soldiers and weapons
are authored primitives with baked vertex colours, and surfaces use canvas-drawn textures.
Every enemy shares one material so a full wave stays inside the draw-call budget.
