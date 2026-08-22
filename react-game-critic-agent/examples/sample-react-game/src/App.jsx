import { useEffect, useRef, useState } from 'react';

const ARENA = { width: 640, height: 480 };
const PLAYER_SIZE = 24;
const STEP = 16;

/**
 * Minimal fixture game for exercising the critic agent: an arrow-key controlled
 * square confined to a bordered arena, with a click-to-restart game-over state.
 */
export default function App() {
  const [pos, setPos] = useState({ x: ARENA.width / 2, y: ARENA.height / 2 });
  const [gameOver, setGameOver] = useState(false);
  const posRef = useRef(pos);
  posRef.current = pos;

  useEffect(() => {
    function onKeyDown(e) {
      if (gameOver) return;
      const delta = { ArrowUp: [0, -STEP], ArrowDown: [0, STEP], ArrowLeft: [-STEP, 0], ArrowRight: [STEP, 0] }[e.key];
      if (!delta) return;
      e.preventDefault();
      setPos((prev) => {
        const next = {
          x: Math.max(0, Math.min(ARENA.width - PLAYER_SIZE, prev.x + delta[0])),
          y: Math.max(0, Math.min(ARENA.height - PLAYER_SIZE, prev.y + delta[1])),
        };
        return next;
      });
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [gameOver]);

  function handleArenaClick() {
    if (gameOver) {
      setGameOver(false);
      setPos({ x: ARENA.width / 2, y: ARENA.height / 2 });
    }
  }

  return (
    <div style={{ fontFamily: 'sans-serif', textAlign: 'center', padding: 24 }}>
      <h1>Dodge the Walls</h1>
      <p>Use arrow keys to move the square. Click the arena to restart after game over.</p>
      <div
        onClick={handleArenaClick}
        style={{
          position: 'relative',
          width: ARENA.width,
          height: ARENA.height,
          margin: '0 auto',
          border: '2px solid #333',
          background: '#f4f4f4',
          overflow: 'hidden',
        }}
      >
        {!gameOver && (
          <div
            style={{
              position: 'absolute',
              left: pos.x,
              top: pos.y,
              width: PLAYER_SIZE,
              height: PLAYER_SIZE,
              background: '#3366ff',
              borderRadius: 4,
            }}
          />
        )}
        {gameOver && <div style={{ paddingTop: 200 }}>Game over — click to restart</div>}
      </div>
    </div>
  );
}
