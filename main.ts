// Room-by-room push-optimal Sokoban solver, v2.
//
// Reads every level in MAKEFUN_INPUT_DIR/<id>/level.txt (XSB: # wall, space floor,
// . goal, $ box, * box on a goal, @ player) and writes MAKEFUN_OUTPUT_DIR/<id>/solution.txt
// with the player's moves (u/d/l/r; an upper letter marks a push), in as few pushes as
// possible. Deterministic: no clock, no randomness; the same input always writes the same
// bytes. One run answers the whole set well inside its 60-second wall budget.
//
// How it wins: the levels are built as a hallway with independent rooms off it (a box pushed
// out of a door can never come back, so every box stays in its room and the level's optimum
// is the sum of its rooms'). The solver therefore cuts the level into rooms and solves each
// room on its own with a breadth-first search over pushes — BFS over pushes is push-optimal
// by construction, and three prunings keep it small: a box may only be pushed onto a live
// square (one from which it can still reach a goal), never into a frozen 2x2 block of
// walls and boxes, and never into a state with a frozen box off its goal (a box no push
// can ever move again). The plain-A* house baseline searches moves, knows no deadlock, and
// minimises moves instead of pushes; this one searches pushes and prunes the dead ones.
//
// v2 robustness (the final set is drawn unseen, so the solver must degrade, never die):
// states live in flat typed arrays, so a full search stays well inside the sandbox's
// memory (v1's arrays-of-arrays ran ~1 KB a state — a million-state search would have
// blown the 512 MB sandbox before its cap ever fired); state keys are exact for any
// box count (live-square indices, with a string key when a number could overflow); and
// when the breadth-first search still fills its state budget, a deterministic
// iterative-deepening A* fallback (admissible push-distance heuristic, work-based node
// caps, fixed candidate order) takes over and returns a valid solution — maybe not
// push-optimal — instead of an empty answer. A search that exhausts its space, rather
// than its budget, means unsolvable, and the fallback is not run then. The fallback
// carries a transposition table (most remaining bound per state): without it the search
// re-expands the same configurations through different push orders and dies on rooms a
// few pushes deep; with it, a state reached with remaining no greater than its stored
// best is pruned.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The most states one room search keeps before it gives up: work, never the clock. */
const MAX_STATES = 1_000_000;
/** The most nodes one fallback search expands, and the most across the whole set. */
const IDA_NODE_CAP = 1_000_000;
const IDA_BUDGET = 3_000_000;
let idaBudgetLeft = IDA_BUDGET;

/** The four directions: dx, dy, the step's letter, the push's letter. */
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** Parse XSB text. Short rows and the outer frame count as walls. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * The live squares and, for each, the fewest pushes a lone box on it needs to reach a
 * goal: a goal is live at distance 0; a square y is live at distance d + 1 when some
 * live square x at distance d lies one push ahead of it, with the player able to stand
 * behind it. A box pushed onto a dead square can never reach a goal, so such pushes are
 * pruned outright; the distances are the admissible heuristic the fallback search uses
 * (each push moves one box one push-graph edge, so a box needs at least its distance).
 */
function liveSquaresAndDist(level: Level): { live: Uint8Array; dist: Int32Array } {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const dist = new Int32Array(wall.length).fill(-1);
  const queue = [...goals];
  for (const g of goals) {
    live[g] = 1;
    dist[g] = 0;
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (wall[y] === 0 && wall[y - d] === 0 && !live[y]) {
        live[y] = 1;
        dist[y] = dist[x]! + 1;
        queue.push(y);
      }
    }
  }
  return { live, dist };
}

/**
 * The squares the player reaches from `from` with `occupied` in the way, stamped into
 * `seen` with `stamp`; returns the lowest square reached, the region's canonical name.
 */
function reach(
  level: Level,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  const { w, wall } = level;
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = k === 0 ? p - w : k === 1 ? p + w : k === 2 ? p - 1 : p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes. */
function frozenBlock(level: Level, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  const { w, wall } = level;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/**
 * Whether any box off its goal is frozen in the current `occupied` position: the least
 * fixpoint over the boxes. A box is frozen when every direction is closed to it — the
 * square ahead a wall or a frozen box, or the square behind one — because walls and
 * frozen boxes never move, so a frozen box can never be pushed again. Boxes on goals
 * count as blockers in the fixpoint but never as dead on their own. Sound by induction
 * on the fixpoint: a box enters it only behind walls or boxes already in it.
 */
function hasFrozenOffGoal(level: Level, occupied: Uint8Array, goal: Uint8Array, boxes: ArrayLike<number>): boolean {
  const { w, wall } = level;
  const frozen = new Uint8Array(wall.length);
  let changed = true;
  while (changed) {
    changed = false;
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i]!;
      if (frozen[b]) continue;
      let stuck = true;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const ahead = b + d;
        const behind = b - d;
        const aheadClosed = wall[ahead] === 1 || frozen[ahead] === 1;
        const behindClosed = wall[behind] === 1 || frozen[behind] === 1;
        if (!aheadClosed && !behindClosed) {
          stuck = false;
          break;
        }
      }
      if (stuck) {
        frozen[b] = 1;
        changed = true;
      }
    }
  }
  for (let i = 0; i < boxes.length; i++) {
    const b = boxes[i]!;
    if (frozen[b] && !goal[b]) return true;
  }
  return false;
}

/** A state's key: the region's lowest square and the boxes' squares, as live-square indices. */
interface KeyCtx {
  liveIdx: Int32Array;
  liveCount: number;
  size: number;
  useString: boolean;
}

function makeKeyCtx(level: Level, live: Uint8Array): KeyCtx {
  const size = level.wall.length;
  const liveIdx = new Int32Array(size).fill(-1);
  let liveCount = 0;
  for (let i = 0; i < size; i++) if (live[i]) liveIdx[i] = liveCount++;
  const nb = level.boxes.length;
  // A number key is exact while size * liveCount ** nb stays under 2 ** 52; past
  // that (only the whole-level search, never a room) the key is a string, still exact.
  const useString = nb * Math.log2(Math.max(liveCount, 2)) + Math.log2(size) > 52;
  return { liveIdx, liveCount, size, useString };
}

function keyOf(ctx: KeyCtx, boxes: ArrayLike<number>, low: number): number | string {
  if (ctx.useString) {
    let s = '' + low;
    for (let i = 0; i < boxes.length; i++) s += ',' + boxes[i];
    return s;
  }
  let k = low;
  for (let i = 0; i < boxes.length; i++) k = k * ctx.liveCount + ctx.liveIdx[boxes[i]!]!;
  return k;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters. */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/** The element of `a` that `b` does not hold (the one box that moved), or -1. */
function onlyIn(a: Int32Array, b: Int32Array): number {
  for (let i = 0; i < a.length; i++) {
    let foundIn = false;
    for (let j = 0; j < b.length; j++) {
      if (b[j] === a[i]) {
        foundIn = true;
        break;
      }
    }
    if (!foundIn) return a[i]!;
  }
  return -1;
}

/** Insertion sort of a tiny array, deterministic. */
function sortSmall(a: Int32Array): void {
  for (let i = 1; i < a.length; i++) {
    const v = a[i]!;
    let j = i - 1;
    while (j >= 0 && a[j]! > v) {
      a[j + 1] = a[j]!;
      j--;
    }
    a[j + 1] = v;
  }
}

/**
 * The push-optimal solution of a level: {pushes, moves}, 'capped' when the search fills
 * its state budget before deciding, or null when no solution exists. Breadth-first over
 * pushes — every edge costs one push, so the first solved state met is optimal. A state
 * is the boxes' squares plus the player's reachable region, named by its lowest square.
 * States live in flat typed arrays (one row per state) with a Map from key to row, so a
 * full search stays well inside the sandbox's memory.
 */
function solvePushOptimal(level: Level, maxStates: number): { pushes: number; moves: string } | 'capped' | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const nb = level.boxes.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const { live } = liveSquaresAndDist(level);
  const ctx = makeKeyCtx(level, live);
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  const solved = (boxes: ArrayLike<number>): boolean => {
    for (let i = 0; i < boxes.length; i++) if (!goal[boxes[i]!]) return false;
    return true;
  };
  if (level.boxes.some((b) => !live[b])) return null;
  for (const b of level.boxes) occupied[b] = 1;
  if (hasFrozenOffGoal(level, occupied, goal, level.boxes)) {
    for (const b of level.boxes) occupied[b] = 0;
    return null;
  }

  const boxStore = new Int32Array(maxStates * nb);
  const playerStore = new Int32Array(maxStates);
  const parentStore = new Int32Array(maxStates);
  const index = new Map<number | string, number>();
  const cur = new Int32Array(nb);
  const nxt = new Int32Array(nb);
  const putBoxes = (id: number, src: ArrayLike<number>): void => {
    for (let i = 0; i < nb; i++) boxStore[id * nb + i] = src[i]!;
  };
  const getBoxes = (id: number, out: Int32Array): void => {
    for (let i = 0; i < nb; i++) out[i] = boxStore[id * nb + i]!;
  };

  let count = 0;
  putBoxes(0, level.boxes);
  playerStore[0] = level.player;
  parentStore[0] = -1;
  count = 1;
  index.set(keyOf(ctx, level.boxes, reach(level, occupied, level.player, seen, ++stamp, stack)), 0);
  for (const b of level.boxes) occupied[b] = 0;
  if (solved(level.boxes)) return { pushes: 0, moves: '' };
  let found = -1;
  let capped = false;
  for (let head = 0; found < 0 && !capped && head < count; head++) {
    getBoxes(head, cur);
    for (const b of cur) occupied[b] = 1;
    reach(level, occupied, playerStore[head]!, seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < nb && found < 0 && !capped; bi++) {
      const b = cur[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        if (frozenBlock(level, occupied, goal, t)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        for (let i = 0; i < nb; i++) nxt[i] = cur[i]!;
        nxt[bi] = t;
        sortSmall(nxt);
        if (hasFrozenOffGoal(level, occupied, goal, nxt)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const key = keyOf(ctx, nxt, reach(level, occupied, b, mark, ++stamp, stack));
        occupied[t] = 0;
        occupied[b] = 1;
        if (index.has(key)) continue;
        if (count >= maxStates) {
          capped = true;
          break;
        }
        index.set(key, count);
        putBoxes(count, nxt);
        playerStore[count] = b;
        parentStore[count] = head;
        if (solved(nxt)) {
          found = count;
          break;
        }
        count++;
      }
    }
    for (const x of cur) occupied[x] = 0;
  }
  if (capped) return 'capped';
  if (found < 0) return null;

  // The path of states, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parentStore[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  const boxes = new Int32Array(nb);
  const nextBoxes = new Int32Array(nb);
  getBoxes(0, boxes);
  for (const s of path) {
    getBoxes(s, nextBoxes);
    const from = onlyIn(boxes, nextBoxes);
    const to = onlyIn(nextBoxes, boxes);
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    for (let i = 0; i < nb; i++) boxes[i] = nextBoxes[i]!;
  }
  return { pushes: path.length, moves };
}

/** Thrown inside the fallback search when its work budget runs out. */
class CapHit extends Error {}

/**
 * The fallback search: iterative-deepening A* over pushes with the admissible
 * push-distance heuristic, run only when the breadth-first search fills its state
 * budget. It returns a valid solution — not necessarily push-optimal — or null. All
 * budgets are work, never the clock: a node cap per call and a shared budget across
 * the set, so two runs write the same bytes. Candidates are ordered by their heuristic,
 * best first, so the pruning bites early; states already on the current path are
 * skipped (a solution holding such a cycle shortens by dropping it, and the
 * reconstruction replays walks from wherever the player actually stands). Deliberately
 * no skip of the exact reverse of the last push: pushing a box aside to let the player
 * through a doorway and back onto its goal is a real maneuver in these rooms, so the
 * reverse is left for the path check to catch when it truly cycles.
 */
function solveIDAStar(level: Level): { pushes: number; moves: string } | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const { live, dist } = liveSquaresAndDist(level);
  const ctx = makeKeyCtx(level, live);
  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  if (level.boxes.some((b) => !live[b])) return null;
  for (const b of level.boxes) occupied[b] = 1;
  if (hasFrozenOffGoal(level, occupied, goal, level.boxes)) {
    for (const b of level.boxes) occupied[b] = 0;
    return null;
  }
  for (const b of level.boxes) occupied[b] = 0;
  const hOf = (boxes: ArrayLike<number>): number => {
    let h = 0;
    for (let i = 0; i < boxes.length; i++) h += dist[boxes[i]!]!;
    return h;
  };
  const isSolved = (boxes: ArrayLike<number>): boolean => {
    for (let i = 0; i < boxes.length; i++) if (!goal[boxes[i]!]) return false;
    return true;
  };
  if (isSolved(level.boxes)) return { pushes: 0, moves: '' };

  interface Frame {
    from: number;
    to: number;
    dir: number;
  }
  const path: Frame[] = [];
  const onPath = new Set<number | string>();
  // Transposition table: a state's key to the most remaining bound (bound - pushes)
  // it was expanded under. Without it the search re-expands the same configurations
  // through different push orders — Sokoban is thick with transpositions — and the node
  // budget dies on rooms a few pushes deep. With it, a state reached with remaining
  // no greater than its stored best is pruned: its subtree was already searched from
  // a bound at least as generous, so nothing new hides there. Note the table keys on
  // REMAINING bound, not pushes: a state expanded at fewer pushes under a lower bound
  // was not searched as deeply as one reached at more pushes under a higher bound, so
  // keying on pushes alone would prune live branches (that exact bug returned empty
  // on the practice set in testing).
  const tt = new Map<number | string, number>();
  let nodes = 0;
  let minNext = Infinity;

  const dfs = (boxes: number[], player: number, g: number, bound: number, key: number | string): boolean => {
    if (++nodes > IDA_NODE_CAP || idaBudgetLeft <= 0) throw new CapHit();
    idaBudgetLeft--;
    const f = g + hOf(boxes);
    if (f > bound) {
      if (f < minNext) minNext = f;
      return false;
    }
    if (isSolved(boxes)) return true;
    const rem = bound - g;
    const prevRem = tt.get(key);
    if (prevRem !== undefined && prevRem >= rem) return false;
    tt.set(key, rem);
    for (const b of boxes) occupied[b] = 1;
    reach(level, occupied, player, seen, ++stamp, stack);
    const here = stamp;
    // Candidates: [heuristic after, direction, box, from, to, key] — best first.
    const cands: [number, number, number, number, number, number | string][] = [];
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di]!;
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        let hAfter = Infinity;
        let key: number | string = '';
        if (!frozenBlock(level, occupied, goal, t)) {
          const nxt = boxes.slice();
          nxt[bi] = t;
          nxt.sort((x, y) => x - y);
          if (!hasFrozenOffGoal(level, occupied, goal, nxt)) {
            hAfter = hOf(nxt);
            key = keyOf(ctx, nxt, reach(level, occupied, b, mark, ++stamp, stack));
          }
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (hAfter === Infinity || onPath.has(key)) continue;
        cands.push([hAfter, di, bi, b, t, key]);
      }
    }
    for (const x of boxes) occupied[x] = 0;
    cands.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    for (const [, di, , b, t, key] of cands) {
      const nxt = boxes.slice();
      const bi = boxes.indexOf(b);
      nxt[bi] = t;
      nxt.sort((x, y) => x - y);
      onPath.add(key);
      path.push({ from: b, to: t, dir: di });
      if (dfs(nxt, b, g + 1, bound, key)) return true;
      path.pop();
      onPath.delete(key);
    }
    return false;
  };

  for (const b of level.boxes) occupied[b] = 1;
  const startKey = keyOf(ctx, level.boxes, reach(level, occupied, level.player, seen, ++stamp, stack));
  for (const b of level.boxes) occupied[b] = 0;
  onPath.add(startKey);
  let bound = hOf(level.boxes);
  try {
    for (;;) {
      minNext = Infinity;
      if (dfs(level.boxes.slice(), level.player, 0, bound, startKey)) {
        let moves = '';
        let player = level.player;
        let boxes = level.boxes.slice();
        for (const fr of path) {
          const d = DIRS[fr.dir]![1] * w + DIRS[fr.dir]![0];
          for (const b of boxes) occupied[b] = 1;
          moves += walkTo(level, occupied, player, fr.from - d);
          for (const b of boxes) occupied[b] = 0;
          moves += DIRS[fr.dir]![3];
          player = fr.from;
          boxes[boxes.indexOf(fr.from)] = fr.to;
          boxes.sort((x, y) => x - y);
        }
        return { pushes: path.length, moves };
      }
      if (minNext === Infinity) return null;
      bound = minNext;
    }
  } catch (e) {
    if (e instanceof CapHit) return null;
    throw e;
  } finally {
    onPath.clear();
  }
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway — a box there
 * reaches no goal) out of the floor, and each part left that holds a box is a room. Null
 * when some part holds more or fewer boxes than goals, or a box starts in the hallway.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const { live } = liveSquaresAndDist(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * The level solved room by room: each room's push-optimal solution in turn, the other
 * rooms' boxes standing as walls (on their goals once their room is done), the player
 * walking on from where the last room left it. A box never leaves its room, so the rooms'
 * optima add up to the level's. A room whose breadth-first search fills its budget falls
 * back to the iterative-deepening search instead of failing the level; a room with no
 * solution at all fails it outright.
 */
function solveByRooms(level: Level): { pushes: number; moves: string } | null {
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = solvePushOptimal(level, MAX_STATES);
    if (whole === 'capped') return solveIDAStar(level);
    return whole;
  }
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    let solved = solvePushOptimal(sub, MAX_STATES);
    if (solved === 'capped') solved = solveIDAStar(sub);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return { pushes, moves };
}

/** The moves for a level in XSB: a string of u, d, l and r (upper marks a push). */
const solve = (text: string): string => {
  const level = parseLevel(text);
  if (level.boxes.length === 0) return '';
  const solved = solveByRooms(level);
  return solved === null ? '' : solved.moves;
};

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
