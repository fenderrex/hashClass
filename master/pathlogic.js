let map;
let overlay = null;
let underlay = null;
let imageUrl = null;
const obstacles = [];
const blockedSet = new Set();
const NONblockedSet = new Set();
let dragging = null;
let polylinePath = null;
let pathPanInterval = null;
let isPanning = false;
let circlePanInterval = null;
let circleAngle = 0;
const circleRadiusMeters = 200;
const drawnShapes = [];
const canvas   = document.getElementById('canvas');
const ctx      = canvas.getContext('2d');
const debugEl  = document.getElementById('debug');
const gridInput= document.getElementById('gridSizeInput');
const mapUpload= document.getElementById('mapUpload');
let gridSize = parseInt(gridInput.value,10);
let cols = Math.floor(canvas.width / gridSize);
let rows = Math.floor(canvas.height/ gridSize);
let start = { x: 2,      y: 2 };
let goal  = { x: cols-3, y: rows-3 };
//const obstacles   = [];         // fallback circles
//const blockedSet  = new Set();  // per-cell blocks from image
//const NONblockedSet  = new Set();  // per-cell blocks from image
//let dragging = null;
/* ───────────── debug helpers ───────────── */
const logDebug  = m => { debugEl.textContent += m + '\n'; debugEl.scrollTop = debugEl.scrollHeight; };
const clearDebug= () => { debugEl.textContent = ''; };

/* ───────────── resize / grid change ───────────── */
gridInput.addEventListener('change', ()=>{
  gridSize = parseInt(gridInput.value,10);
  cols = Math.floor(canvas.width / gridSize);
  rows = Math.floor(canvas.height/ gridSize);
  start = { x: 2,      y: 2 };
  goal  = { x: cols-3, y: rows-3 };
  obstacles.length = 0; blockedSet.clear();
  if (mapUpload.files.length) handleImageUpload(); else { randomizeObstacles(); runSelectedPath(); }
});

function debounce(func, delay) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), delay);
  };
}
document.getElementById('binary').addEventListener('change', () => {
  applyBinaryThresholdFromSlider();
  runSelectedPath();
});


function applyBinaryThresholdFromSlider() {
  if (!window.originalImageData) return;

  // Clone original data (non-destructive)
  const imageData = new ImageData(
    new Uint8ClampedArray(window.originalImageData.data),
    window.originalImageData.width,
    window.originalImageData.height
  );

  const data = imageData.data;
  const t = parseFloat(document.getElementById('binary').value);

  blockedSet.clear();
  NONblockedSet.clear();

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const i = (y * cols + x) * 4;
      const alpha = data[i + 3];
      if (alpha < t * 255) {
        data[i + 3] = 0;
        blockedSet.add(`${x},${y}`);
      } else {
        data[i + 3] = 255;
        NONblockedSet.add(`${x},${y}`);
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

/* ───────────── image upload ───────────── */
function fileSplitRoad(file){

    if (!file || !file.type.includes("png")) return;

    // Extract bbox from filename if present
    parseBboxFromFilename(file.name);

    // Try reading PNG tEXt metadata
    const reader = new FileReader();
    reader.onload = e => readPNGTextChunk(e.target.result);
    reader.readAsArrayBuffer(file);

    // Read image as DataURL for display
    const imgReader = new FileReader();
    imgReader.onload = e => {
      imageUrl = e.target.result;
      updateOverlay();
    };
	//astar
	//astar
    imgReader.readAsDataURL(file);
	const img = new Image();
  img.onload = ()=>{
    const off = document.createElement('canvas');
    off.width = cols; off.height = rows;
    const octx = off.getContext('2d');
    octx.drawImage(img,0,0,cols,rows);
const imageData = octx.getImageData(0,0,cols,rows);
window.originalImageData = imageData;  // <-- store full imageData globally
const data = imageData.data;    blockedSet.clear();
    for(let y=0;y<rows;y++){
      for(let x=0;x<cols;x++){
      const i = (y * cols + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const a = data[i + 3];

      const isTransparent = a === 0;
      const isWhite = r >= 250 && g >= 250 && b >= 250;

      if (isTransparent) {
        blockedSet.add(`${x},${y}`);
      }else{
	    NONblockedSet.add(`${x},${y}`);
	  }
      }
    }
				  //alert("cool");
	  randomizeStartAndGoal();
    logDebug(`Map loaded – blocked cells: ${blockedSet.size}`);
    runSelectedPath();
  };
  const rdr=new FileReader();
  rdr.onload=e=>img.src=e.target.result;
  rdr.readAsDataURL(file);
  randomizeStartAndGoal();

}

function pickRandomWhiteGridCell() {
  const cells = Array.from(NONblockedSet);
  if (cells.length === 0) {
    console.error("No valid NONblockedSet cells available.");
    return null;
  }
  const randomIndex = Math.floor(Math.random() * cells.length);
  const [x, y] = cells[randomIndex].split(',').map(Number);
  return { x, y };
}


function randomizeStartAndGoal() {
//alert("loaded");
  const cells = Array.from(NONblockedSet);
  if (cells.length < 2) {
    console.error('Not enough free cells to pick start and goal.');
    return;
  }

  // pick two distinct indices
  const i = Math.floor(Math.random() * cells.length);
  let j;
  do {
    j = Math.floor(Math.random() * cells.length);
  } while (j === i);

  // parse “x,y” → { x: Number, y: Number }
  const [x1, y1] = cells[i].split(',').map(Number);
  const [x2, y2] = cells[j].split(',').map(Number);

  start = { x: x1, y: y1 };
  goal  = { x: x2, y: y2 };
}
mapUpload.addEventListener('change', handleImageUpload);
  document.getElementById('imageLoader').addEventListener('change', evt => {
    const file = evt.target.files[0];
	//alert("party");
fileSplitRoad(file);
  });
function handleImageUpload(){

  const file = mapUpload.files[0]; if(!file) return;
  fileSplitRoad(file);
}

/* ───────────── obstacle fallback ───────────── */
function randomizeObstacles(){
  blockedSet.clear(); obstacles.length=0;
  for(let i=0;i<8;i++){
    obstacles.push({ cx:Math.floor(Math.random()*cols),
                     cy:Math.floor(Math.random()*rows),
                     r:4.5+Math.random()*2 });
  }
  logDebug(`Obstacles: ${obstacles.length}`);
}

/* ───────────── collision test ───────────── */
function isBlocked(x,y){
  const xi=Math.floor(x), yi=Math.floor(y);
  if(xi<0||yi<0||xi>=cols||yi>=rows) return true;
  if(blockedSet.size) return blockedSet.has(`${xi},${yi}`);
  return obstacles.some(o=>Math.hypot(o.cx-xi,o.cy-yi)<o.r+0.5);
}

/* ───────────── drawing helpers ───────────── */
function drawGrid(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle='#eee';
  for(let i=0;i<=cols;i++){ ctx.beginPath(); ctx.moveTo(i*gridSize,0); ctx.lineTo(i*gridSize,canvas.height); ctx.stroke();}
  for(let j=0;j<=rows;j++){ ctx.beginPath(); ctx.moveTo(0,j*gridSize); ctx.lineTo(canvas.width,j*gridSize);  ctx.stroke();}
}
function drawObs(){
  ctx.fillStyle='black';
  if(blockedSet.size){
    blockedSet.forEach(k=>{ const [x,y]=k.split(','); ctx.fillRect(x*gridSize,y*gridSize,gridSize,gridSize); });
  }else{
    obstacles.forEach(o=>{ ctx.beginPath(); ctx.arc(o.cx*gridSize,o.cy*gridSize,o.r*gridSize,0,2*Math.PI); ctx.fill();});
  }
}
function drawPoints(){
  ctx.fillStyle='red';  ctx.beginPath(); ctx.arc((start.x+0.5)*gridSize,(start.y+0.5)*gridSize,gridSize/2,0,2*Math.PI); ctx.fill();
  ctx.fillStyle='lime'; ctx.beginPath(); ctx.arc((goal.x +0.5)*gridSize,(goal.y +0.5)*gridSize,gridSize/2,0,2*Math.PI); ctx.fill();
}
function drawPath(path,color){
  if(!path||path.length<2) return;
  ctx.strokeStyle=color; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo((path[0].x+0.5)*gridSize,(path[0].y+0.5)*gridSize);
  path.forEach(p=>ctx.lineTo((p.x+0.5)*gridSize,(p.y+0.5)*gridSize));
  ctx.stroke();
  ctx.fillStyle=color;
  path.forEach(p=>{ ctx.beginPath(); ctx.arc((p.x+0.5)*gridSize,(p.y+0.5)*gridSize,3,0,2*Math.PI); ctx.fill();});
}

/* ───────────── mouse drag start/goal ───────────── */
function getCellFromMouse(e){
  const r=canvas.getBoundingClientRect();
  return { x:Math.floor((e.clientX-r.left)/gridSize),
           y:Math.floor((e.clientY-r.top )/gridSize)};
}
// disable default right-click menu
canvas.addEventListener('contextmenu', e => e.preventDefault());
function findNearestValidCell(targetX, targetY) {
  let minDist = Infinity;
  let nearest = null;

  for (const cellStr of NONblockedSet) {
    const [x, y] = cellStr.split(',').map(Number);
    const dist = Math.hypot(x - targetX, y - targetY);
    if (dist < minDist) {
      minDist = dist;
      nearest = { x, y };
    }
  }

  return nearest;
}

canvas.addEventListener('mousedown', e => {
  const c = getCellFromMouse(e);
  if (c.x < 0 || c.x >= cols || c.y < 0 || c.y >= rows) return;

  const nearest = findNearestValidCell(c.x, c.y);
  if (!nearest) {
    console.warn("No valid nearby cell found");
    return;
  }

  if (e.button === 0) {         // left-click → start
    start = nearest;
  } else if (e.button === 2) {  // right-click → goal
    goal = nearest;
  } else {
    return; // ignore middle-click
  }

  runSelectedPath();
});

canvas.addEventListener('mousemove',e=>{
  if(!dragging) return;
  const c=getCellFromMouse(e);
  if(c.x>=0&&c.x<cols&&c.y>=0&&c.y<rows){
    if(dragging==='start') start=c; else goal=c;
    runSelectedPath();
  }
});
canvas.addEventListener('mouseup',()=>dragging=null);
canvas.addEventListener('mouseleave',()=>dragging=null);

/* ───────────── Bresenham helper ───────────── */
function bresenhamCells(a,b){
  const pts=[]; let x0=a.x,y0=a.y; const x1=b.x,y1=b.y;
  const dx=Math.abs(x1-x0), dy=Math.abs(y1-y0);
  const sx=x0<x1?1:-1, sy=y0<y1?1:-1; let err=dx-dy;
  while(true){
    pts.push({x:x0,y:y0});
    if(x0===x1 && y0===y1) break;
    const e2=2*err;
    if(e2>-dy){ err-=dy; x0+=sx; }
    if(e2< dx){ err+=dx; y0+=sy; }
  }
  return pts;
}
function getRed(x, y) {
  if (!window.originalImageData) return 0;
  const i = (y * window.originalImageData.width + x) * 4;
  return window.originalImageData.data[i]; // Red channel
}

/* ───────────── A* with Bresenham bias ───────────── */
function astarBetween(s, g, lineSet, callbacks = {}) {
  const { onVisit = () => {}, onFinish = () => {} } = callbacks;

  const open = []; // min-heap substitute
  const gScore = new Map([[`${s.x},${s.y}`, 0]]);
  const fScore = new Map();
  const cameFrom = {};
  const hash = n => `${n.x},${n.y}`;
  const heuristic = n => Math.hypot(g.x - n.x, g.y - n.y); // Euclidean

  fScore.set(hash(s), heuristic(s));
  open.push({ node: s, f: fScore.get(hash(s)) });

  while (open.length) {
    // extract lowest-f
    let idx = 0;
    open.forEach((e, i) => { if (e.f < open[idx].f) idx = i; });
    const current = open.splice(idx, 1)[0].node;

    // visit callback
    onVisit(current);

    // goal test
    if (current.x === g.x && current.y === g.y) {
      // reconstruct path
      const path = [];
      let n = current;
      while (n) {
        path.push(n);
        n = cameFrom[hash(n)];
      }
      const result = path.reverse();

      // finish callback
      onFinish(result);

      return result;
    }

    // neighbors
    [[1,0],[-1,0],[0,1],[0,-1]].forEach(([dx, dy]) => {
      const nx = current.x + dx, ny = current.y + dy;
      if (nx < 0 || ny < 0 || nx >= cols || ny >= rows || isBlocked(nx, ny)) return;
      const nh = `${nx},${ny}`;

      // cost bias
      //const stepCost = lineSet.has(nh) ? 0.5 : 1;
	  const baseCost = lineSet.has(nh) ? 0.5 : 1;

const red = getRed(nx, ny);
const redBonus = red / 255;  // range 0–1

const pixelAlpha = getAlpha(nx, ny);
const transparencyCost = 1 - ((pixelAlpha / 2) / (255 / 2));

// Favor high-red pixels (subtract up to 0.4)
const stepCost = baseCost + (transparencyCost * 1.6) - (redBonus * 0.4);  
    const tentative = gScore.get(hash(current)) + stepCost;

      if (tentative < (gScore.get(nh) ?? Infinity)) {
        cameFrom[nh] = current;
        gScore.set(nh, tentative);

        const offLinePenalty = lineSet.has(nh) ? 0 : 0.2;
        const f = tentative + heuristic({ x: nx, y: ny }) + offLinePenalty;
        fScore.set(nh, f);

        if (!open.some(o => o.node.x === nx && o.node.y === ny)) {
          open.push({ node: { x: nx, y: ny }, f });
        }
      }
    });
  }

  // no path found
  onFinish(null);
  return null;
}
function getAlpha(x, y) {
  if (!window.originalImageData) return 255;
  if (x < 0 || y < 0 || x >= window.originalImageData.width || y >= window.originalImageData.height) return 255;
  const i = (y * window.originalImageData.width + x) * 4;
  return window.originalImageData.data[i + 3];
}

/* ───────────── line-of-sight (for Ear-cut) ───────────── */
function hasLOS(a,b){
  let x0=a.x,y0=a.y; const x1=b.x,y1=b.y;
  const dx=Math.abs(x1-x0), dy=Math.abs(y1-y0);
  const sx=x0<x1?1:-1, sy=y0<y1?1:-1; let err=dx-dy;
  while(x0!==x1||y0!==y1){
    if(isBlocked(x0,y0)) return false;
    const e2=2*err; if(e2>-dy){err-=dy;x0+=sx;} if(e2<dx){err+=dx;y0+=sy;}
  }
  return !isBlocked(x1,y1);
}

/* ───────────── path post-processing helpers (earcut, etc.) ───────────── */
function earcutRolling(path,w){
  const out=[path[0]]; let i=0;
  while(i<path.length-1){
    let end=Math.min(i+w, path.length-1), j=end;
    while(j>i+1 && !hasLOS(path[i],path[j])) j--;
    out.push(path[j]); i=j;
  }
  logDebug(`Earcut passes: ${out.length}`); return out;
}
const subdiv=(p,s=2)=>p.flatMap((pt,i)=>i<p.length-1?[...Array(s).keys()].map(k=>({
  x:pt.x+(p[i+1].x-pt.x)*(k/s), y:pt.y+(p[i+1].y-pt.y)*(k/s)})):pt);
function bezierSmooth(path,step){
  if(path.length<3) return path;
  const out=[path[0]], d=step||1/Math.max(1,path.length-1);
  for(let i=1;i<path.length-1;i++){
    const p0=path[i-1],p1=path[i],p2=path[i+1];
    for(let t=0;t<=1;t+=d){
      const x=(1-t)*(1-t)*p0.x+2*(1-t)*t*p1.x+t*t*p2.x;
      const y=(1-t)*(1-t)*p0.y+2*(1-t)*t*p1.y+t*t*p2.y;
      if(!isBlocked(x,y)) out.push({x,y});
    }
  } out.push(path[path.length-1]); return out;
}
const chaikin=p=>p.flatMap((pt,i)=>i<p.length-1?[
  {x:0.75*pt.x+0.25*p[i+1].x,y:0.75*pt.y+0.25*p[i+1].y},
  {x:0.25*pt.x+0.75*p[i+1].x,y:0.25*pt.y+0.75*p[i+1].y}]:pt);
const chaikinIter=(p,it)=>{ let r=p; for(let k=0;k<it;k++) r=chaikin(r); return r;};
function applyAlphaThreshold() {
  if (!imageUrl) return;  // Don't run if no image loaded

  const img = new Image();
  img.src = imageUrl;
  img.onload = () => {
    const offCanvas = document.createElement('canvas');
    offCanvas.width = cols;
    offCanvas.height = rows;
    const offCtx = offCanvas.getContext('2d');
    offCtx.drawImage(img, 0, 0, cols, rows);
    const imageData = offCtx.getImageData(0, 0, cols, rows);
    const data = imageData.data;

    const t = parseFloat(document.getElementById('binary').value);

    for (let i = 0; i < data.length; i += 4) {
      data[i + 3] = (data[i + 3] < t * 255) ? 0 : 255;
    }

    offCtx.putImageData(imageData, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(offCanvas, 0, 0, canvas.width, canvas.height);
  };
}
document.getElementById('binary').addEventListener('input', applyAlphaThreshold);
//let polylinePath = null;
/* ───────────── main run ───────────── */
// LIMIT_RECURSION: ensures no more than 3 nested runs
function runSelectedPath(depth = 0)  { // depth tracks recursion, max 3) {
  // Removed unused 'y' parameter to avoid confusion
  applyBinaryThresholdFromSlider();
  clearDebug(); logDebug(`Run: ${new Date().toLocaleTimeString()}`);
  drawGrid(); drawObs();
  drawPoints();

  // Cache DOM references (e.g., binary checkbox, thresholdInput) outside this function for performance
  if (document.getElementById('binary').checked) {
    // TODO: `thresholdInput` may be undefined; cache DOM ref outside or use getElementById
    const t = thresholdInput.value;            // your “threshold” slider (0–1)
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = img.data;
    const halfH = canvas.height / 2;            // top half only

    for (let y = 0; y < halfH; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const i = (y * canvas.width + x) * 4;
		data[i+3] = (data[i+3] < t * 255 ? 0 : 255);
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  /* ── Bresenham straight-shot check ── */
  const lineCells = bresenhamCells(start, goal);
  if (lineCells.every(c => !isBlocked(c.x, c.y))) {
    logDebug(`Direct Bresenham (len ${lineCells.length})`);
    drawPath(lineCells, 'gold');
    return;
  }

  /* Precompute a Set for quick "on-line?" check */
  // Consider reusing a string key generator or preallocating Set to reduce GC pressure
  const lineSet = new Set(lineCells.map(c => `${c.x},${c.y}`));

  /* Base path with A* (Bresenham-biased) */
  // PERFORMANCE: consider making `lineSet` and A* parameters reusable to reduce allocations
  const raw = astarBetween(start, goal, lineSet, {
    onFinish: path => {
      if (path) {
        console.log('found path of length', path.length);
        astarBetween(goal, start, lineSet);
      } else {
        console.log('no path found');
      }
    }
  });
  if (!raw) {
    logDebug('No path');
    return;
  }
  logDebug(`Raw length: ${raw.length}`);

  const wSz = parseInt(document.getElementById('windowSize').value, 10);
  const ear = subdiv(earcutRolling(raw, wSz), 1);
  logDebug(`After earcut: ${ear.length}`);
  const bez = bezierSmooth(ear, 1 / Math.max(1, wSz));
  logDebug(`After bezier: ${bez.length}`);

  const cIt = parseInt(document.getElementById('chaikinIter').value, 10);
  const alg = document.getElementById('algorithm').value;
  let res;
  // STRUCTURE: consider adding a `default` case to handle unexpected `alg` values
  switch (alg) {
    case 'astar_corridor_earcut':
      res = ear;
      break;
    case 'astar_corridor_earcut_bezier':
      res = bez;
      break;
    case 'astar_corridor_earcut_bezier_second':
      res = [];
      for (let i = 0; i < bez.length - 1; i++) {
        const seg = astarBetween(bez[i], bez[i + 1], lineSet);
        if (seg) {
          if (res.length) res.pop();
          res = res.concat(seg);
        }
      }
      logDebug(`2nd A* length: ${res.length}`);
      break;
    case 'astar_corridor_dijkstra_smooth':
      res = astarBetween(start, goal, new Set());
      if (res) {
        logDebug(`Dijkstra raw: ${res.length}`);
        res = subdiv(earcutRolling(res, wSz), 2);
        logDebug(`Dijkstra earcut: ${res.length}`);
      }
      break;
    case 'combo':
      res = subdiv(ear, 3);
      logDebug(`Combo length: ${res.length}`);
      break;
    case 'full_pipeline':
      res = ear;
      if (cIt > 0) {
        res = chaikinIter(res, cIt);
        logDebug(`After chaikin: ${res.length}`);
      }
      res = earcutRolling(res, wSz);
      logDebug(`Final earcut: ${res.length}`);
      break;
  }
  if (cIt > 0 && alg !== 'full_pipeline') {
    res = chaikinIter(res, cIt);
    logDebug(`Chaikin -> ${res.length}`);
  }

const color = alg === 'combo' ? 'green' : alg === 'full_pipeline' ? 'magenta' : alg.includes('bezier') ? 'purple' : 'blue';
const pathLatLngs = res.map(p => cellToLatLng(p.x, p.y));
console.log('Raw path:', pathLatLngs);
ResetPan();
startPathPan(pathLatLngs, 50,  () => {
  if (depth + 1 < 500) {
	applyBinaryThresholdFromSlider();
    const { x: gx, y: gy } = pickRandomWhiteGridCell();
    goal = { x: gx, y: gy };
	
    runSelectedPath(depth + 1);
	
  }
});
// Draw on Google Map as purple polyline
if (polylinePath) polylinePath.setMap(null);  // clear previous
polylinePath = new google.maps.Polyline({
  path: pathLatLngs,
  geodesic: true,
  strokeColor: '#800080', // purple
  strokeOpacity: 1.0,
  strokeWeight: 4
});
polylinePath.setMap(map);

// Start path pan animation like before
let pathPanInterval = null;


drawPath(res, color);
highlightRotations(res);
}

function ResetPan() {
  if (pathPanInterval) {
    clearInterval(pathPanInterval);
    pathPanInterval = null;
  }
}

function startPathPan(initialPath, intervalMs = 100, onCycleComplete) {
  ResetPan();

  if (!initialPath || !initialPath.length) {
    console.error("No valid path provided to pan.");
    return;
  }

  let idx = 0;
  const sanitized = preprocessPath(initialPath); // <-- sanitize once

  if (!sanitized.length) {
    console.error("Sanitized path is empty.");
    return;
  }

  pathPanInterval = setInterval(() => {
    const runningBox = document.getElementById('runningBox');
    const pauseBox = document.getElementById('pause');
    if (!runningBox) return;

    if (pauseBox.checked && runningBox.checked) {
      const point = sanitized[idx];
      const progressBar = document.getElementById('progressBar');
      progressBar.value = Math.floor((idx / sanitized.length) * 100);

      if (!isValidLatLng(point)) {
        console.error('Invalid LatLng at index', idx, point);
        ResetPan();
        return;
      }

      map.panTo(point);
      map.setZoom(25);
      idx++;
    }

    if (idx >= sanitized.length) {
      ResetPan();
      if (typeof onCycleComplete === 'function') {
        try { onCycleComplete(sanitized); }
        catch (e) { console.warn('onCycleComplete error:', e); }
      }
    }
  }, intervalMs);
}

function extractGoogleMapPixelsFromExistingCanvas() {
  const mapEl = document.getElementById('map');
  const mapRect = mapEl.getBoundingClientRect();

  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');

  const tiles = Array.from(document.querySelectorAll('img[src*="googleapis.com/vt"]'));

  for (const tile of tiles) {
    const x = parseFloat(tile.style.left || '0');
    const y = parseFloat(tile.style.top || '0');
    try {
      ctx.drawImage(tile, x, y);
    } catch (e) {
      console.warn('Tile draw failed:', tile.src);
    }
  }

  const centerX = Math.floor(mapRect.width / 2);
  const centerY = Math.floor(mapRect.height / 2);
  const [r, g, b, a] = ctx.getImageData(centerX, centerY, 1, 1).data;

  console.log(`📍 Map center pixel: R=${r} G=${g} B=${b} A=${a}`);
}

function getCenterPixelColor() {
  const mapEl = document.getElementById('map');
  const mapRect = mapEl.getBoundingClientRect();

  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');

  const tiles = Array.from(document.querySelectorAll('img[src*="googleapis.com/vt"]'));

  for (const tile of tiles) {
    const x = parseFloat(tile.style.left || '0');
    const y = parseFloat(tile.style.top || '0');
    try {
      ctx.drawImage(tile, x, y);
    } catch (e) {
      console.warn('Tile draw failed:', tile.src);
    }
  }

  const centerX = Math.floor(mapRect.width / 2);
  const centerY = Math.floor(mapRect.height / 2);
  const [r, g, b, a] = ctx.getImageData(centerX, centerY, 1, 1).data;

  console.log(`📍 Map center pixel: R=${r} G=${g} B=${b} A=${a}`);
}

/* ───────────── UI bindings ───────────── */
document.getElementById('run').onclick       = runSelectedPath;
document.getElementById('randomize').onclick = ()=>{ randomizeObstacles(); runSelectedPath(); };

/* ───────────── boot ───────────── */
    // Immediately grab the `key` param from the URL
    (function loadGoogleMaps() {
      const params = new URLSearchParams(window.location.search);
      const apiKey = params.get('key');
      if (!apiKey) {
        console.error('No Google Maps API key found in URL (use ?key=YOUR_KEY)');
        return;
      }
      // Prevent double-loading if this runs twice
      if (document.getElementById('gmaps-script')) return;

      const script = document.createElement('script');
      script.id = 'gmaps-script';
      script.src = 
        `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}` +
        `&libraries=drawing,geometry&callback=initMap`;
      script.async = true;
      script.defer = true;
      document.head.appendChild(script);
    })();
document.getElementById('randomize').onclick = ()=>{ randomizeObstacles(); runSelectedPath(); };

function cellToLatLng(cellX, cellY) {
  const south = parseFloat(document.getElementById('south').value);
  const west  = parseFloat(document.getElementById('west').value);
  const north = parseFloat(document.getElementById('north').value);
  const east  = parseFloat(document.getElementById('east').value);
  const rows = Math.floor(canvas.height / gridSize);
  const cols = Math.floor(canvas.width / gridSize);
  if ([south, west, north, east].some(isNaN)) {
    console.error('One or more bounding box values are invalid');
    return null;
  }
  if (cellX === undefined || cellY === undefined) {
    console.error('cellX or cellY is undefined:', cellX, cellY);
    return null;
  }
  const latRange = north - south;
  const lngRange = east - west;
  // linear interpolation with fractional cell positions works fine
  const lat = north - (cellY / rows) * latRange;
  const lng = east+(-lngRange) - (cellX / cols) * -lngRange;
  if (isNaN(lat) || isNaN(lng)) {
    console.error('Computed lat or lng is NaN', { cellX, cellY, lat, lng });
    return null;
  }
  return new google.maps.LatLng(lat, lng);
}
//let circlePanInterval = null;
//let circleAngle = 0;
//const circleRadiusMeters = 200;  // radius of circle
//const circleCenter = new google.maps.LatLng(33.7263, -116.3834); // your center
const panButton = document.getElementById('circlePanBtn');
const examplePath = [
  new google.maps.LatLng(33.7263, -116.3834),
  new google.maps.LatLng(33.7270, -116.3850),
  new google.maps.LatLng(33.7800, -116.3650)
];
let pathIndex = 0;
//let pathPanInterval = null;
// Then call:
//startPathPan(examplePath);

function isValidLatLng(point) {
  return point && (
    (typeof point.lat === 'function' && typeof point.lng === 'function') || 
    (typeof point.lat === 'number' && typeof point.lng === 'number')
  );
}
function hideMapTiles() {
map.setMapTypeId('roadmap');
overlay.setOpacity(1);
  map.setOptions({
    styles: [
      { elementType: 'geometry', stylers: [{ visibility: 'off' }] },
      { elementType: 'labels', stylers: [{ visibility: 'off' }] },
      { featureType: 'administrative', stylers: [{ visibility: 'off' }] },
      { featureType: 'poi', stylers: [{ visibility: 'off' }] },
      { featureType: 'road', stylers: [{ visibility: 'off' }] },
      { featureType: 'transit', stylers: [{ visibility: 'off' }] },
      { featureType: 'water', stylers: [{ visibility: 'off' }] }
    ]
  });
  
}
function preprocessPath(rawPath) {
  return rawPath.reduce((out, p, i) => {
    if (isValidLatLng(p)) {
      out.push(p);
    }
    else if (p && typeof p.x === 'number' && typeof p.y === 'number') {
      out.push(cellToLatLng(p.x, p.y));
    }
    else {
      console.warn(`Dropping invalid path element at index ${i}:`, p);
    }
    return out;
  }, []);
}
function showMapTiles() {
  map.setOptions({ styles: null });
  map.setMapTypeId('satellite');
  overlay.setOpacity(0);
}
// 1) Define a helper to zoom in on the current `goal`
function zoomToGoal(zoomLevel = 18) {
  // convert grid cell → LatLng
  const ll = cellToLatLng(goal.x, goal.y);
  if (!ll) {
    console.error('zoomToGoal: invalid goal cell', goal);
    return;
  }
  // center the map and set zoom
  map.setCenter(ll);
  map.setZoom(zoomLevel);
}
function startPathPan(initialPath, intervalMs = 100, onCycleComplete) {
  ResetPan(); 
  intervalMs = 50;
  if (!initialPath || !initialPath.length) return;
  if (pathPanInterval) return;

  let path = initialPath.slice();
  let idx  = 0;

  pathPanInterval = setInterval(() => {
    const sanitized = preprocessPath(path);
    if (!sanitized.length) {
      console.error("No valid LatLngs in path — cannot pan.");
      stopPathPan();
      return;
    }

    const runningBox = document.getElementById('runningBox');
    const pauseBox   = document.getElementById('pause');
    if (!runningBox) return;

    if (pauseBox.checked && runningBox.checked) {
      const point = sanitized[idx];
      const progressBar = document.getElementById('progressBar');
      progressBar.value = Math.floor(((idx+1) / sanitized.length) * 100);
      if (!isValidLatLng(point)) {
        console.error('Invalid point at index', idx, point);
        stopPathPan();
        return;
      }
      map.panTo(point);
      map.setZoom(18);
      idx++;
    }

    if (idx >= path.length) {
      if (typeof onCycleComplete === 'function') {
        try { onCycleComplete(path); }
        catch (e) { console.warn('onCycleComplete error:', e); }
      }
      //ResetPan();  // <-- stop after full path played
    }
  }, intervalMs);
}

function ResetPan() {
  if (pathPanInterval) {
    clearInterval(pathPanInterval);
    pathPanInterval = null;
  }
}
//let isPanning = false;
panButton.addEventListener('click', () => {
  if (isPanning) {
    ResetPan();
    panButton.textContent = 'Start Circle Pan';
  } else {
   runSelectedPath();
    panButton.textContent = 'Stop Circle Pan';
  }
  isPanning = !isPanning;
});
function highlightRotations(path, angleThreshold = 30) {
  for (let i = 1; i < path.length - 1; i++) {
    const a = path[i - 1], b = path[i], c = path[i + 1];
    const angle = computeTurnAngle(a, b, c);

    if (Math.abs(angle) > angleThreshold) {
      const px = (b.x + 0.5) * gridSize;
      const py = (b.y + 0.5) * gridSize;

      ctx.beginPath();
      ctx.arc(px, py, gridSize / 2, 0, 2 * Math.PI);
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 3;
      ctx.stroke();
    }
  }
}

function computeTurnAngle(a, b, c) {
  const angle = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  return angle * (180 / Math.PI);
}
function onExportGeoJSON() {
  alert("Preparing download");

  if (!drawnShapes.length) {
    alert("No shapes drawn to export.");
    return;
  }

  const features = drawnShapes.map(shape => {
    const geometry = convertOverlayToGeoJSON(shape);
    if (!geometry) return null;

    return {
      type: "Feature",
      geometry,
      properties: {}  // Add metadata here if needed
    };
  }).filter(f => f !== null);

  if (!features.length) {
    alert("No valid shapes found.");
    return;
  }

  const geojson = {
    type: "FeatureCollection",
    features
  };

  const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "shapes.geojson";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
// Converts Google Maps overlay shape to GeoJSON geometry object
function convertOverlayToGeoJSON(overlay) {
  if (overlay.getPath) {
    const path = overlay.getPath().getArray().map(latlng => [latlng.lng(), latlng.lat()]);

    if (overlay instanceof google.maps.Polygon || overlay.type === google.maps.drawing.OverlayType.POLYGON) {
      return {
        type: "Polygon",
        coordinates: [path]
      };
    }

    if (overlay instanceof google.maps.Polyline || overlay.type === google.maps.drawing.OverlayType.POLYLINE) {
      return {
        type: "LineString",
        coordinates: path
      };
    }
  }

  return null;
}
// Convert MVCArray path to [ [lng, lat], ... ] array
function pathToCoords(path) {
  const coords = [];
  for (let i = 0; i < path.getLength(); i++) {
    const latlng = path.getAt(i);
    coords.push([latlng.lng(), latlng.lat()]);
  }
  // Close polygon ring if polygon
  if (path.getLength() > 0 && path.getAt(0).equals(path.getAt(path.getLength()-1)) === false) {
    coords.push(coords[0]);
  }
  return coords;
}
// Trigger browser download of text file
function downloadTextFile(filename, text) {
  const blob = new Blob([text], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
function toggleControls() {
  const panel = document.getElementById('controls');
  panel.classList.toggle('hidden');
}
class FlatRotatingOverlay extends google.maps.OverlayView {
  constructor(url, bounds, opacity = 0.7) {
    super();
    this.url = url;
    this.bounds = bounds;
    this.opacity = opacity;
    this.height=0;
	this.width=0;
	    // Defensive initializations
    this.div = null;
    this.img = null;
    this.rotation = 0; // default rotation value
  }
  setRotationZ(deg) {
    this.rotation = deg;

    if (!this.img) {
      console.warn('setRotationZ called before this.img was ready');
      return;
    }

    this.img.style.transform = `rotateZ(${deg}deg)`;
  }
    onRemove() {
    if (this.div) {
      this.div.parentNode.removeChild(this.div);
      this.div = null;
      this.img = null;
    }
  }
  onAdd() {
  const div = document.createElement('div');
  div.style.borderStyle = 'none';
  div.style.borderWidth = '0px';
  div.style.position = 'absolute';
  div.id = 'blackOut';  
  const img = document.createElement('img');
  img.src = this.url;
  img.style.width = '100%';
  img.style.height = '100%';
  img.style.position = 'absolute';
  img.style.opacity = this.opacity;

  div.appendChild(img);
  this.div = div;
  this.img = img; // ✅ Define this.img here

  const panes = this.getPanes();
  panes.overlayLayer.appendChild(div);
}
  
  
setRotation(rx = 0, ry = 0, rz = 0) {
  this.rx = rx;   // pitch   (X)
  this.ry = ry;   // yaw     (Y)
  this.rz = rz;   // roll    (Z)

  if (this.img) {
    /* order matters: X → Y → Z keeps intuition - change if you need */
    this.img.style.transform =
      `rotateX(${this.rx}deg); rotateY(${this.ry}deg); rotateZ(${this.rz}deg);`;
  }
}
  // ← New!
  setOpacity(opacity) {
    this.opacity = opacity;
    if (this.img) {
      this.img.style.opacity = opacity;
    }
  }
/* keep your legacy single-axis call */
  setAngle(a){ this.setRotation(20,45,a % 360); }
onAdd() {
  // 1) create holder DIV
  this.holder = document.createElement('div');
  this.holder.id = 'blackOut';            // ← give it an ID
  Object.assign(this.holder.style, {
    position: 'absolute',
    overflow: 'hidden',
    willChange: 'transform',
    top: '0',
    left: '0',
    width: '100%',
    height: '100%'
  });

  // 2) create image
  this.img = document.createElement('img');
  this.img.src = this.url;
  Object.assign(this.img.style, {
    position: 'absolute',
    top: 0, left: 0,
    width: '100%', height: '100%',
    opacity: this.opacity,
    transformOrigin: 'center center'
  });

  this.holder.appendChild(this.img);

  // 3) add to the correct pane
  this.getPanes().overlayLayer.appendChild(this.holder);
}


draw() {
	const proj = this.getProjection();
	const sw = proj.fromLatLngToDivPixel(this.bounds.getSouthWest());
	const ne = proj.fromLatLngToDivPixel(this.bounds.getNorthEast());
	const angle = parseInt(map.getHeading() ?? 0, 10);
	this.holder.style.opacity= this.opacity;
  if (angle === 0) {
    this.holder.style.left   = `${sw.x}px`;
    this.holder.style.top    = `${ne.y}px`;
    this.holder.style.width  = `${ne.x - sw.x}px`;
    this.holder.style.height = `${sw.y - ne.y}px`;
	this.height=sw.y - ne.y;
	this.width=ne.x - sw.x;
	this.holder.style.transform = `rotate(${angle}deg)`;
  } else if (angle === 90) {//know bug
	//dontrender//errors prevent the view angle
	this.holder.style.height =this.holder.style.width;
	this.holder.style.width  =10;
  } else if (angle === 180) {
  	this.height=  ne.y-sw.y;
	this.width= sw.x-ne.x;
    this.holder.style.left   = `${ne.x}px`;
    this.holder.style.top    = `${sw.y}px`;
    this.holder.style.width  = `${this.width}px`;
    this.holder.style.height = `${this.height}px`;
	this.holder.style.transform = `rotate(${angle}deg)`;
	//this.holder.style.transform = 'scaleX(-1)';
  } else if (angle === 270) {//know bug
     //dontrender//errors prevent the view angle
	 this.holder.style.width  =10;
	 this.holder.style.height =10;
  }
}
  onRemove() { this.holder.remove(); }

}
  function initMap() {
    map = new google.maps.Map(document.getElementById("map"), {
      center: { lat: 33.7263, lng: -116.3834 },
      zoom: 15,
      mapTypeId: "satellite"
    });
	
	randomizeObstacles(); runSelectedPath();

    // Define overlay class after Maps API is ready
    class FlatRotatingOverlay extends google.maps.OverlayView {
      constructor(url, bounds, opacity = 1) {
        super();
        this.url = url;
        this.bounds = bounds;
        this.opacity = opacity;
      }
      onAdd() {
        this.holder = document.createElement('div');
        Object.assign(this.holder.style, {
          position: 'absolute', overflow: 'hidden', willChange: 'transform'
        });
        this.img = document.createElement('img');
        this.img.src = this.url;
        Object.assign(this.img.style, {
          position: 'absolute', top: 0, left: 0,
          width: '100%', height: '100%',
          transformOrigin: 'center center', opacity: this.opacity
        });
        this.holder.appendChild(this.img);
        this.getPanes().mapPane.appendChild(this.holder);
      }
      draw() {
        const proj = this.getProjection();
        const sw = proj.fromLatLngToDivPixel(this.bounds.getSouthWest());
        const ne = proj.fromLatLngToDivPixel(this.bounds.getNorthEast());
        Object.assign(this.holder.style, {
          left: `${sw.x}px`,
          top: `${ne.y}px`,
          width: `${ne.x - sw.x}px`,
          height: `${sw.y - ne.y}px`
        });
      }
      onRemove() {
        this.holder.remove();
      }
      setOpacity(o) {
        this.opacity = o;
        if (this.img) this.img.style.opacity = o;
      }
      setRotationZ(deg) {
        if (!this.img) return;
        this.img.style.transform = `rotateZ(${deg}deg)`;
      }
    }

    setupDrawing();
    setupUI();

	
  }
  
  //var drawnShapes=[];
  function setupDrawing() {
    drawingManager = new google.maps.drawing.DrawingManager({
      drawingMode: null,
      drawingControl: false,
      polygonOptions: { editable: true },
      polylineOptions: { editable: true },
      rectangleOptions: { editable: true },
      circleOptions: { editable: true }
    });
    drawingManager.setMap(map);
    google.maps.event.addListener(drawingManager, 'overlaycomplete', e => {
      drawnShapes.push(e.overlay);
      e.overlay.setEditable(true);
      drawingManager.setDrawingMode(null);
      document.getElementById('toggle-draw').textContent = 'Start Drawing';
    });
  }

  function setupUI() {
    const fileInput = document.getElementById('imageLoader');
    const southIn = document.getElementById('south');
    const westIn = document.getElementById('west');
    const northIn = document.getElementById('north');
    const eastIn = document.getElementById('east');
    const opacityR = document.getElementById('opacity');
    fileInput.addEventListener('change', onFileChange);
    document.getElementById('toggle-draw').addEventListener('click', onToggleDraw);
    document.getElementById('export-geojson').addEventListener('click', onExportGeoJSON);
    document.getElementById('export-png').addEventListener('click', onExportPNG);
    opacityR.addEventListener('input', () => {
     // if (overlay) overlay.setOpacity(parseFloat(opacityR.value));
    });
    [southIn, westIn, northIn, eastIn].forEach(input =>
      input.addEventListener('input', updateOverlay)
    );
    function onFileChange(evt) {
	//alert("yo");
      const file = evt.target.files[0];
      if (!file || !file.type.includes('png')) return;
      const reader = new FileReader();
      reader.onload = e => {
        imageUrl = e.target.result;
        updateOverlay();

      };
      reader.readAsDataURL(file);

    }
    function onToggleDraw(e) {
      const mode = drawingManager.getDrawingMode()
        ? null
        : google.maps.drawing.OverlayType.POLYLINE;
      drawingManager.setDrawingMode(mode);
      e.target.textContent = mode ? 'Stop Drawing' : 'Start Drawing';
    }

    function onExportPNG() {
      html2canvas(document.getElementById('map'), { useCORS: true }).then(canvas => {
        const a = document.createElement('a');
        a.href = canvas.toDataURL();
        a.download = 'map-snapshot.png';
        a.click();
      });
    }
	
	  }
 
	  function updateOverlay() {
    if (!imageUrl) return;

    const s = parseFloat(document.getElementById('south').value);
    const w = parseFloat(document.getElementById('west').value);
    const n = parseFloat(document.getElementById('north').value);
    const e = parseFloat(document.getElementById('east').value);
    const o = parseFloat(document.getElementById('opacity').value);
    if ([s, w, n, e].some(isNaN)) return;

    const sw = new google.maps.LatLng(s, w);
    const ne = new google.maps.LatLng(n, e);
    const bounds = new google.maps.LatLngBounds(sw, ne);

    if (overlay) overlay.setMap(null);
    overlay = new FlatRotatingOverlay(imageUrl, bounds, o );
    overlay.setMap(map);
	overlay.setRotationZ(30);
  }

  window.onload = initMap;



  function populateBbox(w, s, e, n) {
    document.getElementById('west').value = w;
    document.getElementById('south').value = s;
    document.getElementById('east').value = e;
    document.getElementById('north').value = n;
    updateOverlay();
  }

  function parseBboxFromFilename(filename) {
    const match = filename.match(/w(-?\d+_\d+)_s(-?\d+_\d+)_e(-?\d+_\d+)_n(-?\d+_\d+)/i);
    if (!match) return;

    // Convert underscore back to decimal
    const [_, w, s, e, n] = match;
    const bbox = [w, s, e, n].map(v => parseFloat(v.replace("_", ".")));
    if (bbox.every(v => !isNaN(v))) {
      populateBbox(...bbox);
    }
  }

  function readPNGTextChunk(arrayBuffer) {
    const data = new DataView(arrayBuffer);
    let offset = 8; // skip PNG header

    while (offset < data.byteLength) {
      const length = data.getUint32(offset); offset += 4;
      const type = String.fromCharCode(
        data.getUint8(offset), data.getUint8(offset + 1),
        data.getUint8(offset + 2), data.getUint8(offset + 3)
      );
      offset += 4;

      if (type === 'tEXt') {
        const chunk = new Uint8Array(arrayBuffer, offset, length);
        const text = new TextDecoder().decode(chunk);
        if (text.startsWith('bbox=')) {
          parseBoundingBoxText(text.slice(5));
        } else if (text.startsWith('bbox:')) {
          parseBoundingBoxText(text.slice(5));
        }
      }

      offset += length + 4; // skip CRC
    }
  }

  function parseBoundingBoxText(text) {
    const match = text.match(/south=([-\d.]+)\s+west=([-\d.]+)\s+north=([-\d.]+)\s+east=([-\d.]+)/);
    if (match) {
      const [_, s, w, n, e] = match.map(Number);
      populateBbox(w, s, e, n);
    }
  }



  document.getElementById('opacity').addEventListener('input', () => {
    if (overlay) overlay.setOpacity(parseFloat(document.getElementById('opacity').value));
  });

  ['south', 'west', 'north', 'east'].forEach(id =>
    document.getElementById(id).addEventListener('input', updateOverlay)
  );
