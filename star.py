#!/usr/bin/env python3 
"""
star.py –Interactive 3‑D path‑finding demo
==========================================
• Uses a black‑and‑white image: white pixels are walkable, black are walls.  
• Skeletonisation (OpenCVximgproc) produces 1‑pixel‑wide corridors.  
• The player follows A* paths along the skeleton; optional “boids” swarm to the
  player while avoiding crowding via a usage‑map cost.

Run
---
    python star.py my_bw_map.png

Controls
--------
Mouse hover/click : set new target (A*)  
WASD            : manual nudges (relative to facing)  
Mouse‑wheel        : zoom camera  
B                  : toggle boids swarm  
ESC /window✕     : quit

Dependencies
------------
    pip install pygame PyOpenGL Pillow opencv-python opencv-contrib-python numpy
"""
from __future__ import annotations
import sys, math, random, heapq
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import glutInit, glutBitmapCharacter, GLUT_BITMAP_HELVETICA_18
from PIL import Image, ImageDraw
import inspect
try:
    import mapbox_earcut as earcut
except ImportError:
    raise ImportError(
        "mapbox-earcut is required for corner cutting. "
        "Run `pip install mapbox-earcut` and retry."
    )
    



class BinIndex:
    """
    Build and query a spatial bin index for arbitrary 2D line segments.

    Attributes:
        bin_size (tuple[float, float]): width and height of each grid cell.
        bins (defaultdict): maps (bin_x, bin_y) to lists of segments.
        segments (list): all stored segments as [((x0, y0), (x1, y1)), ...].
    """
    def __init__(self, segments, bin_size: tuple[float,float]):
        self.bin_size = bin_size
        self.bins = defaultdict(list)
        self.segments = segments

        for seg in segments:
            (x0, y0), (x1, y1) = seg
            # sample points along the segment to assign to bins
            length = np.hypot(x1 - x0, y1 - y0)
            num = max(2, int(length / min(*bin_size)) + 1)
            xs = np.linspace(x0, x1, num)
            ys = np.linspace(y0, y1, num)
            for xi, yi in zip(xs, ys):
                bx = int(xi // bin_size[0])
                by = int(yi // bin_size[1])
                self.bins[(bx, by)].append(seg)

    def recall_by_point(self, point: tuple[float,float]) -> list:
        """
        Return all segments whose sampled points fall in the bin containing `point`.
        """
        bx = int(point[0] // self.bin_size[0])
        by = int(point[1] // self.bin_size[1])
        return self.bins.get((bx, by), [])

    def recall_by_bin(self, bin_coord: tuple[int,int]) -> list:
        """
        Return all segments in the specified bin coordinate (bin_x, bin_y).
        """
        return self.bins.get(bin_coord, [])
# -----------------------------------------------------------------------------
# 🔧  BASIC UTILS --------------------------------------------------------------

SEARCH_RADIUS = 10  # for snapping mouse world‑coords to nearest skeleton pixel



def bresenham_line(x0: int, y0: int, x1: int, y1: int):
    """Integer grid points along a Bresenham segment **always returning a list**.
    Guarantees a fallback so callers never get *None* (prevents TypeError).
    """
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1)
    err = dx - dy if dx > dy else dy - dx
    x, y = x0, y0
    pts: list[tuple[int, int]] = []
    max_iter = (dx + dy + 1) * 2  # generous upper‑bound safeguard
    while max_iter > 0:
        pts.append((x, y))
        if (x, y) == (x1, y1):
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        max_iter -= 1
    # 🛡️Fallback: even if we exceeded max_iter, return whatever we collected
    return pts
 

# -----------------------------------------------------------------------------
# 🏗️  SKELETONISATION ---------------------------------------------------------


def filter_out_loops(skel: np.ndarray) -> np.ndarray:
    print(f"Now in function: {inspect.currentframe().f_code.co_name}")
    bin_ = (skel == 255).astype(np.uint8)
    num, lab = cv2.connectedComponents(bin_, connectivity=8)
    h, w = skel.shape
    keep = np.zeros_like(bin_)
    nbrs = [(-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1)]
    for cid in range(1, num):
        mask = (lab == cid)
        ys, xs = np.nonzero(mask)
        if not xs.size:
            continue
        endpoint = False
        for x, y in zip(xs, ys):
            cnt = 0
            for dx, dy in nbrs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and bin_[ny, nx]:
                    cnt += 1
            if cnt == 1:
                endpoint = True
                break
        if endpoint:
            keep[mask] = 1
    return (keep * 255).astype(np.uint8)


def preprocess(img_bgr: np.ndarray):
    print(f"Now in function: {inspect.currentframe().f_code.co_name}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    skeleton = cv2.ximgproc.thinning(binary)
    skeleton = filter_out_loops(skeleton)
    return gray, binary, skeleton


def split_skeleton(skeleton: np.ndarray, binary: np.ndarray, thresh: float = 0.5):
    print(f"Now in function: {inspect.currentframe().f_code.co_name}")
    sk = (skeleton == 255).astype(np.uint8)
    bin01 = (binary == 255).astype(np.uint8)
    dist = cv2.distanceTransform(bin01, cv2.DIST_L2, 3)
    touch = (sk == 1) & (dist <= thresh)
    sk_cut = sk.copy(); sk_cut[touch] = 0
    num, lab = cv2.connectedComponents(sk_cut, connectivity=8)
    return [(lab == i).astype(np.uint8) * 255 for i in range(1, num)]


# -----------------------------------------------------------------------------
# 🔍  A* PATHFINDING -----------------------------------------------------------


def a_star(skel, binary, penalty, graph, usage, start, goal):
    h, w = skel.shape
    print(f"Now in function: {inspect.currentframe().f_code.co_name}")
    def heur(a, b):
        print(f"Now in function: {a,inspect.currentframe().f_code.co_name}")
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_h = [(heur(start, goal), start)]
    came = {}
    g = {start: 0.0}
    seen = set()
    nbrs = [(-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1)]
    while open_h:
        print(f"Now in function: {inspect.currentframe().f_code.co_name}")
        _, cur = heapq.heappop(open_h)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        cx, cy = cur
        crossings=0
        for dx, dy in nbrs:
            step_cost = 1.0
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h) or binary[ny, nx] == 0:
                continue
            if skel[cy, cx] == 255 and skel[ny, nx] == 255:
                step_cost -= 0.2
            for px, py in bresenham_line(cx, cy, nx, ny):
                if binary[py, px] == 0:
                    crossings += 1
            # apply per‐crossing penalty
            step_cost += crossings * penalty

            # crowd cost remains the same
            step_cost += sum(1 for px,py in bresenham_line(cx, cy, nx, ny)
                if binary[py, px] == 0)
            step_cost += 2.0 * usage[(nx, ny)]
            ng = g[cur] + step_cost
            nbr = (nx, ny)
            if nbr not in g or ng < g[nbr]:
                g[nbr] = ng
                came[nbr] = cur
                heapq.heappush(open_h, (ng + heur(nbr, goal), nbr))
    return []  # no path


# -----------------------------------------------------------------------------
# 🎨  OPENGL DRAW HELPERS ------------------------------------------------------
WINDOW = 800
FOV = 60.0
NEAR, FAR = 0.1, 1000.0


def load_texture(img: Image.Image):
    print(f"Now in function: {inspect.currentframe().f_code.co_name}")
    img = img.transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
    data = img.tobytes(); w, h = img.size
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex, w, h


def quad_floor(tid, w, h):
    print(f"Now in function: {inspect.currentframe().f_code.co_name}")
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tid)
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex3f(0, 0, 0)
    glTexCoord2f(1, 1); glVertex3f(w, 0, 0)
    glTexCoord2f(1, 0); glVertex3f(w, h, 0)
    glTexCoord2f(0, 0); glVertex3f(0, h, 0)
    glEnd()
    glDisable(GL_TEXTURE_2D)


def draw_cube(x, y, z, size=2.0, color=(1, 0, 0)):
    print(f"Now in function: {inspect.currentframe().f_code.co_name}")
    hs = size / 2
    glColor3fv(color)
    glBegin(GL_QUADS)
    # top/bot
    for s in (-hs, hs):
        glVertex3f(x - hs, y - hs, z + s)
        glVertex3f(x + hs, y - hs, z + s)
        glVertex3f(x + hs, y + hs, z + s)
        glVertex3f(x - hs, y + hs, z + s)
    # sides
    for dx, dy in ((hs, hs), (hs, -hs), (-hs, -hs), (-hs, hs)):
        glVertex3f(x + dx, y + dy, z - hs)
        glVertex3f(x + dx, y + dy, z + hs)
        glVertex3f(x + dy, y - dx, z + hs)
        glVertex3f(x + dy, y - dx, z - hs)
    glEnd()


def draw_cone(x, y, z, r=1.5, h=3.0, color=(0, 0, 1)):
    print(f"Now in function: {inspect.currentframe().f_code.co_name}")
    glColor3fv(color)
    sides = 12; step = 2 * math.pi / sides
    glBegin(GL_TRIANGLES)
    for i in range(sides):
        t0, t1 = i * step, (i + 1) * step
        x0, y0 = x + r * math.cos(t0), y + r * math.sin(t0)
        x1, y1 = x + r * math.cos(t1), y + r * math.sin(t1)
        glVertex3f(x, y, z + h)
        glVertex3f(x0, y0, z)
        glVertex3f(x1, y1, z)
    glEnd()


def draw_player(x, y, z, fx, fy):
    print(f"Now in function: {inspect.currentframe().f_code.co_name}")
    draw_cone(x, y, z)
    l = math.hypot(fx, fy) or 1.0
    fx, fy = fx / l, fy / l
    glColor3f(0.2, 0.2, 0.2)
    glBegin(GL_LINES)
    glVertex3f(x, y, z + 3.0)
    glVertex3f(x + fx * 4, y + fy * 4, z + 3.0)
    glEnd()
from typing import List, Tuple

def point_line_distance(pt, a, b):
    """Perpendicular distance from pt to line a→b."""
    x0,y0 = pt; x1,y1 = a; x2,y2 = b
    num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num/den if den else math.hypot(x0 - x1, y0 - y1)

def rdp(points: List[Tuple[float,float]], epsilon: float) -> List[Tuple[float,float]]:
    """
    Ramer–Douglas–Peucker line simplification.
    Keeps endpoints and any point whose distance to the chord > epsilon.
    """
    if len(points) < 3:
        return points[:]
    # find point of max distance
    start, end = points[0], points[-1]
    max_d, index = 0.0, 0
    for i in range(1, len(points)-1):
        d = point_line_distance(points[i], start, end)
        if d > max_d:
            max_d, index = d, i
    # if max distance is above threshold, keep that point and recurse
    if max_d > epsilon:
        left  = rdp(points[:index+1], epsilon)
        right = rdp(points[index:],     epsilon)
        # join, but drop duplicate at index
        return left[:-1] + right
    else:
        # everything is close enough to the straight line: keep only endpoints
        return [start, end]

def chaikin_smooth(path: List[Tuple[float,float]],
                   subdiv: int = 2,
                   chaikin_iters: int = 5,
                   simplify_eps: float = 1.0) -> List[Tuple[float,float]]:
    """
    1) Subdivide each segment into 'subdiv' pieces  
    2) Simplify with RDP (epsilon = simplify_eps)  
    3) Chaikin‑smooth 'chaikin_iters' times  
    """
    # ---- (1) subdivide ----
    pts = []
    for (x0,y0),(x1,y1) in zip(path, path):
        for k in range(subdiv):
            t = k / subdiv
            pts.append((x0*(1-t)+x1*t, y0*(1-t)+y1*t))
    pts.append(path[-1])

    # ---- (2) RDP simplify ----
    pts = rdp(pts, simplify_eps)

    # ---- (3) Chaikin smoothing ----
    for _ in range(chaikin_iters):
        new_pts = []
        for (x0,y0),(x1,y1) in zip(pts, pts):
            new_pts.append((0.75*x0 + 0.25*x1, 0.75*y0 + 0.25*y1))
            new_pts.append((0.25*x0 + 0.75*x1, 0.25*y0 + 0.75*y1))
        pts = [pts[0]] + new_pts + [pts[-1]]

    return pts
# -----------------------------------------------------------------------------
# 🚀  MAIN APPLICATION ---------------------------------------------------------
class App:
    PENALTY = 5
    PLAYER_STEP_MS = 3
    BOID_STEP_MS = 40
    DECAY_MS = 2000

    def __init__(self, img_path: Path):
        print(f"Now in function: {inspect.currentframe().f_code.co_name}")
        # -- OpenGL / pygame window --
        glutInit()
        pygame.init()
        pygame.display.set_caption("3D A* Navigator")
        pygame.display.set_mode((WINDOW, WINDOW), DOUBLEBUF | OPENGL)
        glViewport(0, 0, WINDOW, WINDOW)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity(); gluPerspective(FOV, WINDOW/WINDOW, NEAR, FAR)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity(); glEnable(GL_DEPTH_TEST)

        # -- load image & skeletonise --
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise RuntimeError(f"Cannot load {img_path}")
        gray, binary, skeleton = preprocess(bgr)
        comps = split_skeleton(skeleton, binary)
        self.skel = np.zeros_like(skeleton)
        for c in comps:
            self.skel = cv2.bitwise_or(self.skel, c)
        self.binary = binary

        # -- graph neighbours --
        coords = set(zip(*np.where(self.binary == 255)))
        self.graph = {}
        nbrs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        for y, x in coords:
            neigh = []
            for dx, dy in nbrs:
                nx, ny = x+dx, y+dy
                if (ny, nx) in coords:
                    neigh.append((nx, ny))
            if neigh:
                self.graph[(x, y)] = neigh

        # -- initial player --
        center = (binary.shape[1]//2, binary.shape[0]//2)
        self.player = min(self.graph.keys(), key=lambda p: (p[0]-center[0])**2 + (p[1]-center[1])**2)
        self.prev = self.player
        self.target = None
        self.path = []
        self.usage = defaultdict(int)

        # -- boids swarm --
        self.boids = []
        # manual move vector for WASD
        self.manual_move = None
        self.boid_on = False
        self.last_player = pygame.time.get_ticks()
        self.last_boid = pygame.time.get_ticks()
        self.last_decay = pygame.time.get_ticks()

        # -- camera --
        self.cam_dist, self.cam_height = 50, 25
        self.cam_fx, self.cam_fy = 0, 1
        self.cam_alpha = 0.1
        self.running = True
        self.pos_x, self.pos_y = float(self.player[0]), float(self.player[1])
        # speed in world units (grid cells) per millisecond
        self.speed = .1 
    def add_boids(self, n=20):
        print(f"Now in function: {inspect.currentframe().f_code.co_name}")
        self.boids.clear()
        nodes = list(self.graph.keys())
        for _ in range(n):
            start = random.choice(nodes)
            self.boids.append({"current": start, "path": []})

    def handle_events(self):
        print(f"Now in function: {inspect.currentframe().f_code.co_name}")
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                self.running = False
            elif e.type == KEYDOWN:
                if e.key == K_b:
                    self.boid_on = not self.boid_on
                    if self.boid_on:
                        self.add_boids()
                if e.key in (K_w, K_a, K_s, K_d):
                    dx = {K_w:0, K_s:0, K_a:-1, K_d:1}[e.key]
                    dy = {K_w:1, K_s:-1, K_a:0, K_d:0}[e.key]
                    fx, fy = self.get_forward()
                    rx, ry = -fy, fx
                    mv = (fx*dy + rx*dx, fy*dy + ry*dx)
                    self.manual_move = mv
            elif e.type == MOUSEBUTTONDOWN:
                if e.button == 1:              # left click
                    mx, my = e.pos
                    vx, vy = self.unproject(mx, my)
                    ix, iy = int(round(vx)), int(round(vy))
                    self.target = self.snap(ix, iy)
                elif e.button in (4, 5):       # mouse wheel
                    factor = 0.9 if e.button == 4 else 1.1
                    self.cam_dist *= factor
                    self.cam_height *= factor
            elif False:#e.type == MOUSEMOTION:
                mx, my = e.pos
                vx, vy = self.unproject(mx, my)
                ix, iy = int(round(vx)), int(round(vy))
                self.target = self.snap(ix, iy)
            elif e.type == MOUSEBUTTONDOWN:
                if e.button in (4, 5):
                    factor = 0.9 if e.button == 4 else 1.1
                    self.cam_dist *= factor; self.cam_height *= factor

    def snap(self, x, y):
        print(f"Now in function: {inspect.currentframe().f_code.co_name}")
        best, bd = None, (SEARCH_RADIUS+1)**2
        for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS):
            for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS):
                nx, ny = x+dx, y+dy
                if (nx, ny) in self.graph:
                    d2 = dx*dx + dy*dy
                    if d2 < bd:
                        bd = d2; best = (nx, ny)
        return best

    def unproject(self, mx, my):
        print(f"Now in function: {inspect.currentframe().f_code.co_name}")
        m = glGetDoublev(GL_MODELVIEW_MATRIX)
        p = glGetDoublev(GL_PROJECTION_MATRIX)
        v = glGetIntegerv(GL_VIEWPORT)
        yy = v[3] - my
        near = gluUnProject(mx, yy, 0, m, p, v)
        far = gluUnProject(mx, yy, 1, m, p, v)
        t = -near[2]/(far[2]-near[2])
        return near[0]+t*(far[0]-near[0]), near[1]+t*(far[1]-near[1])

    def get_forward(self):
        print(f"Now in function: {inspect.currentframe().f_code.co_name}")
        dx = self.player[0] - self.prev[0]; dy = self.player[1] - self.prev[1]
        l = math.hypot(dx, dy)
        return (dx/l, dy/l) if l>0 else (0,1)

    def update_player(self):
        now = pygame.time.get_ticks()
        dt  = now - self.last_player
        if dt <= 0:
            return
        self.last_player = now

        # only compute A* if no outstanding path:
        if self.target and self.player != self.target and not self.path:
            raw_path = a_star(self.skel, self.binary, self.PENALTY,
                              self.graph, self.usage,
                              self.player, self.target)
            if raw_path:
                sm = chaikin_smooth(raw_path, chaikin_iters=5)
                self.path = sm
            else:
                self.target = None

        # move along path at constant speed
        dist_to_move = dt * self.speed
        while self.path and dist_to_move > 0:
            nxt_x, nxt_y = self.path[0]
            dx = nxt_x - self.pos_x
            dy = nxt_y - self.pos_y
            seg_len = math.hypot(dx, dy)
            if seg_len <= dist_to_move:
                # consume entire segment
                self.pos_x, self.pos_y = nxt_x, nxt_y
                dist_to_move -= seg_len
                self.player = (nxt_x, nxt_y)
                self.prev   = self.player
                self.path.pop(0)
            else:
                # partial progress
                frac = dist_to_move / seg_len
                self.pos_x += dx * frac
                self.pos_y += dy * frac
                # round to nearest grid for collision/usage maps:
                self.player = self.pos_x,self.pos_y
                dist_to_move = 0

        # if no path left, nothing to do until next target


    def update_boids(self):
        print(f"Now in function: {inspect.currentframe().f_code.co_name}")
        now = pygame.time.get_ticks()
        if not self.boid_on or not self.boids:
            return
        # decay usage map periodically
        if now - self.last_decay > self.DECAY_MS:
            for k in list(self.usage.keys()):
                self.usage[k] //= 2
                if self.usage[k] == 0:
                    del self.usage[k]
            self.last_decay = now
        # update each boid along A* path
        for b in self.boids:
            if not b['path']:
                # compute path once; skip if unsolvable
                path = a_star(self.skel, self.binary, self.PENALTY, self.graph, self.usage, b['current'], self.player)
                if path:
                    b['path'] = path
            if b['path']:
                nxt = b['path'].pop(0)
                self.usage[nxt] += 1
                b['current'] = nxt



    def draw(self):
        print(f"Now in function: OTHER {inspect.currentframe().f_code.co_name}")
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); gluPerspective(FOV, WINDOW/WINDOW, NEAR, FAR)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        # camera
        fx, fy = self.get_forward()
        self.cam_alpha = 0.02
        #self.cam_fx  += (fx - self.cam_fx) * self.cam_alpha#+= (fx - self.cam_fx)*self.cam_alpha; self.cam_fy += (fy - self.cam_fy)*self.cam_alpha   
        ca,da=math.atan2(self.cam_fy,self.cam_fx),math.atan2(fy,fx);angle=ca+(((da-ca+math.pi)%(2*math.pi)-math.pi)*self.cam_alpha);self.cam_fx,self.cam_fy=math.cos(angle),math.sin(angle)

        cx = self.player[0] - self.cam_fx*self.cam_dist; cy = self.player[1] - self.cam_fy*self.cam_dist; cz=self.cam_height
        gluLookAt(cx, cy, cz, self.player[0], self.player[1], 0, 0, 0, 1)
        # floor
        h, w = self.binary.shape; gray = cv2.cvtColor(self.binary, cv2.COLOR_GRAY2RGB)
        pil = Image.fromarray(gray).convert("RGBA"); tex, tw, th = load_texture(pil)
        quad_floor(tex, tw, th)
        glDeleteTextures([tex])
        # boids & player
        for b in self.boids: draw_cube(b['current'][0], b['current'][1], 1)
        fx_play, fy_play = self.get_forward()
        draw_player(self.pos_x, self.pos_y, 0, fx_play, fy_play)
        # HUD
                # in App.draw(), after quad_floor(...)
        if self.target is not None:
            # draw the goal as a green cube  (or any other marker)
            tx, ty = self.target
            # push/pop so it doesn’t inherit any unwanted transforms
            glPushMatrix()
            glTranslatef(tx, ty, 0.5)    # lift it slightly off the ground
            glScalef(1.0, 1.0, 1.0)      # size of the marker
            glColor3f(0.0, 1.0, 0.0)     # bright green
            # you can re-use draw_cube or write a small sphere/quad
            draw_cube(0, 0, 0, size=1.0, color=(0,1,0))
            glPopMatrix()
        
        
        
        
        
        
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0, WINDOW, 0, WINDOW, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glColor3f(1,1,1)
        status = f"Boids: {'On' if self.boid_on else 'Off'}"
        glRasterPos2f(10, WINDOW-20)
        for ch in status: glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)
        pygame.display.flip()

    def run(self):
        print(f"Now in function: {inspect.currentframe().f_code.co_name}")
        while self.running:
            self.handle_events(); self.update_player(); self.update_boids(); self.draw()
        pygame.quit()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python star.py path_to_map.png")
        sys.exit(1)
    App(Path(sys.argv[1])).run()
