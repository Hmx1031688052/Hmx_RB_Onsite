"""Lightweight browser visualizer for live lidar and 3D detections."""

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>PointPillars 实时感知</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #080d14; color: #dce7f3; overflow: hidden; }
    header { height: 54px; display: flex; align-items: center; gap: 20px;
      padding: 0 18px; background: #111a25; border-bottom: 1px solid #263445; }
    h1 { margin: 0; font-size: 17px; font-weight: 650; }
    .pill { padding: 4px 9px; border-radius: 999px; background: #263445; font-size: 12px; }
    #state.live { background: #144b38; color: #8bf0c5; }
    #state.stale { background: #54351b; color: #ffc982; }
    main { display: grid; grid-template-columns: minmax(0, 1fr) 330px;
      height: calc(100vh - 54px); }
    #stage { position: relative; min-width: 0; }
    canvas { width: 100%; height: 100%; display: block; }
    aside { padding: 14px; overflow: auto; background: #0e1620;
      border-left: 1px solid #263445; }
    .summary { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }
    .card { padding: 10px; border: 1px solid #273649; border-radius: 8px; background: #121d29; }
    .card b { display: block; font-size: 19px; margin-top: 2px; }
    .label { color: #8fa4ba; font-size: 11px; text-transform: uppercase; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th { color: #8298ae; text-align: left; position: sticky; top: -14px; background: #0e1620; }
    td, th { padding: 7px 5px; border-bottom: 1px solid #213041; white-space: nowrap; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
    .help { position: absolute; left: 14px; bottom: 12px; color: #8095aa;
      background: rgba(8,13,20,.78); padding: 6px 9px; border-radius: 6px; font-size: 11px; }
    .legend { position: absolute; left: 14px; top: 12px; display: flex; gap: 12px;
      background: rgba(8,13,20,.82); padding: 7px 10px; border-radius: 6px; font-size: 11px; }
    .legend-line { display: inline-block; width: 24px; height: 3px; margin-right: 5px;
      vertical-align: middle; border-radius: 2px; }
  </style>
</head>
<body>
  <header>
    <h1>PointPillars 实时感知</h1>
    <span id="state" class="pill stale">等待点云</span>
    <span id="meta" class="pill">--</span>
    <span id="planMeta" class="pill">Plan: --</span>
  </header>
  <main>
    <section id="stage">
      <canvas id="view"></canvas>
      <div class="legend">
        <span><i class="legend-line" style="background:#39d5ff"></i>Global path</span>
        <span><i class="legend-line" style="background:#ffd166"></i>Local path</span>
        <span><i class="legend-line" style="background:#e879f9"></i>NPC truth</span>
        <span><i class="legend-line" style="background:#52e09b"></i>Ego</span>
      </div>
      <div class="help">俯视图：上方为车辆前进方向；滚轮缩放；双击恢复视野</div>
    </section>
    <aside>
      <div class="summary">
        <div class="card"><span class="label">点云采样</span><b id="pointCount">0</b></div>
        <div class="card"><span class="label">检测目标</span><b id="detCount">0</b></div>
        <div class="card"><span class="label">旁车真值框</span><b id="truthCount">0</b></div>
        <div class="card"><span class="label">车辆</span><b id="carCount">0</b></div>
        <div class="card"><span class="label">行人 / 骑行者</span><b id="vrCount">0 / 0</b></div>
      </div>
      <table>
        <thead><tr><th>类别</th><th>置信度</th><th>距离</th><th>位置 x/y</th></tr></thead>
        <tbody id="detections"></tbody>
      </table>
    </aside>
  </main>
<script>
const canvas = document.getElementById("view"), ctx = canvas.getContext("2d");
const colors = {Car:"#ff5b5b", Pedestrian:"#ffd166", Cyclist:"#4dd6ff", Unknown:"#d58cff"};
let frame = {points:[], detections:[], ground_truth:[], truth_meta:{},
  global_path:[], local_path:[], planning:{},
  timestamp:0, sequence:0, stage:"waiting_lidar"}, range = 65, lastSequence = -1;

function resize() {
  const dpr = window.devicePixelRatio || 1, r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width*dpr));
  canvas.height = Math.max(1, Math.round(r.height*dpr));
  ctx.setTransform(dpr,0,0,dpr,0,0);
  draw();
}
function transform(x,y,w,h) {
  const scale = Math.min(w,h)/(2*range);
  return [w/2-y*scale, h/2-x*scale, scale];
}
function drawGrid(w,h) {
  ctx.fillStyle="#080d14"; ctx.fillRect(0,0,w,h);
  const step=10;
  ctx.lineWidth=1; ctx.strokeStyle="#182534"; ctx.fillStyle="#557087"; ctx.font="10px system-ui";
  for(let m=-Math.floor(range/step)*step;m<=range;m+=step) {
    const [px,py]=transform(m,0,w,h), [,py2]=transform(0,m,w,h);
    ctx.beginPath(); ctx.moveTo(0,py); ctx.lineTo(w,py); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,h); ctx.stroke();
    if(m!==0){ ctx.fillText(m+"m",w/2+4,py-3); ctx.fillText(m+"m",px+3,h/2-4); }
  }
  ctx.strokeStyle="#496178"; ctx.beginPath(); ctx.moveTo(w/2,0); ctx.lineTo(w/2,h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0,h/2); ctx.lineTo(w,h/2); ctx.stroke();
}
function boxCorners(d) {
  const c=Math.cos(d.yaw), s=Math.sin(d.yaw), hl=d.length/2, hw=d.width/2;
  return [[hl,hw],[hl,-hw],[-hl,-hw],[-hl,hw]].map(([a,b]) =>
    [d.x+a*c-b*s, d.y+a*s+b*c]);
}
function drawPath(points,color,width,dashed=false) {
  if(!points || points.length<2) return;
  const r=canvas.getBoundingClientRect(), w=r.width, h=r.height;
  ctx.save(); ctx.strokeStyle=color; ctx.lineWidth=width; ctx.lineJoin="round";
  ctx.lineCap="round"; ctx.setLineDash(dashed?[8,6]:[]);
  ctx.beginPath();
  let started=false;
  for(const p of points) {
    const [px,py]=transform(p[0],p[1],w,h);
    if(!Number.isFinite(px)||!Number.isFinite(py)) continue;
    if(!started){ctx.moveTo(px,py);started=true;}else ctx.lineTo(px,py);
  }
  if(started) ctx.stroke();
  ctx.restore();
}
function draw() {
  const r=canvas.getBoundingClientRect(), w=r.width, h=r.height; drawGrid(w,h);
  drawPath(frame.global_path,"rgba(57,213,255,.82)",2,true);
  for(const p of frame.points || []) {
    const [px,py]=transform(p[0],p[1],w,h);
    if(px<0||px>w||py<0||py>h) continue;
    const z=Math.max(-2,Math.min(2,p[2]||0)), t=(z+2)/4;
    ctx.fillStyle=`rgba(${Math.round(70+100*t)},${Math.round(145+90*t)},${Math.round(205+45*t)},.72)`;
    ctx.fillRect(px,py,1.35,1.35);
  }
  for(const d of (frame.detections || []).filter(d=>d.source!=="npc_truth")) {
    const corners=boxCorners(d).map(p=>transform(p[0],p[1],w,h));
    ctx.strokeStyle=colors[d.label]||colors.Unknown; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(corners[0][0],corners[0][1]);
    for(let i=1;i<corners.length;i++) ctx.lineTo(corners[i][0],corners[i][1]);
    ctx.closePath(); ctx.stroke();
    const [cx,cy,scale]=transform(d.x,d.y,w,h);
    ctx.beginPath(); ctx.moveTo(cx,cy);
    ctx.lineTo(cx-Math.sin(d.yaw)*Math.max(8,d.length*scale*.7),
               cy-Math.cos(d.yaw)*Math.max(8,d.length*scale*.7)); ctx.stroke();
    ctx.fillStyle=ctx.strokeStyle; ctx.font="11px system-ui";
    ctx.fillText(`${d.label} ${(d.score*100).toFixed(0)}%`,corners[0][0]+4,corners[0][1]-4);
  }
  for(const d of frame.ground_truth || []) {
    const corners=boxCorners(d).map(p=>transform(p[0],p[1],w,h));
    ctx.save(); ctx.strokeStyle="#ff4dff"; ctx.fillStyle="rgba(255,77,255,.12)";
    ctx.lineWidth=4; ctx.setLineDash([8,4]);
    ctx.beginPath(); ctx.moveTo(corners[0][0],corners[0][1]);
    for(let i=1;i<corners.length;i++) ctx.lineTo(corners[i][0],corners[i][1]);
    ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle="#ff7aff"; ctx.font="bold 12px system-ui";
    const suffix=d.dimensions_source==="simulator"?"":" (size estimated)";
    ctx.fillText(`GT ${d.role_name}${suffix}`,corners[0][0]+4,corners[0][1]-4);
    ctx.restore();
  }
  const localColor=(frame.planning||{}).emergency?"#ff5b5b":"#ffd166";
  drawPath(frame.local_path,localColor,3,false);
  const [ex,ey,scale]=transform(0,0,w,h);
  ctx.save(); ctx.translate(ex,ey); ctx.fillStyle="#52e09b"; ctx.strokeStyle="#d5fff0";
  ctx.fillRect(-1.0*scale,-2.3*scale,2.0*scale,4.6*scale);
  ctx.strokeRect(-1.0*scale,-2.3*scale,2.0*scale,4.6*scale); ctx.restore();
}
function updatePanel() {
  const ds=(frame.detections||[]).filter(d=>d.source!=="npc_truth"),
    counts={Car:0,Pedestrian:0,Cyclist:0};
  ds.forEach(d=>counts[d.label]=(counts[d.label]||0)+1);
  pointCount.textContent=(frame.points||[]).length; detCount.textContent=ds.length;
  truthCount.textContent=(frame.ground_truth||[]).length;
  carCount.textContent=counts.Car||0; vrCount.textContent=`${counts.Pedestrian||0} / ${counts.Cyclist||0}`;
  const planning=frame.planning||{}, gp=(frame.global_path||[]).length, lp=(frame.local_path||[]).length;
  const tm=frame.truth_meta||{};
  const sync=Number.isFinite(tm.sync_delta_ms)?` · GTΔ${tm.sync_delta_ms.toFixed(0)}ms`:"";
  planMeta.textContent=`Plan: ${planning.behavior||"--"} · ${Number(planning.target_speed||0).toFixed(1)} m/s · G${gp}/L${lp}${sync}`;
  detections.innerHTML=ds.slice().sort((a,b)=>a.distance-b.distance).map(d =>
    `<tr><td><span class="dot" style="background:${colors[d.label]||colors.Unknown}"></span>${d.label}</td>`+
    `<td>${(d.score*100).toFixed(1)}%</td><td>${d.distance.toFixed(1)}m</td>`+
    `<td>${d.x.toFixed(1)} / ${d.y.toFixed(1)}</td></tr>`).join("");
}
async function poll() {
  try {
    const res=await fetch("/api/frame",{cache:"no-store"});
    if(!res.ok) throw new Error(res.status);
    frame=await res.json(); const age=Date.now()/1000-frame.timestamp;
    const live=age<1.5, stage=frame.stage||"waiting_lidar";
    state.textContent=!live?"数据暂停":(stage==="ready_gt"?"GT 实时":(stage==="detecting"?"检测中":(stage==="ready"?"实时":(stage==="waiting_gt"?"等待 GT":"等待雷达"))));
    state.className="pill "+(live && stage!=="waiting_lidar" && stage!=="waiting_gt"?"live":"stale");
    meta.textContent=`帧 ${frame.sequence||0} · 延迟 ${Math.max(0,age*1000).toFixed(0)} ms`;
    if(frame.sequence!==lastSequence){ lastSequence=frame.sequence; updatePanel(); draw(); }
  } catch(e) { state.textContent="端口断开"; state.className="pill stale"; }
  setTimeout(poll,100);
}
canvas.addEventListener("wheel",e=>{e.preventDefault();range=Math.max(20,Math.min(140,range*(e.deltaY>0?1.12:.89)));draw();},{passive:false});
canvas.addEventListener("dblclick",()=>{range=65;draw();});
window.addEventListener("resize",resize); resize(); poll();
</script>
</body></html>"""


class PerceptionWebVisualizer:
    """Serve the latest detector frame through a small built-in HTTP server."""

    LABELS = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}

    def __init__(
        self,
        host="127.0.0.1",
        port=8765,
        max_points=7000,
        gt_only=True,
    ):
        self.host = str(host)
        self.port = int(port)
        self.max_points = max(100, int(max_points))
        self.gt_only = bool(gt_only)
        self._lock = threading.Lock()
        self._frame_json = json.dumps(
            {
                "timestamp": time.time(),
                "sequence": 0,
                "stage": "waiting_lidar",
                "points": [],
                "detections": [],
                "ground_truth": [],
                "truth_meta": {},
                "global_path": [],
                "local_path": [],
                "planning": {},
            }
        ).encode("utf-8")
        self._sequence = 0
        self._latest_stage = "waiting_lidar"
        self._latest_points = []
        self._latest_detections = []
        self._latest_ground_truth = []
        self._latest_truth_meta = {}
        self._latest_global_path = []
        self._latest_local_path = []
        self._latest_planning = {}
        self._server = None
        self._thread = None

    def _handler_class(self):
        visualizer = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status, content_type, body):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/":
                    self._send(200, "text/html; charset=utf-8", _PAGE.encode("utf-8"))
                elif path == "/api/frame":
                    with visualizer._lock:
                        body = visualizer._frame_json
                    self._send(200, "application/json; charset=utf-8", body)
                elif path == "/health":
                    self._send(200, "application/json", b'{"ok":true}')
                elif path == "/favicon.ico":
                    self._send(204, "image/x-icon", b"")
                else:
                    self._send(404, "text/plain; charset=utf-8", b"not found")

            def log_message(self, _format, *_args):
                return

        return Handler

    def start(self):
        if self._server is not None:
            return True
        try:
            self._server = ThreadingHTTPServer(
                (self.host, self.port), self._handler_class()
            )
        except OSError as exc:
            print(
                "[perception-web][WARN] cannot listen on "
                f"{self.host}:{self.port}: {exc}"
            )
            self._server = None
            return False
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="perception-web",
            daemon=True,
        )
        self._thread.start()
        if self.host in ("0.0.0.0", "::"):
            print(
                f"[perception-web] listening on {self.host}:{self.port}; "
                f"local http://127.0.0.1:{self.port}"
            )
        else:
            print(f"[perception-web] open http://{self.host}:{self.port}")
        print(
            "[perception-web] box_layer="
            + ("GT_ONLY" if self.gt_only else "DETECTOR_AND_GT")
        )
        return True

    def stop(self):
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _prepare_points(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] < 3:
            points = np.empty((0, 3), dtype=float)
        else:
            points = points[:, :3]
            points = points[np.all(np.isfinite(points), axis=1)]
        if points.shape[0] > self.max_points:
            indices = np.linspace(
                0, points.shape[0] - 1, self.max_points, dtype=np.int64
            )
            points = points[indices]
        return np.round(points, 3).tolist()

    def _store_frame(
        self,
        stage=None,
        points=None,
        detections=None,
        ground_truth=None,
        truth_meta=None,
        global_path=None,
        local_path=None,
        planning=None,
    ):
        with self._lock:
            if stage is not None:
                self._latest_stage = str(stage)
            if points is not None:
                self._latest_points = points
            if detections is not None:
                self._latest_detections = detections
            if ground_truth is not None:
                self._latest_ground_truth = ground_truth
            if truth_meta is not None:
                self._latest_truth_meta = truth_meta
            if global_path is not None:
                self._latest_global_path = global_path
            if local_path is not None:
                self._latest_local_path = local_path
            if planning is not None:
                self._latest_planning = planning
            self._sequence += 1
            payload = {
                "timestamp": time.time(),
                "sequence": self._sequence,
                "stage": self._latest_stage,
                "points": self._latest_points,
                "detections": self._latest_detections,
                "ground_truth": self._latest_ground_truth,
                "truth_meta": self._latest_truth_meta,
                "global_path": self._latest_global_path,
                "local_path": self._latest_local_path,
                "planning": self._latest_planning,
            }
            self._frame_json = json.dumps(
                payload, separators=(",", ":")
            ).encode("utf-8")

    def publish_points(self, points):
        """Publish lidar immediately, before detector inference completes."""
        self._store_frame("detecting", points=self._prepare_points(points))

    def publish_waiting(self):
        """Tell the page that no fresh lidar sample is currently available."""
        self._store_frame("waiting_lidar")

    def publish_ground_truth(
        self,
        npc_truth,
        lidar_timestamp=None,
        sync_delta_s=None,
    ):
        """Overlay decoded simulator NPC boxes in the lidar/ego view.

        Some simulator vehicle assets publish placeholder 1 x 1 x 1
        dimensions. Those boxes use a display-only fallback and are marked
        as estimated; the decoded JSONL continues to preserve the raw values.
        """
        if not isinstance(npc_truth, dict):
            with self._lock:
                detector_boxes = (
                    []
                    if self.gt_only
                    else [
                        item
                        for item in self._latest_detections
                        if item.get("source") != "npc_truth"
                    ]
                )
            self._store_frame(
                stage="waiting_gt",
                ground_truth=[],
                truth_meta={},
                detections=detector_boxes,
            )
            return
        model_size_fallbacks = {
            "Veh_Lynkco": (4.8, 2.0, 1.7),
            "Veh_GeometryC2": (4.8, 2.0, 1.7),
        }
        boxes = []
        for role in npc_truth.get("roles", []):
            position = role.get("position", {})
            dimensions = role.get("dimensions", {})
            try:
                x = float(position["x"])
                y = float(position["y"])
                z = float(position["z"])
                yaw = float(role["yaw"])
                length = float(dimensions["length"])
                width = float(dimensions["width"])
                height = float(dimensions["height"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(
                math.isfinite(value)
                for value in (x, y, z, yaw, length, width, height)
            ):
                continue
            dimensions_source = "simulator"
            if not role.get("dimensions_valid", False):
                length, width, height = model_size_fallbacks.get(
                    str(role.get("model_name", "")),
                    (4.7, 2.0, 1.7),
                )
                dimensions_source = "display_fallback"
            boxes.append(
                {
                    "label": str(role.get("class_name", "Unknown")),
                    "role_name": str(role.get("role_name", "?")),
                    "model_name": str(role.get("model_name", "")),
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "z": round(z, 3),
                    "length": round(max(0.05, length), 3),
                    "width": round(max(0.05, width), 3),
                    "height": round(max(0.05, height), 3),
                    "yaw": round(yaw, 4),
                    "distance": round(math.hypot(x, y), 3),
                    "dimensions_source": dimensions_source,
                }
            )
        truth_timestamp = npc_truth.get("timestamp_s")
        if sync_delta_s is None:
            try:
                sync_delta_s = (
                    float(truth_timestamp) - float(lidar_timestamp)
                )
            except (TypeError, ValueError):
                sync_delta_s = None
        meta = {
            "decoder": npc_truth.get("decoder"),
            "timestamp_s": truth_timestamp,
            "lidar_timestamp_s": lidar_timestamp,
            "sync_delta_ms": (
                round(float(sync_delta_s) * 1000.0, 3)
                if sync_delta_s is not None
                and math.isfinite(float(sync_delta_s))
                else None
            ),
            "coordinate_frame": (
                npc_truth.get("roles", [{}])[0].get(
                    "coordinate_frame"
                )
                if npc_truth.get("roles")
                else None
            ),
        }
        # Keep a copy in ``detections`` so a browser tab that still has the
        # pre-ground-truth JavaScript can draw the boxes immediately. The new
        # page filters these compatibility entries and draws the magenta,
        # dashed ``ground_truth`` layer instead.
        with self._lock:
            detector_boxes = (
                []
                if self.gt_only
                else [
                    item
                    for item in self._latest_detections
                    if item.get("source") != "npc_truth"
                ]
            )
        compatibility_boxes = []
        for box in boxes:
            item = dict(box)
            item.update(
                {
                    "label": "Car",
                    "score": 1.0,
                    "source": "npc_truth",
                }
            )
            compatibility_boxes.append(item)
        self._store_frame(
            stage="ready_gt",
            ground_truth=boxes,
            truth_meta=meta,
            detections=detector_boxes + compatibility_boxes,
        )

    @staticmethod
    def _world_path_to_ego(xs, ys, ego, max_points=1200):
        try:
            xs = np.asarray(xs, dtype=float).reshape(-1)
            ys = np.asarray(ys, dtype=float).reshape(-1)
            ego_x = float(getattr(ego, "x"))
            ego_y = float(getattr(ego, "y"))
            ego_yaw = float(getattr(ego, "theta"))
        except (TypeError, ValueError, AttributeError):
            return []
        count = min(xs.size, ys.size)
        if count < 2:
            return []
        xs = xs[:count]
        ys = ys[:count]
        valid = np.isfinite(xs) & np.isfinite(ys)
        dx = xs[valid] - ego_x
        dy = ys[valid] - ego_y
        cos_yaw = math.cos(ego_yaw)
        sin_yaw = math.sin(ego_yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        visible = (
            (local_x >= -30.0)
            & (local_x <= 180.0)
            & (np.abs(local_y) <= 150.0)
        )
        path = np.column_stack((local_x[visible], local_y[visible]))
        if path.shape[0] > max_points:
            indices = np.linspace(
                0, path.shape[0] - 1, max_points, dtype=np.int64
            )
            path = path[indices]
        return np.round(path, 3).tolist()

    def publish_paths(
        self,
        ego,
        global_path,
        local_trajectory,
        behavior="",
        target_speed=0.0,
        emergency=False,
    ):
        """Overlay map-frame global/local paths in the lidar ego frame."""
        if ego is None:
            self._store_frame(
                global_path=[], local_path=[], planning={}
            )
            return
        global_points = []
        if isinstance(global_path, dict):
            global_points = self._world_path_to_ego(
                global_path.get("x", []),
                global_path.get("y", []),
                ego,
            )
        local_points = []
        if local_trajectory is not None:
            local_points = self._world_path_to_ego(
                getattr(local_trajectory, "x", []),
                getattr(local_trajectory, "y", []),
                ego,
            )
        planning = {
            "behavior": str(behavior or ""),
            "target_speed": round(max(0.0, float(target_speed)), 3),
            "emergency": bool(emergency),
        }
        self._store_frame(
            global_path=global_points,
            local_path=local_points,
            planning=planning,
        )

    def publish(self, points, boxes, labels, scores):
        prepared_points = self._prepare_points(points)

        boxes = np.asarray(boxes, dtype=float)
        labels = np.asarray(labels).reshape(-1)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        detections = []
        count = min(
            boxes.shape[0] if boxes.ndim == 2 else 0,
            labels.size,
            scores.size,
        )
        for index in range(count):
            box = boxes[index]
            if box.size < 7 or not np.all(np.isfinite(box[:7])):
                continue
            label_id = int(labels[index])
            x, y = float(box[0]), float(box[1])
            detections.append(
                {
                    "label": self.LABELS.get(label_id, f"Class {label_id}"),
                    "score": round(max(0.0, min(1.0, float(scores[index]))), 4),
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "z": round(float(box[2]), 3),
                    "length": round(max(0.05, float(box[3])), 3),
                    "width": round(max(0.05, float(box[4])), 3),
                    "height": round(max(0.05, float(box[5])), 3),
                    "yaw": round(float(box[6]), 4),
                    "distance": round(math.hypot(x, y), 3),
                    "source": "detector",
                }
            )

        self._store_frame(
            "ready",
            points=prepared_points,
            detections=[] if self.gt_only else detections,
        )
