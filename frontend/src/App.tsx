import { type MouseEvent, type ReactNode, useEffect, useState } from 'react'

const THERMAL_W = 640
const THERMAL_H = 480
const RGB_W = 640
const RGB_H = 480

interface MinMax {
  min_raw: number
  max_raw: number
  min_x: number
  min_y: number
  max_x: number
  max_y: number
}

interface Probe {
  x: number
  y: number
  raw: number
}

interface BBox { x: number; y: number; w: number; h: number }
interface Pt { x: number; y: number }

interface FaceRegions {
  bbox: BBox
  nose_tip: Pt
  forehead_center: Pt
  cheek_left: Pt
  cheek_right: Pt
  forehead_box: BBox
  cheek_left_box: BBox
  cheek_right_box: BBox
}

interface FaceData {
  timestamp: number
  faces: FaceRegions[]
}

interface PosePt { x: number; y: number; visible: boolean }

interface PoseRegions {
  shoulder_l: PosePt
  shoulder_r: PosePt
  elbow_l: PosePt
  elbow_r: PosePt
  wrist_l: PosePt
  wrist_r: PosePt
  hip_l: PosePt
  hip_r: PosePt
  hand_l_box: BBox | null
  hand_r_box: BBox | null
  torso_box: BBox | null
}

interface PoseData {
  timestamp: number
  poses: PoseRegions[]
}

interface CalPair {
  rgb: { x: number; y: number }
  thermal: { x: number; y: number }
}

interface CalStatus {
  pairs: CalPair[]
  pair_count: number
  homography_ready: boolean
  homography: number[][] | null
}

interface Status {
  rgb_connected: boolean
  thermal_connected: boolean
  cellplus_connected: boolean
  face_connected: boolean
  pose_connected: boolean
  colormap_idx: number
  colormap_presets: number[]
}

const COLORMAP_LABELS: Record<number, string> = {
  0: 'Gray',
  6: 'Preset A',
  13: 'Preset B',
  16: 'Preset C',
  17: 'Preset D',
}

export default function App() {
  const [status, setStatus] = useState<Status | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [minmax, setMinmax] = useState<MinMax | null>(null)
  const [probe, setProbe] = useState<Probe | null>(null)
  const [face, setFace] = useState<FaceData | null>(null)
  const [pose, setPose] = useState<PoseData | null>(null)
  const [cal, setCal] = useState<CalStatus | null>(null)
  const [calMode, setCalMode] = useState(false)
  const [calBusy, setCalBusy] = useState(false)
  const [calError, setCalError] = useState<string | null>(null)
  const [calPending, setCalPending] = useState<{ x: number; y: number } | null>(null)

  async function refreshStatus() {
    try {
      const r = await fetch('/api/status')
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      setStatus(await r.json())
      setError(null)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  useEffect(() => {
    refreshStatus()
    const t = setInterval(refreshStatus, 5000)
    return () => clearInterval(t)
  }, [])

  useEffect(() => {
    if (!status?.cellplus_connected) return
    let stopped = false
    async function poll() {
      try {
        const r = await fetch('/api/temperature/minmax')
        if (r.ok && !stopped) setMinmax(await r.json())
      } catch {}
    }
    poll()
    const t = setInterval(poll, 1000)
    return () => {
      stopped = true
      clearInterval(t)
    }
  }, [status?.cellplus_connected])

  useEffect(() => {
    if (!status?.face_connected) return
    let stopped = false
    async function poll() {
      try {
        const r = await fetch('/api/face')
        if (r.ok && !stopped) setFace(await r.json())
      } catch {}
    }
    poll()
    const t = setInterval(poll, 200)
    return () => {
      stopped = true
      clearInterval(t)
    }
  }, [status?.face_connected])

  useEffect(() => {
    if (!status?.pose_connected) return
    let stopped = false
    async function poll() {
      try {
        const r = await fetch('/api/pose')
        if (r.ok && !stopped) setPose(await r.json())
      } catch {}
    }
    poll()
    const t = setInterval(poll, 250)
    return () => {
      stopped = true
      clearInterval(t)
    }
  }, [status?.pose_connected])

  async function refreshCal() {
    try {
      const r = await fetch('/api/calibration')
      if (r.ok) setCal(await r.json())
    } catch {}
  }

  useEffect(() => {
    refreshCal()
  }, [])

  async function calReset() {
    setCalBusy(true)
    setCalError(null)
    setCalPending(null)
    try {
      const r = await fetch('/api/calibration/reset', { method: 'POST' })
      if (r.ok) setCal(await r.json())
    } finally {
      setCalBusy(false)
    }
  }

  function calRgbClick(x: number, y: number) {
    if (!calMode) return
    setCalError(null)
    setCalPending({ x, y })
  }

  async function calThermalClick(x: number, y: number) {
    if (!calMode || !calPending) return
    setCalBusy(true)
    setCalError(null)
    try {
      const r = await fetch('/api/calibration/pair', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rgb_x: calPending.x,
          rgb_y: calPending.y,
          thermal_x: x,
          thermal_y: y,
        }),
      })
      const data = await r.json()
      if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`)
      setCal(data)
      setCalPending(null)
    } catch (e) {
      setCalError((e as Error).message)
    } finally {
      setCalBusy(false)
    }
  }

  async function probePixel(nx: number, ny: number) {
    try {
      const r = await fetch('/api/temperature/pixel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x: nx, y: ny }),
      })
      if (r.ok) setProbe(await r.json())
    } catch (e) {
      setError((e as Error).message)
    }
  }

  async function setColormap(idx: number) {
    setBusy(true)
    try {
      const r = await fetch('/api/colormap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: idx }),
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const data = await r.json()
      setStatus((s) => (s ? { ...s, colormap_idx: data.current } : s))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>HeatSight</h1>
        <p className="sub">RGB + 열화상 듀얼 카메라 라이브 뷰</p>
      </header>

      <section className="status">
        <Dot ok={status?.rgb_connected} label="RGB" />
        <Dot ok={status?.thermal_connected} label="Thermal" />
        <Dot ok={status?.cellplus_connected} label="Cellplus Control" />
        <Dot ok={status?.face_connected && (face?.faces.length ?? 0) > 0} label="Face" />
        <Dot ok={status?.pose_connected && (pose?.poses.length ?? 0) > 0} label="Pose" />
        <Dot ok={cal?.homography_ready} label="Calibrated" />
        <button
          className="cal-toggle"
          onClick={() => setCalMode((v) => !v)}
        >
          {calMode ? 'Close calibration' : cal?.homography_ready ? 'Recalibrate' : 'Start calibration'}
        </button>
        {error && <span className="error">⚠ {error}</span>}
      </section>

      {calMode && (
        <section className="cal-panel">
          <div className="cal-progress">
            <div className="cal-dots">
              {[0, 1, 2, 3].map((i) => (
                <span
                  key={i}
                  className={`cal-dot ${i < (cal?.pair_count ?? 0) ? 'filled' : ''}`}
                />
              ))}
            </div>
            <div className="cal-status">
              {cal?.homography_ready ? (
                <span className="cal-done">✓ Calibration ready — 추가 페어로 정밀도 향상 가능</span>
              ) : calPending ? (
                <span className="cal-hint">
                  다음: <strong>Thermal 패널</strong>에서 같은 실세계 지점 클릭
                </span>
              ) : (
                <span className="cal-hint">
                  Pair {(cal?.pair_count ?? 0) + 1} / 4 — <strong>RGB 패널</strong>에서 한 점 클릭
                </span>
              )}
            </div>
          </div>
          <div className="cal-actions">
            <button className="cal-secondary" onClick={calReset} disabled={calBusy}>
              Reset
            </button>
          </div>
          {calError && <div className="cal-error">⚠ {calError}</div>}
          <p className="cal-tip">
            평면 위 점(노트북/책상 모서리, 종이 꼭짓점) 4개가 가장 정확합니다.
            양쪽에서 또렷이 보여야 하니 따뜻한 머그컵·핫팩을 잠깐 올려두고 클릭하면 thermal에서도 보입니다.
          </p>
        </section>
      )}

      <section className="cameras">
        <RGBPanel
          src="/api/video/rgb"
          face={face?.faces[0] ?? null}
          pose={pose?.poses[0] ?? null}
          calPairs={cal?.pairs ?? []}
          calMode={calMode}
          calPending={calPending}
          onCalibrationClick={calRgbClick}
        />
        <ThermalPanel
          src="/api/video/thermal"
          minmax={minmax}
          probe={probe}
          calPairs={cal?.pairs ?? []}
          calMode={calMode}
          calPending={calPending}
          onProbe={probePixel}
          onCalibrationClick={calThermalClick}
          colormapSelect={
            <select
              className="cmap-select"
              disabled={busy || !status?.cellplus_connected}
              value={status?.colormap_idx ?? 0}
              onChange={(e) => setColormap(Number(e.target.value))}
              aria-label="Thermal Colormap"
            >
              {status?.colormap_presets.map((idx) => (
                <option key={idx} value={idx}>
                  #{String(idx).padStart(2, '0')} · {COLORMAP_LABELS[idx] ?? `Index ${idx}`}
                </option>
              ))}
            </select>
          }
        />
      </section>

      {(minmax || probe) && (
        <section className="temp-readout">
          {minmax && (
            <>
              <Readout label="Min" value={minmax.min_raw} hint={`(${minmax.min_x}, ${minmax.min_y})`} tone="cool" />
              <Readout label="Max" value={minmax.max_raw} hint={`(${minmax.max_x}, ${minmax.max_y})`} tone="hot" />
            </>
          )}
          {probe && (
            <Readout
              label="Probe"
              value={probe.raw}
              hint={`(${probe.x}, ${probe.y})${
                minmax
                  ? ` · ${Math.round(((probe.raw - minmax.min_raw) /
                      Math.max(1, minmax.max_raw - minmax.min_raw)) * 100)}%`
                  : ''
              }`}
              tone="probe"
            />
          )}
        </section>
      )}
    </div>
  )
}

function Dot({ ok, label }: { ok?: boolean; label: string }) {
  return (
    <span className="dot-wrap">
      <span className={`dot ${ok ? 'ok' : 'off'}`} />
      {label}
    </span>
  )
}

function Panel({
  title,
  subtitle,
  src,
  headerExtra,
}: {
  title: string
  subtitle: string
  src: string
  headerExtra?: ReactNode
}) {
  return (
    <div className="panel">
      <div className="panel-title">
        <div className="panel-title-text">
          <span className="panel-name">{title}</span>
          <span className="panel-sub">{subtitle}</span>
        </div>
        {headerExtra && <div className="panel-title-extra">{headerExtra}</div>}
      </div>
      <img src={src} alt={title} />
    </div>
  )
}

function RGBPanel({
  src,
  face,
  pose,
  calPairs,
  calMode,
  calPending,
  onCalibrationClick,
}: {
  src: string
  face: FaceRegions | null
  pose: PoseRegions | null
  calPairs: CalPair[]
  calMode: boolean
  calPending: { x: number; y: number } | null
  onCalibrationClick: (x: number, y: number) => void
}) {
  function handleClick(e: MouseEvent<HTMLImageElement>) {
    if (!calMode) return
    const rect = e.currentTarget.getBoundingClientRect()
    const px = (e.clientX - rect.left) / rect.width
    const py = (e.clientY - rect.top) / rect.height
    const x = Math.max(0, Math.min(RGB_W - 1, Math.round(px * RGB_W)))
    const y = Math.max(0, Math.min(RGB_H - 1, Math.round(py * RGB_H)))
    onCalibrationClick(x, y)
  }

  const tags: string[] = []
  if (face) tags.push('face')
  if (pose) tags.push('pose')
  const subtitle = calMode
    ? 'click to add a calibration point'
    : `MacBook FaceTime HD  ·  ${tags.length ? tags.join(' + ') : 'detecting…'}`

  return (
    <div className={`panel ${calMode ? 'cal-active' : ''}`}>
      <div className="panel-title">
        <div className="panel-title-text">
          <span className="panel-name">RGB</span>
          <span className="panel-sub">{subtitle}</span>
        </div>
      </div>
      <div className="rgb-canvas">
        <img src={src} alt="RGB" onClick={handleClick} />
        <svg
          className="rgb-overlay"
          viewBox={`0 0 ${RGB_W} ${RGB_H}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {!calMode && pose && <PoseOverlay pose={pose} />}
          {!calMode && face && <FaceOverlay face={face} />}
          {calPairs.map((p, i) => (
            <CalMarker key={i} x={p.rgb.x} y={p.rgb.y} index={i + 1} />
          ))}
          {calMode && calPending && (
            <CalPending x={calPending.x} y={calPending.y} index={calPairs.length + 1} />
          )}
        </svg>
      </div>
    </div>
  )
}

function CalMarker({ x, y, index }: { x: number; y: number; index: number }) {
  return (
    <g>
      <circle cx={x} cy={y} r={10} fill="rgba(255, 217, 61, 0.2)" stroke="#ffd93d" strokeWidth={2} />
      <text x={x} y={y + 4} textAnchor="middle" fill="#ffd93d" fontSize={11} fontWeight={700}>
        {index}
      </text>
    </g>
  )
}

function CalPending({ x, y, index }: { x: number; y: number; index: number }) {
  return (
    <g>
      <circle
        cx={x}
        cy={y}
        r={12}
        fill="none"
        stroke="#ffd93d"
        strokeWidth={2}
        strokeDasharray="4 3"
      />
      <text x={x} y={y + 4} textAnchor="middle" fill="#ffd93d" fontSize={11} fontWeight={700}>
        {index}?
      </text>
    </g>
  )
}

function PoseOverlay({ pose }: { pose: PoseRegions }) {
  const bone = (a: PosePt, b: PosePt, color: string) =>
    a.visible && b.visible ? (
      <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={color} strokeWidth={3} strokeLinecap="round" />
    ) : null

  const joint = (p: PosePt, color: string) =>
    p.visible ? <circle cx={p.x} cy={p.y} r={5} fill={color} /> : null

  return (
    <g>
      {/* 몸통 박스 */}
      {pose.torso_box && (
        <rect
          x={pose.torso_box.x}
          y={pose.torso_box.y}
          width={pose.torso_box.w}
          height={pose.torso_box.h}
          fill="rgba(168, 142, 255, 0.06)"
          stroke="#a88eff"
          strokeWidth={1.5}
          strokeDasharray="4 3"
        />
      )}
      {/* 골격 */}
      {bone(pose.shoulder_l, pose.shoulder_r, '#a88eff')}
      {bone(pose.shoulder_l, pose.elbow_l, '#a88eff')}
      {bone(pose.elbow_l, pose.wrist_l, '#a88eff')}
      {bone(pose.shoulder_r, pose.elbow_r, '#a88eff')}
      {bone(pose.elbow_r, pose.wrist_r, '#a88eff')}
      {bone(pose.shoulder_l, pose.hip_l, '#a88eff')}
      {bone(pose.shoulder_r, pose.hip_r, '#a88eff')}
      {bone(pose.hip_l, pose.hip_r, '#a88eff')}
      {/* 관절 점 */}
      {joint(pose.shoulder_l, '#a88eff')}
      {joint(pose.shoulder_r, '#a88eff')}
      {joint(pose.elbow_l, '#a88eff')}
      {joint(pose.elbow_r, '#a88eff')}
      {joint(pose.wrist_l, '#a88eff')}
      {joint(pose.wrist_r, '#a88eff')}
      {/* 손 박스 */}
      {pose.hand_l_box && (
        <RegionBox box={pose.hand_l_box} color="#58a6ff" label="hand L" />
      )}
      {pose.hand_r_box && (
        <RegionBox box={pose.hand_r_box} color="#58a6ff" label="hand R" />
      )}
    </g>
  )
}

function FaceOverlay({ face }: { face: FaceRegions }) {
  return (
    <g>
      {/* 얼굴 전체 bbox */}
      <rect
        x={face.bbox.x}
        y={face.bbox.y}
        width={face.bbox.w}
        height={face.bbox.h}
        fill="none"
        stroke="#3fb950"
        strokeWidth={2}
      />
      {/* 핵심 영역들 */}
      <RegionBox box={face.forehead_box}    color="#ffd166" label="forehead" />
      <RegionBox box={face.cheek_left_box}  color="#ff8b3d" label="cheek L" />
      <RegionBox box={face.cheek_right_box} color="#ff8b3d" label="cheek R" />
      {/* 코끝 점 */}
      <circle cx={face.nose_tip.x} cy={face.nose_tip.y} r={4} fill="#58a6ff" />
      <text x={face.nose_tip.x + 8} y={face.nose_tip.y + 4} fill="#58a6ff" fontSize={11}>
        nose
      </text>
    </g>
  )
}

function RegionBox({ box, color, label }: { box: BBox; color: string; label: string }) {
  return (
    <g>
      <rect
        x={box.x}
        y={box.y}
        width={box.w}
        height={box.h}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeDasharray="3 2"
      />
      <text x={box.x} y={box.y - 3} fill={color} fontSize={10} fontWeight={600}>
        {label}
      </text>
    </g>
  )
}

function ThermalPanel({
  src,
  minmax,
  probe,
  calPairs,
  calMode,
  calPending,
  onProbe,
  onCalibrationClick,
  colormapSelect,
}: {
  src: string
  minmax: MinMax | null
  probe: Probe | null
  calPairs: CalPair[]
  calMode: boolean
  calPending: { x: number; y: number } | null
  onProbe: (x: number, y: number) => void
  onCalibrationClick: (x: number, y: number) => void
  colormapSelect: ReactNode
}) {
  function handleClick(e: MouseEvent<HTMLImageElement>) {
    const target = e.currentTarget
    const rect = target.getBoundingClientRect()
    const px = (e.clientX - rect.left) / rect.width
    const py = (e.clientY - rect.top) / rect.height
    const x = Math.max(0, Math.min(THERMAL_W - 1, Math.round(px * THERMAL_W)))
    const y = Math.max(0, Math.min(THERMAL_H - 1, Math.round(py * THERMAL_H)))
    if (calMode) {
      onCalibrationClick(x, y)
    } else {
      onProbe(x, y)
    }
  }

  const subtitle = calMode
    ? calPending
      ? 'click to pair with the RGB point'
      : 'pick the RGB point first'
    : 'Cellplus  ·  click to probe'

  return (
    <div className={`panel ${calMode ? 'cal-active' : ''}`}>
      <div className="panel-title">
        <div className="panel-title-text">
          <span className="panel-name">Thermal</span>
          <span className="panel-sub">{subtitle}</span>
        </div>
        <div className="panel-title-extra">{colormapSelect}</div>
      </div>
      <div className="thermal-canvas">
        <img src={src} alt="Thermal" onClick={handleClick} />
        <svg
          className="thermal-overlay"
          viewBox={`0 0 ${THERMAL_W} ${THERMAL_H}`}
          preserveAspectRatio="xMidYMid meet"
        >
          {!calMode && minmax && <Marker x={minmax.min_x} y={minmax.min_y} color="#58a6ff" label="min" />}
          {!calMode && minmax && <Marker x={minmax.max_x} y={minmax.max_y} color="#ff6a3d" label="max" />}
          {!calMode && probe && <Marker x={probe.x} y={probe.y} color="#e6edf3" label="probe" />}
          {calPairs.map((p, i) => (
            <CalMarker key={i} x={p.thermal.x} y={p.thermal.y} index={i + 1} />
          ))}
        </svg>
      </div>
    </div>
  )
}

function Marker({ x, y, color, label }: { x: number; y: number; color: string; label: string }) {
  return (
    <g>
      <circle cx={x} cy={y} r={14} fill="none" stroke={color} strokeWidth={2} />
      <line x1={x - 22} y1={y} x2={x - 6} y2={y} stroke={color} strokeWidth={2} />
      <line x1={x + 6} y1={y} x2={x + 22} y2={y} stroke={color} strokeWidth={2} />
      <line x1={x} y1={y - 22} x2={x} y2={y - 6} stroke={color} strokeWidth={2} />
      <line x1={x} y1={y + 6} x2={x} y2={y + 22} stroke={color} strokeWidth={2} />
      <text x={x + 18} y={y - 14} fill={color} fontSize={14} fontWeight={600}>
        {label}
      </text>
    </g>
  )
}

function Readout({
  label,
  value,
  hint,
  tone,
}: {
  label: string
  value: number
  hint?: string
  tone: 'cool' | 'hot' | 'probe'
}) {
  return (
    <div className={`readout readout-${tone}`}>
      <div className="readout-label">{label}</div>
      <div className="readout-value">{value}</div>
      {hint && <div className="readout-hint">{hint}</div>}
    </div>
  )
}
