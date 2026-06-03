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
  min_celsius?: number | null
  max_celsius?: number | null
}

interface Probe {
  x: number
  y: number
  raw: number
  celsius?: number | null
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
  quad: Pt[]
  forehead_quad: Pt[]
  cheek_left_quad: Pt[]
  cheek_right_quad: Pt[]
  roll_deg: number
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

interface BehaviorData {
  timestamp: number
  arms_crossed: number
  hunched: number
  hands_up: number
  fanning: number
  touching_face: number
  comfort: 'hot' | 'cold' | 'neutral'
  clip_scores?: Record<string, number>
  vlm_comfort?: 'hot' | 'cold' | 'neutral' | null
  vlm_answer?: string | null
  fused_comfort?: 'hot' | 'cold' | 'neutral'
}

interface Status {
  rgb_connected: boolean
  thermal_connected: boolean
  cellplus_connected: boolean
  face_connected: boolean
  face_thermal_connected: boolean
  pose_connected: boolean
  colormap_idx: number
  colormap_presets: number[]
}

interface HvacRecommendation {
  occupants: number
  occupant_details: {
    id: string
    index: number
    label: string
    comfort: 'hot' | 'cold' | 'neutral' | 'unknown'
    comfort_score?: number | null
    skin_temperature_c?: number | null
    rgb_bbox?: BBox | null
    thermal_bbox?: BBox | null
    sample_count: number
  }[]
  comfort: 'hot' | 'cold' | 'neutral'
  comfort_score: number
  skin_temperature_c?: number | null
  skin_temperatures_c: number[]
  control: {
    mode: 'cooling' | 'heating' | 'eco' | 'standby'
    mode_label: string
    target_setpoint_c?: number | null
    target_delta_c: number
    fan_percent: number
    reason: string
  }
  energy: {
    estimated_saving_percent: number
    strategy: string
  }
  signals: {
    thermal_score?: number | null
    behavior_score: number
    behavior_comfort?: 'hot' | 'cold' | 'neutral' | null
    calibrated: boolean
    thermal_error?: string | null
    data_quality: string
  }
  camera: {
    rgb_connected: boolean
    thermal_connected: boolean
    cellplus_connected: boolean
  }
}

interface FaceTemp {
  index: number
  rgb_bbox: BBox
  thermal_bbox: BBox
  thermal_corners: [number, number][]
  mean_raw: number
  mean_celsius: number | null
  sample_count: number
}

interface FaceTempData {
  timestamp: number
  grid: number
  faces: FaceTemp[]
}

function fmtCelsius(c: number | null | undefined): string | null {
  return c == null ? null : `${c.toFixed(1)} °C`
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
  const [faceThermal, setFaceThermal] = useState<FaceData | null>(null)
  const [faceTemps, setFaceTemps] = useState<FaceTempData | null>(null)
  const [pose, setPose] = useState<PoseData | null>(null)
  const [behavior, setBehavior] = useState<BehaviorData | null>(null)
  const [hvac, setHvac] = useState<HvacRecommendation | null>(null)
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
    if (!status?.face_thermal_connected) return
    let stopped = false
    async function poll() {
      try {
        const r = await fetch('/api/face/thermal')
        if (r.ok && !stopped) setFaceThermal(await r.json())
      } catch {}
    }
    poll()
    const t = setInterval(poll, 250)
    return () => {
      stopped = true
      clearInterval(t)
    }
  }, [status?.face_thermal_connected])

  useEffect(() => {
    if (!status?.face_connected || !status?.cellplus_connected || !cal?.homography_ready) return
    let stopped = false
    let inflight = false
    async function poll() {
      if (inflight) return
      inflight = true
      try {
        const r = await fetch('/api/temperature/faces')
        if (r.ok && !stopped) setFaceTemps(await r.json())
      } catch {
      } finally {
        inflight = false
      }
    }
    poll()
    const t = setInterval(poll, 1500)
    return () => {
      stopped = true
      clearInterval(t)
    }
  }, [status?.face_connected, status?.cellplus_connected, cal?.homography_ready])

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

  useEffect(() => {
    let stopped = false
    async function poll() {
      try {
        const r = await fetch('/api/behavior')
        if (r.ok && !stopped) setBehavior(await r.json())
      } catch {}
    }
    poll()
    const t = setInterval(poll, 500)
    return () => {
      stopped = true
      clearInterval(t)
    }
  }, [])

  useEffect(() => {
    let stopped = false
    async function poll() {
      try {
        const r = await fetch('/api/hvac/recommendation')
        if (r.ok && !stopped) setHvac(await r.json())
      } catch {}
    }
    poll()
    const t = setInterval(poll, 2500)
    return () => {
      stopped = true
      clearInterval(t)
    }
  }, [])

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
        <p className="sub">재실자 체온 인지 기반 실내 지능형 냉난방 제어 시스템</p>
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

      {hvac && <HvacSection hvac={hvac} />}

      <section className="cameras">
        <RGBPanel
          src="/api/video/rgb"
          faces={face?.faces ?? []}
          faceTemps={faceTemps?.faces ?? []}
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
          thermalFaces={faceThermal?.faces ?? []}
          faceTemps={faceTemps?.faces ?? []}
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

      {behavior && <BehaviorSection behavior={behavior} />}

      {(minmax || probe) && (
        <section className="temp-readout">
          {minmax && (
            <>
              <Readout
                label="Min"
                value={fmtCelsius(minmax.min_celsius) ?? `raw ${minmax.min_raw}`}
                hint={`(${minmax.min_x}, ${minmax.min_y})`}
                tone="cool"
              />
              <Readout
                label="Max"
                value={fmtCelsius(minmax.max_celsius) ?? `raw ${minmax.max_raw}`}
                hint={`(${minmax.max_x}, ${minmax.max_y})`}
                tone="hot"
              />
            </>
          )}
          {probe && (
            <Readout
              label="Probe"
              value={fmtCelsius(probe.celsius) ?? `raw ${probe.raw}`}
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
  faces,
  faceTemps,
  pose,
  calPairs,
  calMode,
  calPending,
  onCalibrationClick,
}: {
  src: string
  faces: FaceRegions[]
  faceTemps: FaceTemp[]
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
  if (faces.length > 0) tags.push(`${faces.length} face${faces.length > 1 ? 's' : ''}`)
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
          {!calMode &&
            faces.map((f, i) => (
              <FaceOverlay
                key={i}
                face={f}
                celsius={faceTemps[i]?.mean_celsius ?? null}
              />
            ))}
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

function HvacSection({ hvac }: { hvac: HvacRecommendation }) {
  const comfortInfo = comfortMeta(hvac.comfort)
  const scorePct = Math.round(((hvac.comfort_score + 1) / 2) * 100)
  const delta = hvac.control.target_delta_c
  const deltaLabel =
    delta === 0
      ? '유지'
      : `${delta > 0 ? '+' : ''}${delta.toFixed(1)} °C`
  const setpointLabel =
    hvac.control.target_setpoint_c == null
      ? '유지'
      : `${hvac.control.target_setpoint_c.toFixed(1)} °C`
  const skinLabel = fmtCelsius(hvac.skin_temperature_c) ?? '측정 대기'
  const thermalState = hvac.signals.thermal_score == null
    ? hvac.signals.calibrated
      ? '피부온도 대기'
      : '보정 대기'
    : `${Math.round(hvac.signals.thermal_score * 100)}`

  return (
    <section className={`hvac hvac-mode-${hvac.control.mode}`}>
      <div className="hvac-primary">
        <div className="hvac-eyebrow">HVAC control</div>
        <div className="hvac-mode">{hvac.control.mode_label}</div>
        <div className="hvac-reason">{hvac.control.reason}</div>
        <div className="comfort-meter" aria-label="comfort score">
          <span>추움</span>
          <div className="comfort-track">
            <div className="comfort-pin" style={{ left: `${scorePct}%` }} />
          </div>
          <span>더움</span>
        </div>
      </div>
      <div className="hvac-metrics">
        <Metric label="재실자" value={`${hvac.occupants}명`} />
        <Metric label="피부 평균" value={skinLabel} />
        <Metric label="쾌적도" value={comfortInfo.label} tone={comfortInfo.cls} />
        <Metric label="설정 온도" value={setpointLabel} hint={deltaLabel} />
        <Metric label="팬 출력" value={`${hvac.control.fan_percent}%`} />
        <Metric
          label="예상 절감"
          value={`${hvac.energy.estimated_saving_percent}%`}
          hint={hvac.energy.strategy}
        />
      </div>
      <div className="hvac-signals">
        <span>thermal {thermalState}</span>
        <span>behavior {Math.round(hvac.signals.behavior_score * 100)}</span>
        <span>{hvac.signals.data_quality}</span>
      </div>
      <div className="occupants">
        {hvac.occupant_details.length > 0 ? (
          hvac.occupant_details.map((person) => (
            <OccupantCard key={person.id} person={person} />
          ))
        ) : (
          <div className="occupant-empty">재실자를 찾는 중</div>
        )}
      </div>
    </section>
  )
}

function comfortMeta(comfort: 'hot' | 'cold' | 'neutral' | 'unknown') {
  return {
    hot: { label: '더움', cls: 'metric-hot' },
    cold: { label: '추움', cls: 'metric-cold' },
    neutral: { label: '쾌적', cls: 'metric-neutral' },
    unknown: { label: '대기', cls: 'metric-unknown' },
  }[comfort]
}

function OccupantCard({
  person,
}: {
  person: HvacRecommendation['occupant_details'][number]
}) {
  const meta = comfortMeta(person.comfort)
  const score =
    person.comfort_score == null
      ? '온도 대기'
      : `${Math.round(person.comfort_score * 100)}`
  return (
    <div className={`occupant ${meta.cls}`}>
      <div className="occupant-head">
        <span className="occupant-name">{person.label}</span>
        <span className="occupant-comfort">{meta.label}</span>
      </div>
      <div className="occupant-temp">
        {fmtCelsius(person.skin_temperature_c) ?? '피부온도 대기'}
      </div>
      <div className="occupant-meta">
        <span>score {score}</span>
        <span>{person.sample_count} samples</span>
      </div>
    </div>
  )
}

function Metric({
  label,
  value,
  hint,
  tone,
}: {
  label: string
  value: string
  hint?: string
  tone?: string
}) {
  return (
    <div className={`metric ${tone ?? ''}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
      {hint && <span className="metric-hint">{hint}</span>}
    </div>
  )
}

function BehaviorSection({ behavior }: { behavior: BehaviorData }) {
  const signals: { key: keyof BehaviorData; label: string; emoji: string }[] = [
    { key: 'arms_crossed',  label: 'arms crossed',  emoji: '🤞' },
    { key: 'hunched',       label: 'hunched',       emoji: '🫨' },
    { key: 'hands_up',      label: 'hands up',      emoji: '🙋' },
    { key: 'fanning',       label: 'fanning',       emoji: '🌬️' },
    { key: 'touching_face', label: 'touching face', emoji: '🫳' },
  ]
  const fused = behavior.fused_comfort ?? behavior.comfort
  const comfortInfo = {
    hot:     { label: '덥다',  emoji: '🔥', cls: 'comfort-hot' },
    cold:    { label: '춥다',  emoji: '❄️', cls: 'comfort-cold' },
    neutral: { label: '보통',  emoji: '😐', cls: 'comfort-neutral' },
  }[fused]

  const clipEntries = behavior.clip_scores
    ? Object.entries(behavior.clip_scores)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
    : []

  return (
    <section className="behavior">
      <div className={`comfort-badge ${comfortInfo.cls}`}>
        <span className="comfort-emoji">{comfortInfo.emoji}</span>
        <span className="comfort-label">{comfortInfo.label}</span>
        <span className="comfort-sub">heuristic + clip + vlm</span>
        <div className="comfort-detail">
          <span>휴리스틱: <strong>{behavior.comfort}</strong></span>
          {behavior.vlm_comfort && (
            <span>VLM: <strong>{behavior.vlm_comfort}</strong></span>
          )}
        </div>
      </div>
      <div className="behavior-signals">
        {signals.map((s) => {
          const v = behavior[s.key] as number
          return (
            <div key={s.key} className="signal" data-strong={v > 0.5}>
              <div className="signal-head">
                <span className="signal-emoji">{s.emoji}</span>
                <span className="signal-label">{s.label}</span>
                <span className="signal-value">{Math.round(v * 100)}</span>
              </div>
              <div className="signal-bar">
                <div className="signal-bar-fill" style={{ width: `${v * 100}%` }} />
              </div>
            </div>
          )
        })}
      </div>

      {(clipEntries.length > 0 || behavior.vlm_answer) && (
        <div className="behavior-secondary">
          {clipEntries.length > 0 && (
            <div className="clip-box">
              <div className="box-title">CLIP top matches</div>
              {clipEntries.map(([q, v]) => (
                <div key={q} className="clip-row">
                  <span className="clip-q">{q}</span>
                  <span className="clip-v">{Math.round(v * 100)}</span>
                </div>
              ))}
            </div>
          )}
          {behavior.vlm_answer && (
            <div className="vlm-box">
              <div className="box-title">VLM (Qwen2-VL-2B)</div>
              <div className="vlm-answer">{behavior.vlm_answer}</div>
            </div>
          )}
        </div>
      )}
    </section>
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

function FaceOverlay({
  face,
  celsius,
}: {
  face: FaceRegions
  celsius: number | null
}) {
  const tempLabel = celsius != null ? `${celsius.toFixed(1)} °C` : null
  // 회전된 quad가 있으면 그걸로, 없으면 axis-aligned bbox로 fallback.
  const quad = face.quad && face.quad.length === 4 ? face.quad : null
  const quadPoints = quad ? quad.map((p) => `${p.x},${p.y}`).join(' ') : null
  const labelAnchor = quad
    ? quad.reduce(
        (acc, p) => (p.y < acc.y || (p.y === acc.y && p.x < acc.x) ? p : acc),
        quad[0],
      )
    : { x: face.bbox.x, y: face.bbox.y }
  return (
    <g>
      {/* 얼굴 전체 bbox — roll에 맞춰 회전된 polygon */}
      {quadPoints ? (
        <polygon
          points={quadPoints}
          fill="none"
          stroke="#3fb950"
          strokeWidth={2}
        />
      ) : (
        <rect
          x={face.bbox.x}
          y={face.bbox.y}
          width={face.bbox.w}
          height={face.bbox.h}
          fill="none"
          stroke="#3fb950"
          strokeWidth={2}
        />
      )}
      {tempLabel && (
        <g>
          <rect
            x={labelAnchor.x}
            y={Math.max(0, labelAnchor.y - 22)}
            width={Math.max(72, tempLabel.length * 8)}
            height={20}
            fill="rgba(0, 0, 0, 0.65)"
            rx={3}
          />
          <text
            x={labelAnchor.x + 6}
            y={Math.max(14, labelAnchor.y - 7)}
            fill="#3fb950"
            fontSize={14}
            fontWeight={700}
          >
            {tempLabel}
          </text>
        </g>
      )}
      {/* 핵심 영역들 — roll에 맞춰 회전된 polygon */}
      <RegionQuad quad={face.forehead_quad}    box={face.forehead_box}    color="#ffd166" label="forehead" />
      <RegionQuad quad={face.cheek_left_quad}  box={face.cheek_left_box}  color="#ff8b3d" label="cheek L" />
      <RegionQuad quad={face.cheek_right_quad} box={face.cheek_right_box} color="#ff8b3d" label="cheek R" />
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

function RegionQuad({
  quad,
  box,
  color,
  label,
}: {
  quad: Pt[]
  box: BBox
  color: string
  label: string
}) {
  if (!quad || quad.length !== 4) {
    return <RegionBox box={box} color={color} label={label} />
  }
  const points = quad.map((p) => `${p.x},${p.y}`).join(' ')
  const anchor = quad.reduce(
    (acc, p) => (p.y < acc.y || (p.y === acc.y && p.x < acc.x) ? p : acc),
    quad[0],
  )
  return (
    <g>
      <polygon
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeDasharray="3 2"
      />
      <text x={anchor.x} y={anchor.y - 3} fill={color} fontSize={10} fontWeight={600}>
        {label}
      </text>
    </g>
  )
}

function applyH(H: number[][], p: Pt): Pt {
  const x = H[0][0] * p.x + H[0][1] * p.y + H[0][2]
  const y = H[1][0] * p.x + H[1][1] * p.y + H[1][2]
  const w = H[2][0] * p.x + H[2][1] * p.y + H[2][2]
  if (!w) return { x: 0, y: 0 }
  return { x: x / w, y: y / w }
}

function transformQuad(quad: Pt[] | undefined, H: number[][] | null): Pt[] | null {
  if (!H || !quad || quad.length !== 4) return null
  return quad.map((p) => applyH(H, p))
}

function bboxToQuad(b: BBox): Pt[] {
  return [
    { x: b.x, y: b.y },
    { x: b.x + b.w, y: b.y },
    { x: b.x + b.w, y: b.y + b.h },
    { x: b.x, y: b.y + b.h },
  ]
}

function quadOrBbox(quad: Pt[] | undefined, box: BBox): Pt[] {
  return quad && quad.length === 4 ? quad : bboxToQuad(box)
}

function ThermalPanel({
  src,
  minmax,
  probe,
  thermalFaces,
  faceTemps,
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
  thermalFaces: FaceRegions[]
  faceTemps: FaceTemp[]
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
          {!calMode && minmax && (
            <Marker x={minmax.min_x} y={minmax.min_y} color="#58a6ff"
                    label={fmtCelsius(minmax.min_celsius) ?? 'min'} />
          )}
          {!calMode && minmax && (
            <Marker x={minmax.max_x} y={minmax.max_y} color="#ff6a3d"
                    label={fmtCelsius(minmax.max_celsius) ?? 'max'} />
          )}
          {!calMode && probe && (
            <Marker x={probe.x} y={probe.y} color="#e6edf3"
                    label={fmtCelsius(probe.celsius) ?? `raw ${probe.raw}`} />
          )}
          {!calMode &&
            thermalFaces.map((f, i) => {
              const celsius =
                faceTemps.find((t) => t.index === i)?.mean_celsius ?? null
              return <FaceOverlay key={i} face={f} celsius={celsius} />
            })}
          {calPairs.map((p, i) => (
            <CalMarker key={i} x={p.thermal.x} y={p.thermal.y} index={i + 1} />
          ))}
        </svg>
      </div>
    </div>
  )
}

function FaceThermalQuadOverlay({
  quad,
  forehead,
  cheekL,
  cheekR,
  celsius,
}: {
  quad: Pt[]
  forehead: Pt[] | null
  cheekL: Pt[] | null
  cheekR: Pt[] | null
  celsius: number | null
}) {
  const tempLabel = celsius != null ? `${celsius.toFixed(1)} °C` : null
  const facePts = quad.map((p) => `${p.x},${p.y}`).join(' ')
  const anchor = quad.reduce(
    (acc, p) => (p.y < acc.y || (p.y === acc.y && p.x < acc.x) ? p : acc),
    quad[0],
  )
  const labelX = anchor.x
  const labelY = Math.max(0, anchor.y - 22)
  const drawSubQuad = (q: Pt[] | null, color: string) =>
    q && (
      <polygon
        points={q.map((p) => `${p.x},${p.y}`).join(' ')}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeDasharray="3 2"
      />
    )
  return (
    <g>
      <polygon
        points={facePts}
        fill="rgba(63, 185, 80, 0.10)"
        stroke="#3fb950"
        strokeWidth={2}
      />
      {drawSubQuad(forehead, '#ffd166')}
      {drawSubQuad(cheekL, '#ff8b3d')}
      {drawSubQuad(cheekR, '#ff8b3d')}
      {tempLabel && (
        <>
          <rect
            x={labelX}
            y={labelY}
            width={Math.max(72, tempLabel.length * 8)}
            height={20}
            fill="rgba(0, 0, 0, 0.65)"
            rx={3}
          />
          <text
            x={labelX + 6}
            y={labelY + 14}
            fill="#3fb950"
            fontSize={13}
            fontWeight={700}
          >
            {tempLabel}
          </text>
        </>
      )}
    </g>
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
  value: number | string
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
