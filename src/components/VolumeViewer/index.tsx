/**
 * Primary 3D field visualization using Three.js ray-marched volume rendering
 * plus a wireframe overlay of the physical coil geometry.
 *
 * The solver returns a normalized [0,1] scalar field on a regular grid.
 * We load it into a Data3DTexture and ray-march through it in GLSL3.
 * A TubeGeometry coil is rendered on top so you can see what you're building.
 */

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { CoilEntity, CoilParams, CoilType, FieldMaximum, FieldName, VolumeData } from "../../lib/fieldTypes";
import { FIELD_CHIP, FIELD_UNITS } from "../../lib/fieldTypes";
import { FIELD_CHIP_COLOR, DEFAULT_CHIP_COLOR } from "../../lib/colormap";

interface Props {
  volume:        VolumeData | null;
  selectedField: FieldName;
  isSolving:     boolean;
  maxima:        FieldMaximum[];
  entity:        CoilEntity;
  domainRadius:  number;          // meters
  /** Lead attachment points per entity [[start_m, end_m]]. */
  leadPoints?:   [[number,number,number],[number,number,number]][];
}

// ---------------------------------------------------------------------------
// GLSL shaders
// ---------------------------------------------------------------------------

const VERT = /* glsl */`
// NOTE: do NOT redeclare "in vec3 position" here.
// Three.js r165 prepends "#define attribute in" + "attribute vec3 position;" to
// every vertex shader, which expands to "in vec3 position;".  A second declaration
// causes a GLSL ES 3.0 redefinition error on WebKit → silent black output.
out vec3 vOrigin;
out vec3 vDirection;
uniform mat4 uInverseModel;

void main() {
  vOrigin    = vec3(uInverseModel * vec4(cameraPosition, 1.0));
  vDirection = position - vOrigin;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAG = /* glsl */`
precision highp float;
precision highp sampler3D;

uniform sampler3D uVolume;
uniform float     uThreshold;
uniform float     uOpacity;
uniform int       uColormap;   // 0 = viridis, 1 = plasma

in vec3 vOrigin;
in vec3 vDirection;
out vec4 fragColor;
// Three.js GLSL3 mode: NO pc_fragColor injection. Must use explicit out var.

vec3 viridis(float t) {
  const vec3 c0 = vec3(0.2777, 0.0054, 0.3341);
  const vec3 c1 = vec3(0.1051, 1.4046, 1.3846);
  const vec3 c2 = vec3(-0.3309, 0.2148, 0.0951);
  const vec3 c3 = vec3(-4.6342, -5.7991, -19.3324);
  const vec3 c4 = vec3(6.2283, 14.1799, 56.6906);
  const vec3 c5 = vec3(4.7764, -13.7451, -65.3530);
  const vec3 c6 = vec3(-5.4355, 4.6459, 26.3124);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), 0.0, 1.0);
}

vec3 plasma(float t) {
  const vec3 c0 = vec3(0.0587, 0.0233, 0.5433);
  const vec3 c1 = vec3(2.1765, 0.2384, 0.7540);
  const vec3 c2 = vec3(-2.6895, -7.4559, 3.1108);
  const vec3 c3 = vec3(6.1303, 42.3462, -28.5189);
  const vec3 c4 = vec3(-11.1074, -82.6663, 60.1398);
  const vec3 c5 = vec3(10.0231, 71.4136, -54.0722);
  const vec3 c6 = vec3(-3.6587, -22.9315, 18.1919);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), 0.0, 1.0);
}

vec3 transferColor(float t) {
  return uColormap == 0 ? viridis(t) : plasma(t);
}

vec2 hitBox(vec3 orig, vec3 dir) {
  vec3 inv = 1.0 / dir;
  vec3 t0 = (vec3(-0.5) - orig) * inv;
  vec3 t1 = (vec3( 0.5) - orig) * inv;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  return vec2(max(tmin.x, max(tmin.y, tmin.z)),
              min(tmax.x, min(tmax.y, tmax.z)));
}

void main() {
  vec3  dir    = normalize(vDirection);
  vec2  bounds = hitBox(vOrigin, dir);
  if (bounds.x > bounds.y) discard;
  bounds.x = max(bounds.x, 0.0);

  const int STEPS = 160;
  float dt = (bounds.y - bounds.x) / float(STEPS);

  vec4 accum = vec4(0.0);
  for (int i = 0; i < STEPS; i++) {
    float t   = bounds.x + (float(i) + 0.5) * dt;
    vec3  pos = vOrigin + t * dir;
    float val = texture(uVolume, pos + 0.5).r;

    if (val > uThreshold) {
      float mapped = (val - uThreshold) / max(1.0 - uThreshold, 0.001);
      vec3  col    = transferColor(mapped);
      float alpha  = mapped * dt * uOpacity * 18.0;
      alpha = clamp(alpha, 0.0, 0.3);

      accum.rgb += (1.0 - accum.a) * col * alpha;
      accum.a   += (1.0 - accum.a) * alpha;
      if (accum.a > 0.98) break;
    }
  }

  if (accum.a < 0.004) discard;
  fragColor = accum;
}
`;

// ---------------------------------------------------------------------------
// Coil path generation — mirrors coil.py geometry, physical coords (meters)
// ---------------------------------------------------------------------------

type Pt3 = [number, number, number];

function buildCoilPath(p: CoilParams): Pt3[] {
  switch (p.coil_type as CoilType) {
    case "solenoid":
    case "open_helix":       return solenoidPath(p);
    case "toroid":           return toroidPath(p);
    case "toroid_poloidal":  return toroidPoloidalPath(p);
    case "flat_spiral":      return flatSpiralPath(p);
    case "rodin":            return rodinPath(p);
    case "capacitor_symmetric":
    case "capacitor_asymmetric": return capacitorPath(p);
    default:                 return [];
  }
}

/** Two circular discs (plate outlines) for capacitor visualisation. */
function capacitorPath(p: CoilParams): Pt3[] {
  const R    = p.radius_m;
  const gap  = (p.plate_gap_m ?? 0) > 0 ? (p.plate_gap_m ?? 0.02) : 0.02;
  const S    = 48;
  const pts: Pt3[] = [];
  // Top plate at z = +gap/2
  for (let i = 0; i <= S; i++) {
    const theta = (2 * Math.PI * i) / S;
    pts.push([R * Math.cos(theta), R * Math.sin(theta), gap / 2]);
  }
  pts.push([NaN, NaN, NaN]);
  // Small electrode at z = -gap/2 (for asymmetric, radius = R/aspect)
  const r2 = p.coil_type === "capacitor_asymmetric"
    ? R / Math.max(p.plate_aspect ?? 5, 1)
    : R;
  for (let i = 0; i <= S; i++) {
    const theta = (2 * Math.PI * i) / S;
    pts.push([r2 * Math.cos(theta), r2 * Math.sin(theta), -gap / 2]);
  }
  return pts;
}

/** Continuous helix. */
function solenoidPath(p: CoilParams): Pt3[] {
  const R = p.radius_m, N = p.turns, pitch = p.pitch_m;
  const z0 = -N * pitch / 2;
  const S = 48;
  const pts: Pt3[] = [];
  for (let i = 0; i <= N * S; i++) {
    const theta = (2 * Math.PI * i) / S;
    pts.push([R * Math.cos(theta), R * Math.sin(theta), z0 + (i / S) * pitch]);
  }
  return pts;
}

/** N small poloidal loops arranged azimuthally around the torus. */
function toroidPath(p: CoilParams): Pt3[] {
  const R = p.radius_m, N = p.turns, pitch = p.pitch_m;
  const r_minor = Math.max(pitch * N / (2 * Math.PI), p.wire_radius_m * 4);
  const S = 32;
  const pts: Pt3[] = [];
  for (let i = 0; i < N; i++) {
    const phi = (2 * Math.PI * i) / N;
    const cx = R * Math.cos(phi), cy = R * Math.sin(phi);
    for (let j = 0; j <= S; j++) {
      const theta = (2 * Math.PI * j) / S;
      // Poloidal loop: in the plane containing the z-axis and radial dir at phi
      pts.push([
        cx + r_minor * Math.cos(theta) * Math.cos(phi),
        cy + r_minor * Math.cos(theta) * Math.sin(phi),
        r_minor * Math.sin(theta),
      ]);
    }
    // NaN gap so TubeGeometry segments don't connect between loops
    if (i < N - 1) pts.push([NaN, NaN, NaN]);
  }
  return pts;
}

/** N loops in radial (poloidal) planes — each wound the short way around the torus. */
function toroidPoloidalPath(p: CoilParams): Pt3[] {
  const R = p.radius_m, N = p.turns, pitch = p.pitch_m;
  const r_minor = pitch * N / (2 * Math.PI);
  const S = 32;
  const pts: Pt3[] = [];
  for (let i = 0; i < N; i++) {
    const phi = (2 * Math.PI * i) / N;
    const cx = R * Math.cos(phi), cy = R * Math.sin(phi);
    for (let j = 0; j <= S; j++) {
      const theta = (2 * Math.PI * j) / S;
      pts.push([
        cx + r_minor * Math.cos(theta) * Math.cos(phi),
        cy + r_minor * Math.cos(theta) * Math.sin(phi),
        r_minor * Math.sin(theta),
      ]);
    }
    if (i < N - 1) pts.push([NaN, NaN, NaN]);
  }
  return pts;
}

/** Archimedean spiral in the z=0 plane. */
function flatSpiralPath(p: CoilParams): Pt3[] {
  const N = p.turns, pitch = p.pitch_m;
  const r0 = p.radius_m - (N - 1) * pitch / 2;
  const S = 64;
  const pts: Pt3[] = [];
  for (let i = 0; i <= N * S; i++) {
    const theta = (2 * Math.PI * i) / S;
    const r = r0 + (i / S) * pitch;
    if (r > 0) pts.push([r * Math.cos(theta), r * Math.sin(theta), 0]);
  }
  return pts;
}

/** Figure-8 Rodin pattern: alternating-tilt loops arranged on a torus. */
function rodinPath(p: CoilParams): Pt3[] {
  const R = p.radius_m, N = p.turns;
  const r_minor = Math.max(p.pitch_m * N / (2 * Math.PI), p.wire_radius_m * 4);
  const tilt = Math.PI / 4;
  const S = 32;
  const pts: Pt3[] = [];
  for (let i = 0; i < N; i++) {
    const phi = (2 * Math.PI * i) / N;
    const sign = i % 2 === 0 ? 1 : -1;
    const cx = R * Math.cos(phi), cy = R * Math.sin(phi);
    const ct = Math.cos(sign * tilt), st = Math.sin(sign * tilt);
    for (let j = 0; j <= S; j++) {
      const theta = (2 * Math.PI * j) / S;
      const cosT = Math.cos(theta), sinT = Math.sin(theta);
      // Apply tilt: rotate the loop around the radial axis
      const localR = r_minor * cosT;
      const localZ = r_minor * sinT * ct;
      pts.push([
        cx + localR * Math.cos(phi),
        cy + localR * Math.sin(phi),
        localZ + sign * r_minor * st * 0.3,
      ]);
    }
    if (i < N - 1) pts.push([NaN, NaN, NaN]);
  }
  return pts;
}

/**
 * Build Three.js coil objects from physical path points.
 * Splits path at NaN gaps so disconnected loops render correctly.
 * Returns a Group of tubes.
 */
function buildCoilGroup(
  coilParams: CoilParams,
  maxS: number,       // physical normalizer: world = physical / maxS
  tubeFraction: number = 0.006,  // tube radius as fraction of world box
): THREE.Group {
  const raw = buildCoilPath(coilParams);
  const scale = 1 / maxS;
  const tubeR = tubeFraction;

  // Split on NaN sentinels
  const segments: THREE.Vector3[][] = [];
  let cur: THREE.Vector3[] = [];
  for (const [x, y, z] of raw) {
    if (isNaN(x)) {
      if (cur.length > 1) segments.push(cur);
      cur = [];
    } else {
      cur.push(new THREE.Vector3(x * scale, y * scale, z * scale));
    }
  }
  if (cur.length > 1) segments.push(cur);

  const mat = new THREE.MeshStandardMaterial({
    color: 0xff8c00,
    emissive: 0xff6600,
    emissiveIntensity: 0.6,
    roughness: 0.4,
    metalness: 0.2,
  });

  const group = new THREE.Group();
  for (const seg of segments) {
    if (seg.length < 2) continue;
    const curve = new THREE.CatmullRomCurve3(seg, false, "catmullrom", 0.5);
    const numSeg = Math.min(seg.length * 3, 512);
    try {
      const geo = new THREE.TubeGeometry(curve, numSeg, tubeR, 6, false);
      group.add(new THREE.Mesh(geo, mat));
    } catch {
      // Fallback to line for degenerate curves
      const pts = curve.getPoints(numSeg);
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      group.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xff8c00 })));
    }
  }
  return group;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function VolumeViewer({ volume, selectedField, isSolving, maxima, entity, domainRadius, leadPoints }: Props) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const sceneRef  = useRef<SceneState | null>(null);
  const [threshold, setThreshold] = useState(0.02);
  const [opacity,   setOpacity]   = useState(1.8);

  // 0 = viridis, 1 = plasma — plasma for vector magnitudes, viridis otherwise
  const colormap = ["A_magnitude", "B_magnitude", "poynting_mag"].includes(selectedField) ? 1 : 0;

  // Setup scene once on mount
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const state = initScene(el);
    sceneRef.current = state;
    return () => {
      state.controls.dispose();
      state.renderer.dispose();
      state.renderer.domElement.remove();
    };
  }, []);

  // Update volume texture whenever data arrives
  useEffect(() => {
    const state = sceneRef.current;
    if (!state) return;
    if (volume) {
      console.log("[VolumeViewer] updateVolume:", volume.field, "shape:", volume.shape,
        "range:", volume.field_min.toExponential(2), "→", volume.field_max.toExponential(2),
        "data[0..3]:", volume.data.slice(0, 3));
      updateVolume(state, volume);
    } else {
      clearVolume(state);
    }
  }, [volume]);

  // Helper to compute maxS and domain center from volume or domain radius
  const getScaling = () => {
    if (volume) {
      const [x0, x1] = volume.x_range;
      const [y0, y1] = volume.y_range;
      const [z0, z1] = volume.z_range;
      const maxS: number = Math.max(x1 - x0, y1 - y0, z1 - z0);
      const center: [number, number, number] = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2];
      return { maxS, center };
    }
    return { maxS: 2 * domainRadius, center: [0, 0, 0] as [number, number, number] };
  };

  // Update coil geometry when entity or volume bounds change
  useEffect(() => {
    const state = sceneRef.current;
    if (!state) return;
    const { maxS } = getScaling();
    updateCoil(state, entity.coil, maxS);
  }, [entity, domainRadius, volume]);

  // Update lead point markers
  useEffect(() => {
    const state = sceneRef.current;
    if (!state) return;
    const { maxS, center } = getScaling();
    updateLeads(state, leadPoints ?? [], maxS, center);
  }, [leadPoints, domainRadius, volume]);

  // Push uniform changes each frame
  useEffect(() => {
    const state = sceneRef.current;
    if (!state?.material) return;
    state.material.uniforms.uThreshold.value = threshold;
    state.material.uniforms.uOpacity.value   = opacity;
    state.material.uniforms.uColormap.value  = colormap;
  }, [threshold, opacity, colormap]);

  const fieldLabel = (f: string) => FIELD_CHIP[f as FieldName] ?? f;

  /** Format a raw SI value to human-readable with metric prefix (μ, m, k…) */
  const fmtSI = (v: number, unit: string): string => {
    const a = Math.abs(v);
    if (a === 0) return `0 ${unit}`;
    if (a >= 1e6)  return `${(v / 1e6).toPrecision(3)} M${unit}`;
    if (a >= 1e3)  return `${(v / 1e3).toPrecision(3)} k${unit}`;
    if (a >= 1)    return `${v.toPrecision(3)} ${unit}`;
    if (a >= 1e-3) return `${(v * 1e3).toPrecision(3)} m${unit}`;
    if (a >= 1e-6) return `${(v * 1e6).toPrecision(3)} μ${unit}`;
    if (a >= 1e-9) return `${(v * 1e9).toPrecision(3)} n${unit}`;
    return `${v.toExponential(2)} ${unit}`;
  };

  return (
    <div className="absolute inset-0 bg-app overflow-hidden">

      {/* Three.js mount point */}
      <div ref={canvasRef} className="absolute inset-0" />

      {/* Empty-state overlay */}
      {!volume && !isSolving && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <div className="text-5xl mb-4 opacity-10">⬡</div>
            <div className="text-slate-600 text-xs">Adjust controls — field updates automatically</div>
          </div>
        </div>
      )}

      {/* Solving spinner */}
      {isSolving && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <Spinner />
        </div>
      )}

      {/* Top-left: active field max — prominent, updates with every solve */}
      {volume && (
        <div className="absolute top-3 left-3 pointer-events-none flex flex-col gap-0.5">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider">
            {fieldLabel(volume.field)} · peak
          </span>
          <span className="text-base font-semibold tabular-nums text-slate-100">
            {fmtSI(volume.field_max, FIELD_UNITS[volume.field] ?? "")}
          </span>
          <span className="text-[10px] text-slate-600 tabular-nums">
            min {fmtSI(volume.field_min, FIELD_UNITS[volume.field] ?? "")}
            &nbsp;· normalized view
          </span>
          {/* Lead indicator chips */}
          {leadPoints && leadPoints.length > 0 && (
            <div className="mt-1 flex flex-col gap-0.5">
              {leadPoints.map(([a, c], i) => (
                <div key={i} className="flex gap-1 items-center">
                  <span className="text-[9px] px-1 py-0.5 rounded bg-red-900/50 text-red-300 tabular-nums">
                    +V ({a[0].toFixed(3)}, {a[1].toFixed(3)}, {a[2].toFixed(3)})m
                  </span>
                  <span className="text-[9px] px-1 py-0.5 rounded bg-blue-900/50 text-blue-300 tabular-nums">
                    −V ({c[0].toFixed(3)}, {c[1].toFixed(3)}, {c[2].toFixed(3)})m
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Top-right: per-field maxima table */}
      {maxima.length > 0 && (
        <div className="absolute top-3 right-3 flex flex-col items-end gap-1 pointer-events-none">
          {maxima.map(m => (
            <span
              key={m.field}
              className={`text-xs tabular-nums px-2 py-0.5 rounded ${FIELD_CHIP_COLOR[m.field as FieldName] ?? DEFAULT_CHIP_COLOR}`}
            >
              {fieldLabel(m.field)} {fmtSI(m.max_value, FIELD_UNITS[m.field] ?? "")}
            </span>
          ))}
        </div>
      )}

      {/* Bottom-right: transfer-function controls */}
      <div className="absolute bottom-3 right-3 flex flex-col gap-2 w-44 bg-app/80 backdrop-blur-sm rounded border border-rim p-2">
        <Control label="Threshold" value={threshold} min={0}   max={0.9} step={0.01}
          onChange={setThreshold} fmt={v => v.toFixed(2)} />
        <Control label="Opacity"   value={opacity}   min={0.1} max={6}   step={0.1}
          onChange={setOpacity}   fmt={v => v.toFixed(1)} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Three.js scene management
// ---------------------------------------------------------------------------

interface SceneState {
  renderer:   THREE.WebGLRenderer;
  scene:      THREE.Scene;
  camera:     THREE.PerspectiveCamera;
  controls:   OrbitControls;
  mesh:       THREE.Mesh;
  material:   THREE.ShaderMaterial;
  texture:    THREE.Data3DTexture | null;
  coilGroup:  THREE.Group;
  leadGroup:  THREE.Group;    // lead point sphere markers
  animId:     number;
}

function initScene(container: HTMLDivElement): SceneState {
  const w = container.clientWidth  || container.offsetWidth  || 800;
  const h = container.clientHeight || container.offsetHeight || 600;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x09090d, 1);
  renderer.shadowMap.enabled = false;
  container.appendChild(renderer.domElement);

  const scene  = new THREE.Scene();

  // Ambient + directional light for the coil tubes
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
  dirLight.position.set(2, 3, 2);
  scene.add(dirLight);

  const camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 100);
  camera.position.set(1.6, 1.1, 1.6);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance   = 0.5;
  controls.maxDistance   = 8;

  const material = new THREE.ShaderMaterial({
    glslVersion:    THREE.GLSL3,
    vertexShader:   VERT,
    fragmentShader: FRAG,
    uniforms: {
      uVolume:       { value: null },
      uThreshold:    { value: 0.02 },
      uOpacity:      { value: 1.8 },
      uColormap:     { value: 0 },
      uInverseModel: { value: new THREE.Matrix4() },
    },
    side:        THREE.BackSide,
    transparent: true,
    depthTest:   false,   // Don't depth-clip against coil tubes or other opaques
    depthWrite:  false,
  });

  const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
  mesh.renderOrder = 0;   // Render volume first (depthTest disabled, transparent)
  scene.add(mesh);

  // Coil group — rendered after the volume so tubes appear on top
  const coilGroup = new THREE.Group();
  coilGroup.renderOrder = 1;
  scene.add(coilGroup);

  // Lead point markers — spheres at wire endpoints / plate centres
  const leadGroup = new THREE.Group();
  leadGroup.renderOrder = 2;
  scene.add(leadGroup);

  // Resize observer
  const ro = new ResizeObserver(() => {
    const nw = container.clientWidth;
    const nh = container.clientHeight;
    if (nw === 0 || nh === 0) return;
    renderer.setSize(nw, nh);
    camera.aspect = nw / nh;
    camera.updateProjectionMatrix();
  });
  ro.observe(container);

  // Animation loop
  let animId = 0;
  function animate() {
    animId = requestAnimationFrame(animate);
    controls.update();
    mesh.updateMatrixWorld();
    material.uniforms.uInverseModel.value.copy(mesh.matrixWorld).invert();
    renderer.render(scene, camera);
  }
  animate();

  return { renderer, scene, camera, controls, mesh, material, texture: null, coilGroup, leadGroup, animId };
}

function updateVolume(state: SceneState, vol: VolumeData) {
  const [nx, ny, nz] = vol.shape;
  const raw = new Float32Array(vol.data);

  state.texture?.dispose();

  const tex = new THREE.Data3DTexture(raw, nx, ny, nz);
  tex.format      = THREE.RedFormat;
  tex.type        = THREE.FloatType;
  tex.minFilter   = THREE.LinearFilter;
  tex.magFilter   = THREE.LinearFilter;
  tex.wrapS       = THREE.ClampToEdgeWrapping;
  tex.wrapT       = THREE.ClampToEdgeWrapping;
  tex.wrapR       = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;

  state.texture = tex;
  state.material.uniforms.uVolume.value = tex;

  const [x0, x1] = vol.x_range;
  const [y0, y1] = vol.y_range;
  const [z0, z1] = vol.z_range;
  const sx = x1 - x0, sy = y1 - y0, sz = z1 - z0;
  const maxS = Math.max(sx, sy, sz);
  state.mesh.scale.set(sx / maxS, sy / maxS, sz / maxS);

  console.log("[VolumeViewer] texture uploaded:", nx, "×", ny, "×", nz,
    "box scale:", (sx/maxS).toFixed(3), (sy/maxS).toFixed(3), (sz/maxS).toFixed(3));
}

function clearVolume(state: SceneState) {
  state.texture?.dispose();
  state.texture = null;
  state.material.uniforms.uVolume.value = null;
}

/** Render sphere markers at lead attachment points.
 *  Anode (+): bright red.  Cathode (−): bright blue.
 *  `domainCenter` is the physical center of the simulation box [m].
 */
function updateLeads(
  state:        SceneState,
  leadPoints:   [[number,number,number],[number,number,number]][],
  maxS:         number,
  domainCenter: [number,number,number],
) {
  state.leadGroup.clear();
  if (!leadPoints || leadPoints.length === 0) return;

  const anodeMat   = new THREE.MeshStandardMaterial({
    color: 0xff4444, emissive: 0xff2222, emissiveIntensity: 0.8,
    roughness: 0.2, metalness: 0.6,
  });
  const cathodeMat = new THREE.MeshStandardMaterial({
    color: 0x4488ff, emissive: 0x2255ff, emissiveIntensity: 0.8,
    roughness: 0.2, metalness: 0.6,
  });

  const sphereR = 0.018; // sphere radius in Three.js units
  const geo = new THREE.SphereGeometry(sphereR, 12, 8);

  for (const [anode, cathode] of leadPoints) {
    // Map physical → Three.js: p3d = (p_phys - domain_center) / maxS
    const an = new THREE.Mesh(geo, anodeMat);
    an.position.set(
      (anode[0] - domainCenter[0]) / maxS,
      (anode[1] - domainCenter[1]) / maxS,
      (anode[2] - domainCenter[2]) / maxS,
    );
    state.leadGroup.add(an);

    const ca = new THREE.Mesh(geo, cathodeMat);
    ca.position.set(
      (cathode[0] - domainCenter[0]) / maxS,
      (cathode[1] - domainCenter[1]) / maxS,
      (cathode[2] - domainCenter[2]) / maxS,
    );
    state.leadGroup.add(ca);
  }
}

function updateCoil(state: SceneState, coilParams: CoilParams, maxS: number) {
  // Remove old coil geometry
  state.coilGroup.clear();

  const group = buildCoilGroup(coilParams, maxS);
  state.coilGroup.add(group);

  console.log("[VolumeViewer] coil updated:", coilParams.coil_type,
    "turns:", coilParams.turns, "r:", coilParams.radius_m, "maxS:", maxS.toFixed(3));
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

function Control({
  label, value, min, max, step, onChange, fmt,
}: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; fmt: (v: number) => string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-slate-500 w-16 shrink-0">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        className="flex-1 h-1 accent-accent cursor-pointer" />
      <span className="text-xs text-slate-400 w-8 text-right tabular-nums shrink-0">
        {fmt(value)}
      </span>
    </div>
  );
}

function Spinner() {
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="w-8 h-8 border-2 border-slate-700 border-t-accent rounded-full animate-spin" />
      <span className="text-xs text-slate-600">Solving…</span>
    </div>
  );
}
