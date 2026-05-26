/**
 * Primary 3D field visualization using Three.js ray-marched volume rendering.
 *
 * The solver returns a normalized [0,1] scalar field on a regular grid.
 * We load it into a Data3DTexture and ray-march through it in a GLSL shader
 * using front-to-back compositing with a viridis/plasma transfer function.
 *
 * Orbit controls let you rotate/zoom freely around the coil.
 */

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { FieldName, VolumeData } from "../../lib/fieldTypes";
import { FIELD_LABELS } from "../../lib/fieldTypes";

interface Props {
  volume:        VolumeData | null;
  selectedField: FieldName;
  isSolving:     boolean;
}

// ---------------------------------------------------------------------------
// GLSL shaders
// ---------------------------------------------------------------------------

const VERT = /* glsl */`
in vec3 position;
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

// Polynomial viridis approximation (Mateo Zuber / cmasher)
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

// Ray–AABB intersection for unit cube [-0.5, 0.5]^3
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

  const int STEPS = 128;
  float dt = (bounds.y - bounds.x) / float(STEPS);

  vec4 accum = vec4(0.0);
  for (int i = 0; i < STEPS; i++) {
    float t   = bounds.x + (float(i) + 0.5) * dt;
    vec3  pos = vOrigin + t * dir;
    float val = texture(uVolume, pos + 0.5).r;  // [-0.5,0.5] → [0,1]

    if (val > uThreshold) {
      float mapped = (val - uThreshold) / max(1.0 - uThreshold, 0.001);
      vec3  col    = transferColor(mapped);
      float alpha  = mapped * dt * uOpacity * 12.0;
      alpha = clamp(alpha, 0.0, 0.25);

      // Front-to-back compositing
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
// Component
// ---------------------------------------------------------------------------

export function VolumeViewer({ volume, selectedField, isSolving }: Props) {
  const canvasRef  = useRef<HTMLDivElement>(null);
  const sceneRef   = useRef<SceneState | null>(null);
  const [threshold, setThreshold] = useState(0.05);
  const [opacity,   setOpacity]   = useState(1.0);

  // Colormap index: 0=viridis for phi, 1=plasma for others
  const colormap = selectedField === "phi" ? 0 : 1;

  // Setup scene once
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

  // Update volume texture when data changes
  useEffect(() => {
    const state = sceneRef.current;
    if (!state) return;
    if (volume) {
      updateVolume(state, volume);
    } else {
      clearVolume(state);
    }
  }, [volume]);

  // Update uniforms when controls change
  useEffect(() => {
    const state = sceneRef.current;
    if (!state?.material) return;
    state.material.uniforms.uThreshold.value = threshold;
    state.material.uniforms.uOpacity.value   = opacity;
    state.material.uniforms.uColormap.value  = colormap;
  }, [threshold, opacity, colormap]);

  return (
    <div className="relative w-full h-full bg-app">
      {/* Three.js canvas mount */}
      <div ref={canvasRef} className="w-full h-full" />

      {/* Overlay when no data */}
      {!volume && !isSolving && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <div className="text-4xl mb-3 opacity-20">⬡</div>
            <div className="text-slate-600 text-xs">Configure a coil and press Solve</div>
          </div>
        </div>
      )}

      {/* Solving spinner */}
      {isSolving && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <Spinner />
            <div className="text-slate-500 text-xs mt-3">Solving…</div>
          </div>
        </div>
      )}

      {/* Field label */}
      {volume && (
        <div className="absolute top-3 left-3 text-xs text-slate-500 pointer-events-none">
          {FIELD_LABELS[volume.field]}
          <span className="ml-2 text-slate-600">
            [{volume.field_min.toExponential(2)}, {volume.field_max.toExponential(2)}]
          </span>
        </div>
      )}

      {/* Transfer function controls */}
      <div className="absolute bottom-3 right-3 flex flex-col gap-2 w-44 bg-app/80 backdrop-blur-sm rounded border border-rim p-2">
        <Control label="Threshold" value={threshold} min={0} max={0.9} step={0.01}
          onChange={setThreshold} fmt={v => v.toFixed(2)} />
        <Control label="Opacity"   value={opacity}   min={0.1} max={4} step={0.1}
          onChange={setOpacity}   fmt={v => v.toFixed(1)} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Three.js scene management
// ---------------------------------------------------------------------------

interface SceneState {
  renderer:  THREE.WebGLRenderer;
  scene:     THREE.Scene;
  camera:    THREE.PerspectiveCamera;
  controls:  OrbitControls;
  mesh:      THREE.Mesh;
  material:  THREE.ShaderMaterial;
  texture:   THREE.Data3DTexture | null;
  animId:    number;
}

function initScene(container: HTMLDivElement): SceneState {
  const w = container.clientWidth;
  const h = container.clientHeight;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x09090d, 1);
  container.appendChild(renderer.domElement);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, w / h, 0.01, 100);
  camera.position.set(1.6, 1.1, 1.6);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.06;
  controls.minDistance   = 0.5;
  controls.maxDistance   = 8;

  const material = new THREE.ShaderMaterial({
    glslVersion:  THREE.GLSL3,
    vertexShader:   VERT,
    fragmentShader: FRAG,
    uniforms: {
      uVolume:       { value: null },
      uThreshold:    { value: 0.05 },
      uOpacity:      { value: 1.0 },
      uColormap:     { value: 0 },
      uInverseModel: { value: new THREE.Matrix4() },
    },
    side:        THREE.BackSide,
    transparent: true,
    depthWrite:  false,
  });

  const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
  scene.add(mesh);

  // Resize observer
  const ro = new ResizeObserver(() => {
    const nw = container.clientWidth;
    const nh = container.clientHeight;
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
    // Keep inverse model matrix in sync (mesh stays at origin but mat may rotate)
    mesh.updateMatrixWorld();
    material.uniforms.uInverseModel.value.copy(mesh.matrixWorld).invert();
    renderer.render(scene, camera);
  }
  animate();

  return { renderer, scene, camera, controls, mesh, material, texture: null, animId };
}

function updateVolume(state: SceneState, vol: VolumeData) {
  const [nx, ny, nz] = vol.shape;

  // Build Float32Array from normalized data
  const raw = new Float32Array(vol.data);

  // Dispose old texture
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

  // Scale the box to physical aspect ratio
  const [x0, x1] = vol.x_range;
  const [y0, y1] = vol.y_range;
  const [z0, z1] = vol.z_range;
  const sx = x1 - x0, sy = y1 - y0, sz = z1 - z0;
  const maxS = Math.max(sx, sy, sz);
  state.mesh.scale.set(sx / maxS, sy / maxS, sz / maxS);
}

function clearVolume(state: SceneState) {
  state.texture?.dispose();
  state.texture = null;
  state.material.uniforms.uVolume.value = null;
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
    <div className="w-8 h-8 mx-auto border-2 border-slate-700 border-t-accent rounded-full animate-spin" />
  );
}
