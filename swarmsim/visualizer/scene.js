import * as THREE from "three";
import { SwarmSocket } from "./socket.js";

const GRID_SIZE = 32;
const AGENT_COLORS = [
  0x4285f4, 0x34a853, 0xea4335, 0xfbbc04, 0xa142f4, 0x00bcd4,
];
const UNEXPLORED = { r: 40, g: 44, b: 52 };

const container = document.getElementById("canvas-container");
const coverageEl = document.getElementById("coverage");
const stepEl = document.getElementById("step");
const agentCountEl = document.getElementById("agent-count");
const commCountEl = document.getElementById("comm-count");
const statusEl = document.getElementById("status");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1117);

const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
camera.position.set(0, 0, 2);
camera.lookAt(0, 0, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const gridData = new Uint8Array(GRID_SIZE * GRID_SIZE * 4);
const texture = new THREE.DataTexture(gridData, GRID_SIZE, GRID_SIZE, THREE.RGBAFormat);
texture.magFilter = THREE.NearestFilter;
texture.minFilter = THREE.NearestFilter;

const gridMesh = new THREE.Mesh(
  new THREE.PlaneGeometry(2, 2),
  new THREE.MeshBasicMaterial({ map: texture })
);
scene.add(gridMesh);

const agentMeshes = [];
const commLines = new THREE.LineSegments(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 })
);
scene.add(commLines);

function resize() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  renderer.setSize(w, h);
  const aspect = w / h;
  const view = 1.1;
  camera.left = -view * aspect;
  camera.right = view * aspect;
  camera.top = view;
  camera.bottom = -view;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function idToColor(id) {
  if (id === 0) return UNEXPLORED;
  const hex = AGENT_COLORS[(id - 1) % AGENT_COLORS.length];
  return { r: (hex >> 16) & 255, g: (hex >> 8) & 255, b: hex & 255 };
}

function updateGrid(gridB64) {
  const raw = Uint8Array.from(atob(gridB64), (c) => c.charCodeAt(0));
  for (let i = 0; i < raw.length; i++) {
    const color = idToColor(raw[i]);
    const j = i * 4;
    gridData[j] = color.r;
    gridData[j + 1] = color.g;
    gridData[j + 2] = color.b;
    gridData[j + 3] = 255;
  }
  texture.needsUpdate = true;
}

function normToWorld(x, y) {
  return { x: x * 2 - 1, y: 1 - y * 2 };
}

function ensureAgentMesh(id) {
  while (agentMeshes.length <= id) {
    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(0.03, 0.08, 8),
      new THREE.MeshBasicMaterial({ color: AGENT_COLORS[agentMeshes.length % AGENT_COLORS.length] })
    );
    scene.add(cone);
    agentMeshes.push(cone);
  }
  return agentMeshes[id];
}

function updateAgents(agents) {
  agentMeshes.forEach((m) => { m.visible = false; });
  agents.forEach((agent) => {
    const mesh = ensureAgentMesh(agent.id);
    const pos = normToWorld(agent.x, agent.y);
    mesh.position.set(pos.x, pos.y, 0.05);
    mesh.rotation.z = -agent.heading;
    mesh.visible = true;
  });
}

function updateCommLinks(links, agents) {
  const positions = [];
  links.forEach(([i, j]) => {
    const a = agents.find((x) => x.id === i);
    const b = agents.find((x) => x.id === j);
    if (!a || !b) return;
    const pa = normToWorld(a.x, a.y);
    const pb = normToWorld(b.x, b.y);
    positions.push(pa.x, pa.y, 0.02, pb.x, pb.y, 0.02);
  });
  commLines.geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );
}

function updateStats(state) {
  coverageEl.textContent = `${(state.coverage_pct * 100).toFixed(1)}%`;
  stepEl.textContent = state.step;
  agentCountEl.textContent = state.agents.length;
  commCountEl.textContent = state.comm_links.length;
}

function onState(state) {
  updateGrid(state.grid);
  updateAgents(state.agents);
  updateCommLinks(state.comm_links, state.agents);
  updateStats(state);
}

const socket = new SwarmSocket(
  () => {
    statusEl.textContent = "Connected";
    statusEl.style.color = "#34a853";
  },
  () => {
    statusEl.textContent = "Disconnected — reconnecting...";
    statusEl.style.color = "#ea4335";
  },
  onState
);
socket.connect();

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();
