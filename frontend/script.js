/**
 * NetSentinel AI — Phase 1 Dashboard JavaScript
 * ================================================
 * Fetches data from the Flask API and renders the dashboard.
 *
 * Architecture:
 *   API module       → raw fetch calls to Flask endpoints
 *   Render module    → pure DOM-manipulation render functions
 *   App controller   → orchestrates load → fetch → render flow
 *
 * No external JS libraries used.
 */

"use strict";

// ================================================================
// Configuration
// ================================================================

const BASE_URL = window.location.origin; // works when served by Flask
const API = {
    health:   `${BASE_URL}/api/health`,
    load:     `${BASE_URL}/api/traffic/load`,
    status:   `${BASE_URL}/api/traffic/status`,
    stats:    `${BASE_URL}/api/traffic/stats`,
    protocols:`${BASE_URL}/api/traffic/protocols`,
    sources:  `${BASE_URL}/api/traffic/top-sources`,
    dests:    `${BASE_URL}/api/traffic/top-destinations`,
    timeline: (freq) => `${BASE_URL}/api/traffic/timeline?freq=${freq}`,
    connections: `${BASE_URL}/api/traffic/connections`,
};

// Current timeline frequency
let currentFreq = "1min";

// Tooltip element (created once, shared)
const tooltip = createTooltip();

// ================================================================
// API helpers
// ================================================================

async function apiFetch(url) {
    const resp = await fetch(url, {
        headers: { "Accept": "application/json" },
        cache: "no-cache",
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${resp.status}`);
    }
    return resp.json();
}

// ================================================================
// DOM utilities
// ================================================================

function el(id) { return document.getElementById(id); }

function setText(id, value) {
    const node = el(id);
    if (node) node.textContent = value;
}

function show(id)  { el(id) && el(id).classList.remove("hidden"); }
function hide(id)  { el(id) && el(id).classList.add("hidden"); }

function showAlert(msg) {
    setText("alertText", msg);
    show("alertBox");
}

function hideAlert() { hide("alertBox"); }

function formatBytes(bytes) {
    if (bytes === null || bytes === undefined) return "—";
    if (bytes >= 1_073_741_824) return (bytes / 1_073_741_824).toFixed(2) + " GB";
    if (bytes >= 1_048_576)     return (bytes / 1_048_576).toFixed(2) + " MB";
    if (bytes >= 1_024)         return (bytes / 1_024).toFixed(1) + " KB";
    return bytes + " B";
}

function formatNum(n) {
    if (n === null || n === undefined) return "—";
    return Number(n).toLocaleString();
}

function formatTime(isoStr) {
    try {
        const d = new Date(isoStr);
        return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    } catch { return isoStr; }
}

// ================================================================
// Tooltip
// ================================================================

function createTooltip() {
    const t = document.createElement("div");
    t.className = "tooltip";
    document.body.appendChild(t);
    return t;
}

function showTooltip(html, evt) {
    tooltip.innerHTML = html;
    tooltip.classList.add("visible");
    positionTooltip(evt);
}

function hideTooltip() { tooltip.classList.remove("visible"); }

function positionTooltip(evt) {
    const x = evt.clientX + 12;
    const y = evt.clientY - 28;
    tooltip.style.left = `${Math.min(x, window.innerWidth - 200)}px`;
    tooltip.style.top  = `${Math.max(y, 8)}px`;
}

// ================================================================
// RENDER: Backend status badge
// ================================================================

function renderStatus(data) {
    const dot   = el("statusDot");
    const label = el("statusLabel");
    if (!dot || !label) return;

    if (data && data.status === "ok") {
        dot.className   = "status-dot ok";
        label.textContent = "Backend Connected";
    } else {
        dot.className   = "status-dot error";
        label.textContent = "Backend Offline";
    }
}

// ================================================================
// RENDER: Overview stat cards
// ================================================================

function renderOverview(overview) {
    if (!overview) return;
    setText("stat-records",     formatNum(overview.total_records));
    setText("stat-packets",     formatNum(overview.total_packets));
    setText("stat-bytes",       formatBytes(overview.total_bytes));
    setText("stat-connections", formatNum(overview.unique_connections));
    setText("stat-src-ips",     formatNum(overview.unique_source_ips));
    setText("stat-dst-ips",     formatNum(overview.unique_destination_ips));

    // Update phase banner count
    const badge = el("dataCountBadge");
    if (badge) badge.textContent = `${formatNum(overview.total_records)} records loaded`;
}

// ================================================================
// RENDER: Protocol distribution bars
// ================================================================

function renderProtocols(protocols) {
    const container = el("protocolChart");
    if (!container) return;

    if (!protocols || protocols.length === 0) {
        container.innerHTML = `<p class="empty-state">No protocol data</p>`;
        return;
    }

    const protoClass = (name) => {
        const n = (name || "").toUpperCase();
        if (n === "TCP")  return "proto-tcp";
        if (n === "UDP")  return "proto-udp";
        if (n === "ICMP") return "proto-icmp";
        return "proto-other";
    };

    container.innerHTML = protocols.map(p => `
        <div class="proto-bar-row">
            <div class="proto-bar-header">
                <span class="proto-name">${escHtml(p.protocol)}</span>
                <span class="proto-meta">
                    ${formatNum(p.count)} flows &bull; ${p.percentage}%
                    &bull; ${formatBytes(p.bytes)}
                </span>
            </div>
            <div class="proto-bar-track">
                <div class="proto-bar-fill ${protoClass(p.protocol)}"
                     style="width:${p.percentage}%"></div>
            </div>
        </div>
    `).join("");
}

// ================================================================
// RENDER: Connection statistics
// ================================================================

function renderConnectionStats(stats) {
    const container = el("connStats");
    if (!container) return;

    if (!stats) {
        container.innerHTML = `<p class="empty-state">No connection data</p>`;
        return;
    }

    const rows = [
        ["Avg Bytes / Connection", formatBytes(stats.avg_bytes_per_connection)],
        ["Max Bytes in Single Flow", formatBytes(stats.max_bytes_connection)],
        ["Avg Packets / Connection", formatNum(Math.round(stats.avg_packets_per_connection))],
        ["Max Packets in Single Flow", formatNum(stats.max_packets_connection)],
        ["Avg Duration (sec)", (stats.avg_duration_sec || 0).toFixed(2) + " s"],
        ["Max Duration (sec)", (stats.max_duration_sec || 0).toFixed(2) + " s"],
        ["Dominant Protocol", stats.top_protocol_by_connections || "—"],
    ];

    container.innerHTML = rows.map(([k, v]) => `
        <div class="conn-stat-row">
            <span class="conn-stat-key">${escHtml(k)}</span>
            <span class="conn-stat-val">${escHtml(String(v))}</span>
        </div>
    `).join("");
}

// ================================================================
// RENDER: Traffic timeline (simple bar chart)
// ================================================================

function renderTimeline(timeline) {
    const container = el("timelineChart");
    if (!container) return;

    if (!timeline || timeline.length === 0) {
        container.innerHTML = `<p class="empty-state">No timeline data</p>`;
        return;
    }

    const maxBytes = Math.max(...timeline.map(t => t.bytes), 1);

    const bars = timeline.map(bucket => {
        const pct = Math.max((bucket.bytes / maxBytes) * 100, 3);
        const label = formatTime(bucket.bucket);
        return `
            <div class="timeline-bar-wrap"
                 data-bucket="${escHtml(bucket.bucket)}"
                 data-records="${bucket.records}"
                 data-packets="${bucket.packets}"
                 data-bytes="${bucket.bytes}">
                <div class="timeline-bar" style="height:${pct}%"></div>
                <div class="timeline-label">${escHtml(label)}</div>
            </div>
        `;
    }).join("");

    container.innerHTML = `<div class="timeline-chart">${bars}</div>`;

    // Attach tooltip events
    container.querySelectorAll(".timeline-bar-wrap").forEach(wrap => {
        const bar = wrap.querySelector(".timeline-bar");
        bar.addEventListener("mousemove", evt => {
            showTooltip(
                `<strong>${wrap.dataset.bucket.replace("T", " ").substring(0,16)}</strong><br>
                 Records: ${formatNum(wrap.dataset.records)}<br>
                 Packets: ${formatNum(wrap.dataset.packets)}<br>
                 Bytes: ${formatBytes(Number(wrap.dataset.bytes))}`,
                evt
            );
        });
        bar.addEventListener("mouseleave", hideTooltip);
        bar.addEventListener("mousemove", positionTooltip);
    });
}

// ================================================================
// RENDER: IP tables (top sources / destinations)
// ================================================================

function renderIpTable(containerId, items, ipKey) {
    const container = el(containerId);
    if (!container) return;

    if (!items || items.length === 0) {
        container.innerHTML = `<p class="empty-state">No data</p>`;
        return;
    }

    const maxBytes = Math.max(...items.map(i => i.bytes), 1);

    const rows = items.map((item, idx) => {
        const pct = Math.round((item.bytes / maxBytes) * 100);
        return `
            <tr>
                <td class="ip-rank">${idx + 1}</td>
                <td class="ip-addr">${escHtml(item[ipKey])}</td>
                <td>${formatNum(item.connections)}</td>
                <td class="ip-bytes">${formatBytes(item.bytes)}</td>
                <td class="ip-bar-cell">
                    <div class="ip-bar-mini">
                        <div class="ip-bar-mini-fill" style="width:${pct}%"></div>
                    </div>
                </td>
            </tr>
        `;
    }).join("");

    container.innerHTML = `
        <table class="ip-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>IP Address</th>
                    <th>Flows</th>
                    <th>Bytes</th>
                    <th></th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

// ================================================================
// Security: HTML escape
// ================================================================

function escHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

// ================================================================
// APP CONTROLLER
// ================================================================

async function checkHealth() {
    try {
        const data = await apiFetch(API.health);
        renderStatus(data);
        return data.status === "ok";
    } catch {
        renderStatus(null);
        return false;
    }
}

async function loadData() {
    const btn = el("loadDataBtn");
    if (btn) { btn.disabled = true; btn.textContent = "Loading..."; }
    hideAlert();

    try {
        const result = await apiFetch(API.load);
        if (!result.success) {
            showAlert(`Load failed: ${result.message}`);
            return false;
        }
        await refreshDashboard();
        return true;
    } catch (err) {
        showAlert(`Could not load data: ${err.message}`);
        return false;
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = "⟳ Load Sample Data"; }
    }
}

async function refreshDashboard() {
    hideAlert();
    try {
        // Fetch all data in parallel for speed
        const [stats, protocols, sources, dests, timeline, connections] = await Promise.all([
            apiFetch(API.stats),
            apiFetch(API.protocols),
            apiFetch(API.sources),
            apiFetch(API.dests),
            apiFetch(API.timeline(currentFreq)),
            apiFetch(API.connections),
        ]);

        renderOverview(stats.overview);
        renderProtocols(protocols);
        renderConnectionStats(connections);
        renderTimeline(timeline);
        renderIpTable("topSourcesList", sources, "source_ip");
        renderIpTable("topDestsList",   dests,   "destination_ip");
    } catch (err) {
        // Graceful: show error but don't crash
        showAlert(`Dashboard refresh failed: ${err.message} — Try loading data first.`);
    }
}

async function refreshTimeline() {
    hideAlert();
    try {
        const timeline = await apiFetch(API.timeline(currentFreq));
        renderTimeline(timeline);
    } catch (err) {
        showAlert(`Timeline refresh failed: ${err.message}`);
    }
}

// ================================================================
// EVENT LISTENERS
// ================================================================

document.addEventListener("DOMContentLoaded", async () => {

    // Alert close button
    const closeBtn = el("alertClose");
    if (closeBtn) closeBtn.addEventListener("click", hideAlert);

    // Load data button
    const loadBtn = el("loadDataBtn");
    if (loadBtn) loadBtn.addEventListener("click", loadData);

    // Timeline frequency buttons
    document.querySelectorAll(".freq-btn").forEach(btn => {
        btn.addEventListener("click", async () => {
            document.querySelectorAll(".freq-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentFreq = btn.dataset.freq;
            await refreshTimeline();
        });
    });

    // Start-up sequence:
    // 1. Check backend health
    const backendOk = await checkHealth();
    if (!backendOk) {
        showAlert("Cannot connect to the Flask backend. Please start the server (python run.py).");
        return;
    }

    // 2. Auto-load data (Flask auto-loads on startup, try to render immediately)
    try {
        await refreshDashboard();
    } catch {
        // No data yet — normal state before first load
    }
});
