// Leaflet Map Initialization
const map = L.map('map', {
    center: [0, 0],
    zoom: 2,
    zoomControl: false,
    attributionControl: false
});

// Dark mode tile layer (Stamen Toner or similar)
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 19
}).addTo(map);

// Satellite Marker
const satIcon = L.divIcon({
    className: 'sat-marker',
    html: '<div style="width:12px; height:12px; background:#00d2ff; border:2px solid #fff; border-radius:50%; box-shadow:0 0 10px #00d2ff;"></div>',
    iconSize: [12, 12]
});

const marker = L.marker([0, 0], { icon: satIcon }).addTo(map);
const groundTrack = L.polyline([], { color: '#00d2ff', weight: 1, opacity: 0.6 }).addTo(map);

// WebSocket Management
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

const consoleEl = document.getElementById('console');
const pcCard = document.getElementById('pc-card');

function logToConsole(msg) {
    const now = new Date();
    const ts = now.toISOString().split('T')[1].split('Z')[0];
    consoleEl.innerHTML += `[${ts}] ${msg}<br>`;
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

ws.onopen = () => {
    logToConsole('WebSocket Connected to Ground System');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // Update Telemetry Displays
    if (data.alt !== undefined) document.getElementById('alt-val').innerText = data.alt.toFixed(2);
    if (data.vel !== undefined) document.getElementById('vel-val').innerText = data.vel.toFixed(3);
    
    if (data.q0 !== undefined) {
        document.getElementById('q0-val').innerText = data.q0.toFixed(4);
        document.getElementById('q1-val').innerText = data.q1.toFixed(4);
        document.getElementById('q2-val').innerText = data.q2.toFixed(4);
        document.getElementById('q3-val').innerText = data.q3.toFixed(4);
    }
    
    if (data.pc !== undefined) {
        document.getElementById('pc-val').innerText = data.pc.toExponential(3);
        // Highlight card if PC is high
        if (data.pc > 1e-4) {
            pcCard.style.animation = 'danger-pulse 1s infinite alternate';
        } else {
            pcCard.style.animation = 'none';
        }
    }

    // Update Map
    if (data.lat !== undefined && data.lon !== undefined) {
        const pos = [data.lat, data.lon];
        marker.setLatLng(pos);
        groundTrack.addLatLng(pos);
        
        // Keep focus if moving
        // map.panTo(pos);
    }
};

ws.onclose = () => {
    logToConsole('WebSocket Connection Closed');
};

// Periodic UTC clock
setInterval(() => {
    const now = new Date();
    document.getElementById('utc-time').innerText = now.toISOString().replace('T', ' ').split('.')[0] + ' UTC';
}, 1000);

// Style for danger pulse
const style = document.createElement('style');
style.innerHTML = `
    @keyframes danger-pulse {
        from { border-color: rgba(239, 68, 68, 0.4); box-shadow: 0 0 5px rgba(239, 68, 68, 0.2); }
        to { border-color: rgba(239, 68, 68, 1); box-shadow: 0 0 15px rgba(239, 68, 68, 0.5); }
    }
`;
document.head.appendChild(style);
