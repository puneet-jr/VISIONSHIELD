/**
 * VisionShield Tactical SOC Dashboard Controller
 * Real-time WebSockets, Web Audio Siren Synthesizer, Dynamic Zone Drawing
 */

document.addEventListener('DOMContentLoaded', () => {
    let closing=false, eventSocket;
    const toast = message => { const el = document.getElementById('toast'); el.textContent = message; el.classList.add('visible'); setTimeout(() => el.classList.remove('visible'), 4500); };
    const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
    const request = async (url, options) => { const res = await fetch(url, options); if (!res.ok) throw Error(`Request failed (${res.status})`); return res.json(); };
    // DOM Elements
    const hudFps = document.getElementById('hudFps');
    const hudTracks = document.getElementById('hudTracks');
    const sourceSelect = document.getElementById('sourceSelect');
    const eventsList = document.getElementById('eventsList');
    const logCount = document.getElementById('logCount');

    // Incident Dossier Elements
    const incidentCard = document.getElementById('incidentCard');
    const incidentTitle = document.getElementById('incidentTitle');
    const incidentTime = document.getElementById('incidentTime');
    const incidentSummary = document.getElementById('incidentSummary');
    const incidentSubject = document.getElementById('incidentSubject');
    const incidentAction = document.getElementById('incidentAction');
    const incidentSnapshot = document.getElementById('incidentSnapshot');
    const snapshotPlaceholder = document.querySelector('.snapshot-placeholder');

    // Siren & Audio Synthesizer
    const btnToggleSiren = document.getElementById('btnToggleSiren');
    const sirenStatus = document.getElementById('sirenStatus');
    let sirenEnabled = false;
    let audioCtx = null;

    // Mobile QR Modal
    const btnConnectPhone = document.getElementById('btnConnectPhone');
    const qrModal = document.getElementById('qrModal');
    const btnCloseModal = document.getElementById('btnCloseModal');

    // Interactive Polygon Drawing
    const btnDrawZone = document.getElementById('btnDrawZone');
    const btnClearZones = document.getElementById('btnClearZones');
    const btnSaveZone = document.getElementById('btnSaveZone');
    const zoneHint = document.getElementById('zoneHint');
    const zoneCanvas = document.getElementById('zoneCanvas');
    const videoElement = document.getElementById('videoElement');
    const ctx = zoneCanvas.getContext('2d');

    let cameraOnline = false;
    let isDrawing = false;
    let currentPolygon = []; // Array of normalized [x, y] (0..1)
    let totalEvents = 0;
    const seenEvents = new Set();

    // --- 1. Web Audio API Tactical Siren Synthesizer ---
    function initAudio() {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
    }

    function playTacticalSiren() {
        if (!sirenEnabled) return;
        try {
            initAudio();
            const now = audioCtx.currentTime;
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();

            osc.type = 'sawtooth';
            // Oscillating dual-tone emergency pitch
            osc.frequency.setValueAtTime(880, now);
            osc.frequency.exponentialRampToValueAtTime(440, now + 0.25);
            osc.frequency.exponentialRampToValueAtTime(880, now + 0.5);

            gain.gain.setValueAtTime(0.15, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.5);

            osc.connect(gain);
            gain.connect(audioCtx.destination);

            osc.start(now);
            osc.stop(now + 0.5);
        } catch (e) {
            console.warn("Audio siren error:", e);
        }
    }

    btnToggleSiren.addEventListener('click', () => {
        initAudio();
        sirenEnabled = !sirenEnabled;
        sirenStatus.textContent = sirenEnabled ? "ACTIVE" : "OFF";
        btnToggleSiren.style.borderColor = sirenEnabled ? "#ef4444" : "#f59e0b";
        btnToggleSiren.style.color = sirenEnabled ? "#ef4444" : "#f59e0b";
    });

    // --- 2. Mobile QR Modal Controls ---
    async function refreshPairing() {
        qrModal.style.display = 'flex';
        try {
            const access = document.getElementById('pairRole').value;
            const data = await request('/api/connection?access=' + access);
            const link = document.getElementById('phoneLink'); link.href = data.phone_url; link.textContent = data.remote_ready ? 'Open secure phone camera' : 'Local camera link';
            document.getElementById('phoneQr').src = '/api/qr?access=' + access + '&t=' + Date.now();
            document.getElementById('connectionHint').textContent = data.remote_ready ? 'Works across Wi-Fi and mobile data. Keep this private pairing link with your team.' : 'For phone access across networks, restart with: .\\.venv\\Scripts\\python.exe run.py --remote. HTTPS is required for camera access.';
            document.getElementById('copyPhoneLink').onclick = async () => { try { await navigator.clipboard.writeText(data.phone_url); toast('Pairing link copied'); } catch { toast('Open the pairing link to copy it from the address bar'); } };
        } catch (e) { toast(e.message); }
    }
    btnConnectPhone.addEventListener('click', refreshPairing);
    document.getElementById('pairRole').addEventListener('change', refreshPairing);
    btnCloseModal.addEventListener('click', () => {
        qrModal.style.display = 'none';
    });
    window.addEventListener('click', (e) => {
        if (e.target === qrModal) qrModal.style.display = 'none';
    });

    // --- 3. Input Source Switcher ---
    sourceSelect.addEventListener('change', async () => {
        const selected = sourceSelect.value;
        try {
            await request('/api/source', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source: selected })
            });
        } catch (e) {
            toast("Could not switch camera: " + e.message);
        }
    });

    // --- 4. Interactive Polygon Drawing Engine ---
    function resizeCanvas() {
        const image = imageBounds();
        const parent = document.getElementById('videoContainer').getBoundingClientRect();
        zoneCanvas.style.left = (image.left - parent.left) + 'px';
        zoneCanvas.style.top = (image.top - parent.top) + 'px';
        zoneCanvas.style.width = image.width + 'px';
        zoneCanvas.style.height = image.height + 'px';
        zoneCanvas.width = image.width || 640;
        zoneCanvas.height = image.height || 480;
        renderDrawing();
    }
    const resizeObserver=new ResizeObserver(resizeCanvas);resizeObserver.observe(videoElement);
    window.addEventListener('resize', resizeCanvas);
    videoElement.addEventListener('load', resizeCanvas);
    document.getElementById('btnRemoveZones').addEventListener('click', async () => {
        try { await request('/api/zones', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({zones:[]})}); toast('Saved zones removed for this camera'); }
        catch(e) { toast(e.message); }
    });

    btnDrawZone.addEventListener('click', () => {
        if (!cameraOnline || !videoElement.naturalWidth) { toast('Connect a camera and wait for its live frame before drawing a zone'); return; }
        resizeCanvas();
        isDrawing = true;
        currentPolygon = [];
        zoneCanvas.classList.add('active');
        btnDrawZone.style.display = 'none';
        btnClearZones.style.display = 'inline-block';
        btnSaveZone.style.display = 'inline-block';
        zoneHint.textContent = "Click 3 to 20 points on the video to sketch the exclusion perimeter.";
        renderDrawing();
    });

    btnClearZones.addEventListener('click', () => {
        currentPolygon = [];
        renderDrawing();
    });

    zoneCanvas.addEventListener('click', (e) => {
        if (!isDrawing) return;
        const rect = zoneCanvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        if (currentPolygon.length >= 20) { toast("Use at most 20 vertices"); return; }
        currentPolygon.push([Math.round(x * 1000) / 1000, Math.round(y * 1000) / 1000]);
        renderDrawing();
    });

    btnSaveZone.addEventListener('click', async () => {
        if (currentPolygon.length < 3) {
            toast("Please define at least 3 vertices to form a polygon.");
            return;
        }

        try {
            const saved = await request('/api/zones');
            await request('/api/zones', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    zones: [...saved.zones, {
                        id: "custom_sector_" + Date.now(),
                        name: document.getElementById('zoneName').value.trim() || "Restricted area",
                        points: currentPolygon,
                        severity: "HIGH",
                        color: "#ef4444"
                        ,purpose: document.getElementById('zonePurpose').value,
                        dwell_seconds: Number(document.getElementById('zoneDwell').value)
                    }]
                })
            });

            isDrawing = false;
            zoneCanvas.classList.remove('active');
            btnDrawZone.style.display = 'inline-block';
            btnClearZones.style.display = 'none';
            btnSaveZone.style.display = 'none';
            zoneHint.textContent = "Zone saved for this source. Restricted zones use footpoint and dwell; bag zones create review candidates only.";
            ctx.clearRect(0, 0, zoneCanvas.width, zoneCanvas.height);
        } catch (e) {
            toast("Failed to save zone: " + e);
        }
    });

    function renderDrawing() {
        ctx.clearRect(0, 0, zoneCanvas.width, zoneCanvas.height);
        if (!isDrawing || currentPolygon.length === 0) return;

        ctx.strokeStyle = '#00ffcc';
        ctx.fillStyle = 'rgba(0, 255, 204, 0.2)';
        ctx.lineWidth = 2;

        ctx.beginPath();
        const startX = currentPolygon[0][0] * zoneCanvas.width;
        const startY = currentPolygon[0][1] * zoneCanvas.height;
        ctx.moveTo(startX, startY);

        for (let i = 1; i < currentPolygon.length; i++) {
            const px = currentPolygon[i][0] * zoneCanvas.width;
            const py = currentPolygon[i][1] * zoneCanvas.height;
            ctx.lineTo(px, py);
        }

        if (currentPolygon.length >= 3) {
            ctx.closePath();
            ctx.fill();
        }
        ctx.stroke();

        // Draw vertices
        for (const pt of currentPolygon) {
            ctx.fillStyle = '#a4f2cf';
            ctx.beginPath();
            ctx.arc(pt[0] * zoneCanvas.width, pt[1] * zoneCanvas.height, 5, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    // --- 5. WebSocket Real-Time Event Listener ---
    function connectWebSocket() {
        if(closing)return;
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/events`;
        const ws = new WebSocket(wsUrl);
        eventSocket=ws;

        ws.onopen = () => {
            document.getElementById('connectionState').textContent = 'Event hub connected';
        };

        ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                if (message.type === 'SECURITY_EVENT') {
                    const filters=new FormData(document.getElementById('eventFilters'));
                    if([...filters.values()].some(Boolean))loadHistory();
                    else handleSecurityEvent(message.data, !!message.replay);
                } else if (message.type === 'TACTICAL_INCIDENT') {
                    handleIncidentDossier(message.data);
                }
            } catch (e) {
                console.error("Error parsing WS message:", e);
            }
        };

        ws.onclose = () => {
            document.getElementById('connectionState').textContent = 'Reconnecting to event hub...';
            if(!closing)setTimeout(connectWebSocket, 2000);
        };
    }

    function handleSecurityEvent(data, archived = false) {
        const key = data.event_id || JSON.stringify([data.timestamp, data.alert_type, data.track_id, data.zone_id]);
        if (seenEvents.has(key)) return;
        seenEvents.add(key);
        if (seenEvents.size > 500) seenEvents.delete(seenEvents.values().next().value);
        totalEvents++;
        logCount.textContent = `${totalEvents} Events`;

        // Remove empty state placeholder if present
        const emptyState = eventsList.querySelector('.empty-state');
        if (emptyState) emptyState.remove();

        const item = document.createElement('div');
        const severityClass = (data.severity || 'low').toLowerCase();
        item.className = `feed-item ${severityClass}`;

        item.innerHTML = `
            <div class="item-top">
                <span class="item-type">${escapeHtml(data.alert_type)} · ${escapeHtml(data.severity)} review</span>
                <span class="item-time">${escapeHtml(data.timestamp ? data.timestamp.split(' ')[1] : '')}</span>
            </div>
            <div class="item-desc">${escapeHtml(data.legacy ? 'Legacy observation: predates the current review policy. Open details for the original record.' : data.explanation || '')}</div>
        `;
        item.tabIndex = 0; item.setAttribute('role', 'button');
        item.setAttribute('aria-label', 'Review ' + data.alert_type);
        item.onclick = () => openEvent(data.event_id);
        item.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openEvent(data.event_id); } };

        eventsList.insertBefore(item, eventsList.firstChild);
        if (eventsList.children.length > 40) {
            eventsList.removeChild(eventsList.lastChild);
        }

        if (!archived && data.severity === 'HIGH') {
            playTacticalSiren();
        }
    }

    function handleIncidentDossier(dossier, archived = false) {
        incidentTitle.textContent = (archived ? "ARCHIVED: " : "") + (dossier.incident_title || "INCIDENT REVIEW");
        incidentTime.textContent = dossier.timestamp || "";
        incidentSummary.textContent = dossier.executive_summary || "";
        incidentSubject.textContent = dossier.subject_description || "";
        incidentAction.textContent = archived ? "Historical report. Verify the current live scene before taking action." : (dossier.recommended_action || "");

        if (dossier.snapshot_base64) {
            snapshotPlaceholder.style.display = 'none';
            incidentSnapshot.style.display = 'block';
            incidentSnapshot.src = "data:image/jpeg;base64," + dossier.snapshot_base64;
        }

        playTacticalSiren();
    }

    // --- 6. Telemetry Polling Loop ---
    async function poll() {
        if(closing)return;
        try {
            const data = await request('/api/info');
            cameraOnline = data.camera_online;
            hudFps.textContent = data.camera_online ? `${Math.round(data.preview_fps || 0)} / ${(data.processing_fps || 0).toFixed(1)}` : '0 / 0';
            document.getElementById('videoEmpty').style.display = data.camera_online ? 'none' : 'flex';
            hudTracks.textContent = data.active_tracks || '0';
            document.getElementById('cameraState').textContent = data.source_error || (data.camera_online ? 'Camera connected' : data.recorded ? 'Recording ended' : 'Awaiting camera');
            document.querySelector('.live-status span:last-child').textContent = data.recorded ? 'RECORDED DEMO' : data.camera_online ? 'LIVE MONITORING' : 'CAMERA OFFLINE';
            document.getElementById('statLuggage').textContent = !data.camera_online ? 'Waiting for camera' : !data.detector_ready ? 'Loading detector...' : Object.entries(data.detection_counts || {}).map(([name,count])=>`${name}: ${count}`).join(' / ') || 'No confident detections';
            document.getElementById('hudEngine').textContent = data.detector_error ? 'Detector unavailable' : (data.detector_ready ? 'Motion + appearance tracks' : 'Detector idle');
            document.getElementById('inferenceTime').textContent = data.camera_online && data.detector_ready ? data.inference_ms + ' ms' : 'Waiting for camera';
            document.getElementById('modelScope').textContent = `${data.model} / ${data.backend} / ${data.enabled_classes.length} classes / ${data.input_size}px`;
            document.getElementById('videoContainer').classList.toggle('camera-live', data.camera_online);
            document.getElementById('reportMode').textContent = data.report_mode;
            document.getElementById('statZones').textContent = data.zones.length + ' zones';
            if ([...sourceSelect.options].some(o => o.value === data.source_path)) sourceSelect.value = data.source_path;
        } catch { document.getElementById('cameraState').textContent = 'Server unavailable'; }
        setTimeout(poll, 1500);
    }
    poll();
    request('/api/incidents?limit=1').then(data => { if (data.incidents[0]?.report_mode) handleIncidentDossier(data.incidents[0], true); }).catch(() => {});
    document.addEventListener('keydown', event => { if (event.key === 'Escape') { qrModal.style.display = 'none'; isDrawing = false; zoneCanvas.classList.remove('active'); currentPolygon = []; renderDrawing(); btnDrawZone.style.display = 'inline-block'; btnClearZones.style.display = btnSaveZone.style.display = 'none'; } });

    // Camera JPEG delivery never waits for inference. Boxes are separate, age-limited metadata.
    const trackCanvas = document.getElementById('trackingCanvas');
    const trackContext = trackCanvas.getContext('2d');
    let telemetry = null, receivedAt = 0;
    async function pollTracks() {
        if(closing)return;
        try { telemetry = await request('/api/telemetry'); receivedAt = performance.now(); }
        catch { telemetry = null; }
        setTimeout(pollTracks, 100);
    }
    function drawTracks() {
        if(closing)return;
        const image = imageBounds();
        const parent = document.getElementById('videoContainer').getBoundingClientRect();
        if (trackCanvas.width !== Math.round(image.width) || trackCanvas.height !== Math.round(image.height)) {
            trackCanvas.width = Math.round(image.width); trackCanvas.height = Math.round(image.height);
        }
        Object.assign(trackCanvas.style, {left:(image.left-parent.left)+'px', top:(image.top-parent.top)+'px', width:image.width+'px', height:image.height+'px'});
        const c = trackContext, w = trackCanvas.width, h = trackCanvas.height;
        c.clearRect(0,0,w,h);
        if (telemetry) {
            for (const zone of telemetry.zones || []) {
                c.beginPath(); zone.points.forEach((p,i) => i ? c.lineTo(p[0]*w,p[1]*h) : c.moveTo(p[0]*w,p[1]*h)); c.closePath();
                const high = !telemetry.stale && (telemetry.confirmed_zones || []).includes(zone.id) && zone.severity === 'HIGH';
                c.strokeStyle = high ? '#f87171' : '#7dd3fc'; c.fillStyle = high ? '#ef444422' : '#7dd3fc0a'; c.lineWidth = 2; c.fill(); c.stroke();
                if(zone.points.length) { c.font='11px system-ui'; c.fillStyle='#b5dce9'; c.fillText(zone.name,zone.points[0][0]*w+5,zone.points[0][1]*h+15); }
            }
            const age = (telemetry.age_ms || 0) + performance.now() - receivedAt;
            if (!telemetry.stale && age < 850 && telemetry.width && telemetry.height) {
                for (const t of telemetry.tracks || []) {
                    const scaleX=w/telemetry.width, scaleY=h/telemetry.height;
                    const prediction = Math.min(age/1000,.15);
                    const dx=(t.motion?.[0] || 0)*prediction, dy=(t.motion?.[1] || 0)*prediction;
                    const x=Math.max(0,(t.box[0]+dx)*scaleX), y=Math.max(0,(t.box[1]+dy)*scaleY);
                    const bw=(t.box[2]-t.box[0])*scaleX, bh=(t.box[3]-t.box[1])*scaleY;
                    const colour=t.threat_level==='HIGH'?'#f87171':t.threat_level==='MEDIUM'?'#fbbf24':'#a4f2cf';
                    c.strokeStyle=colour;c.lineWidth=2;c.strokeRect(x,y,bw,bh);
                    const label=`${t.label.toUpperCase()} · ${t.person_id || 'OBJECT'} · T${t.track_id} · ${Math.round((t.confidence || 0)*100)}%`;
                    c.font='11px system-ui';const tw=c.measureText(label).width+10;
                    c.fillStyle='#0b141ee8';c.fillRect(x,Math.max(0,y-22),tw,20);c.fillStyle=colour;c.fillText(label,x+5,Math.max(14,y-8));
                    if(t.active_alert) { c.fillStyle=colour;c.fillText(t.active_alert,x+4,Math.min(h-8,y+bh+15)); }
                }
            }
        }
        setTimeout(drawTracks, 1000/24);
    }
    pollTracks(); drawTracks();

    function imageBounds() {
        const rect = videoElement.getBoundingClientRect();
        const ratio = videoElement.naturalWidth / videoElement.naturalHeight || 4/3;
        let width = rect.width, height = width/ratio;
        if (height > rect.height) { height=rect.height; width=height*ratio; }
        return {left:rect.left+(rect.width-width)/2,top:rect.top+(rect.height-height)/2,width,height};
    }

    let selectedEvent;
    async function openEvent(id) {
        if (!id) return;
        try {
            selectedEvent=id;
            const data=await request('/api/events/'+id);
            const event=data.event || data;
            document.getElementById('eventTitle').textContent=event.alert_type+' · '+event.status;
            document.getElementById('eventDetails').textContent=JSON.stringify(event,null,2);
            const evidence=document.getElementById('eventEvidence');
            evidence.hidden=!event.evidence_url;
            evidence.onerror=()=>{ evidence.hidden=true; };
            if(event.evidence_url)evidence.src=event.evidence_url;
            document.getElementById('eventDialog').showModal();
        } catch(e) { toast(e.message); }
    }
    document.getElementById('closeEvent').onclick=()=>document.getElementById('eventDialog').close();
    document.querySelectorAll('[data-review]').forEach(button=>button.onclick=async()=>{
        try { await request('/api/events/'+selectedEvent,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({status:button.dataset.review})}); await openEvent(selectedEvent); await loadHistory(); }
        catch(e){toast(e.message);}
    });
    document.getElementById('deleteEvent').onclick=async()=>{
        try { await request('/api/events/'+selectedEvent,{method:'DELETE'}); document.getElementById('eventDialog').close(); await loadHistory(); }
        catch(e){toast(e.message);}
    };
    async function loadHistory() {
        const query=new URLSearchParams();
        for (const [key,value] of new FormData(document.getElementById('eventFilters'))) {
            if(value)query.set(key==='camera_id'?'camera':key,['start','end'].includes(key)?new Date(value).toISOString():value);
        }
        try { const data=await request('/api/events?limit=100&'+query); seenEvents.clear();totalEvents=0;eventsList.replaceChildren();logCount.textContent='0 Events';for(const event of [...data.events].reverse())handleSecurityEvent(event,true); }
        catch(e){toast(e.message);}
    }
    document.getElementById('eventFilters').onsubmit=e=>{e.preventDefault();loadHistory();};
    request('/api/recordings').then(data=>{for(const r of data.recordings)sourceSelect.add(new Option('RECORDED: '+r.name,r.source));}).catch(()=>{});
    async function refreshInsights() {
        try {
            const data=await request('/api/insights'), model=data.model;
            document.getElementById('insightsSummary').textContent=`${model.checkpoint} | ${model.backend} | ${model.training_status} | ${model.supported_classes.length} classes. Identity: camera/session colour association, no learned ReID. Assessment: rules_v1, human review. CPU ${data.metrics.resource_samples.at(-1)?.cpu_percent_one_core_100 ?? '—'}% (100% per core), RAM ${data.metrics.resource_samples.at(-1)?.rss_mb ?? '—'} MB. Inference skips ${data.metrics.inference_drop_percent}% of captured frames by design.`;
            document.getElementById('insightsRaw').textContent=JSON.stringify(data,null,2);
        }catch(e){toast(e.message);}
    }
    document.getElementById('refreshInsights').onclick=refreshInsights;
    document.getElementById('runDiagnostics').onclick=async()=>{try{document.getElementById('diagnosticsResult').textContent=JSON.stringify(await request('/api/diagnostics'),null,2);}catch(e){toast(e.message);}};
    videoElement.onclick=e=>{
        if(!telemetry || telemetry.stale)return;
        const r=imageBounds(), x=(e.clientX-r.left)/r.width*telemetry.width,y=(e.clientY-r.top)/r.height*telemetry.height;
        const track=(telemetry.tracks||[]).find(t=>x>=t.box[0] && x<=t.box[2] && y>=t.box[1] && y<=t.box[3]);
        if(track)document.getElementById('trackDetails').textContent=JSON.stringify(track);
    };
    // One decoded frame and one latest pending frame: never build a playback queue.
    let preview, reconnect, frameMetadata, pendingFrame, decoding=false, objectUrl;
    let clockOffset=0, clockUncertainty=60000, displayed=0, dropped=0, latency=0, lastSequence=-1;
    const metricStart=performance.now(), clientId=crypto.randomUUID();
    async function syncClock() {
        let best=Infinity;
        for(let i=0;i<3;i++){const before=Date.now();const result=await request('/api/time');const after=Date.now();if(after-before<best){best=after-before;clockOffset=result.server_time_ms-(before+after)/2;clockUncertainty=best/2;}}
    }
    function displayLatest() {
        if(decoding || !pendingFrame)return;
        const {blob,metadata}=pendingFrame;pendingFrame=null;decoding=true;
        if(objectUrl)URL.revokeObjectURL(objectUrl);
        objectUrl=URL.createObjectURL(blob);
        videoElement.onload=()=>{
            displayed++;latency=Math.max(0,Date.now()+clockOffset-(metadata.captured_at||metadata.received_at)*1000);
            if(lastSequence>=0 && metadata.sequence>lastSequence)dropped+=Math.max(0,metadata.sequence-lastSequence-1);
            lastSequence=metadata.sequence;decoding=false;resizeCanvas();displayLatest();
        };
        videoElement.onerror=()=>{decoding=false;displayLatest();};
        videoElement.src=objectUrl;
    }
    function connectPreview() {
        if(closing)return;
        preview=new WebSocket(`${location.protocol==='https:'?'wss:':'ws:'}//${location.host}/ws/preview`);
        preview.binaryType='blob';
        preview.onmessage=e=>{if(typeof e.data==='string'){frameMetadata=JSON.parse(e.data);return;}pendingFrame={blob:e.data,metadata:frameMetadata||{}};displayLatest();};
        preview.onclose=e=>{if(!closing && e.code!==1008)reconnect=setTimeout(connectPreview,1500);else if(e.code===1008)toast('Preview access expired. Pair this browser again.');};
    }
    const metricsTimer=setInterval(()=>{
        if(!displayed)return;
        const payload={client_id:clientId,displayed_fps:displayed/Math.max(1,(performance.now()-metricStart)/1000),display_latency_ms:Math.min(60000,latency),dropped_frames:Math.min(10000000,dropped),clock_uncertainty_ms:clockUncertainty};
        document.getElementById('displayMetrics').textContent=`Displayed ${payload.displayed_fps.toFixed(1)} FPS · capture-to-display estimate ${Math.round(latency)} ms ±${Math.round(clockUncertainty)} ms clock error · ${dropped} preview frames skipped. Hardware sensor delay is unmeasured.`;
        request('/api/client-metrics',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}).catch(()=>{});
    },3000);
    window.addEventListener('pagehide',()=>{closing=true;clearTimeout(reconnect);clearInterval(metricsTimer);preview?.close();eventSocket?.close();resizeObserver.disconnect();if(objectUrl)URL.revokeObjectURL(objectUrl);});
    syncClock().catch(()=>{}).finally(connectPreview);
    loadHistory();refreshInsights();
    // Initial Execution
    connectWebSocket();
});
