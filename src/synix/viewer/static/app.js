/* Viewer — Frontend Logic */

const state = {
    loaded: false,
    layers: [],
    currentLayer: null,
    currentLabel: null,
    listPage: 1,
    listSort: 'date',
    listOrder: 'desc',
    mode: 'browse', // 'browse' | 'search'
    searchQuery: '',
    searchLayer: '',
    searchPage: 1,
};

// --- API helpers ---

async function api(path) {
    const resp = await fetch(path);
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`API error ${resp.status}: ${text}`);
    }
    return resp.json();
}

// --- Initialization ---

async function init() {
    // Wire up events
    document.getElementById('search-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') doSearch();
    });
    document.getElementById('list-sort').addEventListener('change', onSortChange);

    // Hash routing
    window.addEventListener('hashchange', handleHash);

    // Check if server already has data loaded
    try {
        const status = await api('/api/status');
        if (status.loaded) {
            state.loaded = true;
            document.getElementById('logo').textContent = status.title;
            document.title = status.title;
            await loadReleases();
            await loadData();
            return;
        }
    } catch (err) {
        console.error('Failed to check status:', err);
    }

    // No data loaded — show welcome state
    showWelcome();
}

function showWelcome() {
    document.getElementById('layer-nav').innerHTML = '';
    document.getElementById('list-items').innerHTML = '';
    document.getElementById('list-pagination').style.display = 'none';
    document.getElementById('reader-body').innerHTML = `
        <div class="empty-state" style="flex-direction:column;gap:12px;text-align:center">
            <div style="font-size:32px;color:var(--neon);font-family:'JetBrains Mono',monospace;text-shadow:0 0 20px var(--neon-glow)">VIEWER</div>
            <div style="color:var(--text-muted);max-width:400px;line-height:1.6">
                Loading...
            </div>
        </div>
    `;
}

async function loadData() {
    try {
        state.layers = await api('/api/layers');
    } catch (err) {
        console.error('Failed to load layers:', err);
        document.getElementById('layer-nav').innerHTML = '<div style="padding:12px;color:var(--text-muted)">Failed to load layers</div>';
        return;
    }

    renderLayerNav();
    populateSearchFilter();
    document.getElementById('list-pagination').style.display = '';

    // Reset view state
    state.currentLayer = null;
    state.currentLabel = null;
    state.mode = 'browse';
    document.getElementById('reader-header').style.display = 'none';
    document.getElementById('lineage-panel').style.display = 'none';
    document.getElementById('reader-body').innerHTML = '<div class="empty-state">Select an artifact to view</div>';

    handleHash();
}

// --- Hash routing ---

function handleHash() {
    if (!state.loaded) return;

    const hash = location.hash.slice(1);
    if (!hash) {
        if (state.layers.length > 0) {
            selectLayer(state.layers[0].name);
        }
        return;
    }

    const parts = hash.split('/');
    if (parts[0] === 'layer' && parts[1]) {
        selectLayer(parts[1]);
    } else if (parts[0] === 'artifact' && parts[1]) {
        const label = decodeURIComponent(parts.slice(1).join('/'));
        loadArtifact(label);
    } else if (parts[0] === 'search' && parts[1]) {
        const query = decodeURIComponent(parts.slice(1).join('/'));
        document.getElementById('search-input').value = query;
        state.searchQuery = query;
        state.searchPage = 1;
        doSearch();
    }
}

function setHash(hash) {
    if (location.hash.slice(1) !== hash) {
        history.pushState(null, '', '#' + hash);
    }
}

// --- Layer navigation ---

function renderLayerNav() {
    const nav = document.getElementById('layer-nav');
    nav.innerHTML = state.layers.map(layer => `
        <button class="layer-btn" data-layer="${layer.name}" onclick="selectLayer('${layer.name}')">
            <span style="display:flex;align-items:center">
                <span class="layer-dot" style="background:var(--layer-${layer.level})"></span>
                ${layer.name}
            </span>
            <span class="count">${layer.count}</span>
        </button>
    `).join('');
}

function populateSearchFilter() {
    const sel = document.getElementById('search-layer-filter');
    sel.innerHTML = '<option value="">All layers</option>' +
        state.layers.map(l => `<option value="${l.name}">${l.name}</option>`).join('');
}

function selectLayer(name) {
    state.currentLayer = name;
    state.listPage = 1;
    state.mode = 'browse';

    document.querySelectorAll('.layer-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.layer === name);
    });

    document.getElementById('back-to-browse').style.display = 'none';
    setHash(`layer/${name}`);
    loadArtifactList();
}

// --- Metadata badge rendering ---

function renderMetaBadges(metadata) {
    const skip = new Set(['title', 'layer_name', 'layer_level', 'created_at']);
    return Object.entries(metadata || {})
        .filter(([k, v]) => !skip.has(k) && v != null && v !== '')
        .map(([k, v]) => `<span class="meta-badge">${k}: ${v}</span>`)
        .join('');
}

// --- Artifact list ---

async function loadArtifactList() {
    const listItems = document.getElementById('list-items');
    listItems.innerHTML = '<div style="padding:20px;color:var(--text-muted)">Loading...</div>';

    try {
        const data = await api(
            `/api/artifacts?layer=${state.currentLayer}&page=${state.listPage}&per_page=50&sort=${state.listSort}&order=${state.listOrder}`
        );
        renderArtifactList(data);
    } catch (err) {
        listItems.innerHTML = `<div style="padding:20px;color:var(--text-muted)">Error: ${err.message}</div>`;
    }
}

function renderArtifactList(data) {
    const listItems = document.getElementById('list-items');

    if (data.items.length === 0) {
        listItems.innerHTML = '<div style="padding:20px;color:var(--text-muted)">No artifacts found</div>';
    } else {
        listItems.innerHTML = data.items.map(item => `
            <div class="list-item ${item.label === state.currentLabel ? 'active' : ''}"
                 onclick="loadArtifact('${escapeAttr(item.label)}')" title="${escapeHtml(item.title)}">
                <div class="item-title">${escapeHtml(item.title)}</div>
                <div class="item-meta">
                    ${item.date ? `<span>${item.date}</span>` : ''}
                    ${renderMetaBadges(item.metadata)}
                </div>
            </div>
        `).join('');
    }

    document.querySelector('#list-header .title').textContent =
        `${state.currentLayer} (${data.total})`;

    const totalPages = Math.ceil(data.total / data.per_page);
    document.getElementById('page-info').textContent = `${data.page} / ${totalPages}`;
    document.getElementById('prev-page').disabled = data.page <= 1;
    document.getElementById('next-page').disabled = data.page >= totalPages;
}

function onSortChange(e) {
    const [sort, order] = e.target.value.split('-');
    state.listSort = sort;
    state.listOrder = order;
    state.listPage = 1;
    if (state.mode === 'browse') {
        loadArtifactList();
    }
}

function prevPage() {
    if (state.mode === 'search') {
        state.searchPage = Math.max(1, state.searchPage - 1);
        doSearch();
    } else {
        state.listPage = Math.max(1, state.listPage - 1);
        loadArtifactList();
    }
}

function nextPage() {
    if (state.mode === 'search') {
        state.searchPage++;
        doSearch();
    } else {
        state.listPage++;
        loadArtifactList();
    }
}

// --- Artifact reader ---

async function loadArtifact(label) {
    state.currentLabel = label;
    setHash(`artifact/${encodeURIComponent(label)}`);

    const header = document.getElementById('reader-header');
    const body = document.getElementById('reader-body');
    const lineage = document.getElementById('lineage-panel');

    header.style.display = 'none';
    body.innerHTML = '<div class="empty-state">Loading...</div>';
    lineage.style.display = 'none';

    document.querySelectorAll('.list-item').forEach(el => {
        el.classList.toggle('active', el.getAttribute('onclick')?.includes(label));
    });

    try {
        const [artifact, lineageData] = await Promise.all([
            api(`/api/artifact/${encodeURIComponent(label)}`),
            api(`/api/lineage/${encodeURIComponent(label)}`),
        ]);

        renderArtifact(artifact);
        renderLineage(lineageData);
        renderMeta(artifact, lineageData);

        const info = state.layers.find(l => l.name === (artifact.metadata?.layer_name));
        if (info && state.currentLayer !== info.name && state.mode === 'browse') {
            state.currentLayer = info.name;
            document.querySelectorAll('.layer-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.layer === info.name);
            });
        }
    } catch (err) {
        body.innerHTML = `<div class="empty-state" style="color:var(--neon)">Error: ${err.message}</div>`;
    }
}

// --- Pluggable renderers ---

const RENDERERS = {
    'transcript': renderTranscript,
    'default': (content) => `<div class="markdown-content">${marked.parse(content)}</div>`,
};

function renderArtifact(artifact) {
    // Stash raw content for copy button
    state.currentContent = artifact.content || '';

    const header = document.getElementById('reader-header');
    const body = document.getElementById('reader-body');
    const meta = artifact.metadata || {};
    const level = meta.layer_level ?? 0;

    header.style.display = 'flex';
    header.querySelector('.artifact-title').textContent = meta.title || artifact.label;
    header.querySelector('.artifact-meta').innerHTML = `
        <span class="item-badge layer-bg-${level}">${artifact.artifact_type || 'unknown'}</span>
        ${renderMetaBadges(meta)}
    `;

    const content = artifact.content || '';
    const renderer = RENDERERS[artifact.artifact_type] || RENDERERS['default'];
    body.innerHTML = renderer(content, artifact.metadata);

    body.scrollTop = 0;
}

async function copyContent() {
    if (!state.currentContent) return;
    const btn = document.getElementById('copy-btn');
    try {
        await navigator.clipboard.writeText(state.currentContent);
        btn.textContent = 'Copied';
        setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
    } catch (err) {
        btn.textContent = 'Failed';
        setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
    }
}

function renderLineage(data) {
    const panel = document.getElementById('lineage-panel');

    if (data.parents.length === 0 && data.children.length === 0) {
        panel.style.display = 'none';
        return;
    }

    panel.style.display = 'block';
    let html = '';

    if (data.parents.length > 0) {
        const collapsed = data.parents.length > 5;
        html += `
            <div class="lineage-section">
                <div class="lineage-label" ${collapsed ? 'style="cursor:pointer" onclick="toggleLineageSection(this)"' : ''}>
                    Parents (${data.parents.length})${collapsed ? ' <span class="lineage-toggle">Show</span>' : ''}
                </div>
                <div class="lineage-chips ${collapsed ? 'lineage-collapsed' : ''}">
                    ${data.parents.map(p => lineageChip(p)).join('')}
                </div>
            </div>
        `;
    }

    if (data.children.length > 0) {
        const collapsed = data.children.length > 5;
        html += `
            <div class="lineage-section">
                <div class="lineage-label" ${collapsed ? 'style="cursor:pointer" onclick="toggleLineageSection(this)"' : ''}>
                    Children (${data.children.length})${collapsed ? ' <span class="lineage-toggle">Show</span>' : ''}
                </div>
                <div class="lineage-chips ${collapsed ? 'lineage-collapsed' : ''}">
                    ${data.children.map(c => lineageChip(c)).join('')}
                </div>
            </div>
        `;
    }

    panel.innerHTML = html;
}

function toggleLineageSection(labelEl) {
    const chips = labelEl.nextElementSibling;
    const toggle = labelEl.querySelector('.lineage-toggle');
    const isCollapsed = chips.classList.toggle('lineage-collapsed');
    if (toggle) toggle.textContent = isCollapsed ? 'Show' : 'Hide';
}

function lineageChip(item) {
    return `<span class="lineage-chip layer-border-${item.level}"
                 onclick="loadArtifact('${escapeAttr(item.label)}')"
                 title="${escapeHtml(item.label)}">
                ${escapeHtml(item.title || item.label)}
            </span>`;
}

// --- Meta panel ---

function toggleMeta() {
    document.getElementById('meta-panel').classList.toggle('open');
}

function renderMeta(artifact, lineageData) {
    const content = document.getElementById('meta-content');
    const meta = { ...artifact };
    delete meta.content;

    let html = '<pre>' + escapeHtml(JSON.stringify(meta, null, 2)) + '</pre>';
    content.innerHTML = html;
}

// --- Search ---

async function doSearch() {
    const query = document.getElementById('search-input').value.trim();
    if (!query) return;

    state.mode = 'search';
    state.searchQuery = query;
    state.searchLayer = document.getElementById('search-layer-filter').value;

    setHash(`search/${encodeURIComponent(query)}`);

    const listItems = document.getElementById('list-items');
    listItems.innerHTML = '<div style="padding:20px;color:var(--text-muted)">Searching...</div>';
    document.getElementById('back-to-browse').style.display = 'block';
    document.querySelector('#list-header .title').textContent = 'Search Results';

    try {
        const params = new URLSearchParams({
            q: query,
            page: state.searchPage,
            per_page: 20,
        });
        if (state.searchLayer) params.set('layer', state.searchLayer);

        const data = await api(`/api/search?${params}`);
        renderSearchResults(data);
    } catch (err) {
        listItems.innerHTML = `<div style="padding:20px;color:var(--text-muted)">Search error: ${err.message}</div>`;
    }
}

function renderSearchResults(data) {
    const listItems = document.getElementById('list-items');

    if (data.error) {
        listItems.innerHTML = `<div style="padding:20px;color:var(--text-muted)">Search error: ${data.error}</div>`;
        return;
    }

    if (data.items.length === 0) {
        listItems.innerHTML = '<div style="padding:20px;color:var(--text-muted)">No results found</div>';
    } else {
        listItems.innerHTML = data.items.map(item => {
            const meta = item.metadata || {};
            const title = meta.title || item.label;
            const date = meta.date || '';
            return `
                <div class="list-item" onclick="loadArtifact('${escapeAttr(item.label)}')">
                    <div class="item-title">${escapeHtml(title)}</div>
                    <div class="item-meta">
                        <span class="meta-badge">${escapeHtml(item.layer)}</span>
                        ${date ? `<span>${escapeHtml(date)}</span>` : ''}
                    </div>
                    ${item.snippet ? `<div class="search-snippet">${item.snippet}</div>` : ''}
                </div>
            `;
        }).join('');
    }

    document.querySelector('#list-header .title').textContent =
        `Search: "${state.searchQuery}" (${data.total} results)`;

    const totalPages = Math.ceil(data.total / data.per_page) || 1;
    document.getElementById('page-info').textContent = `${data.page} / ${totalPages}`;
    document.getElementById('prev-page').disabled = data.page <= 1;
    document.getElementById('next-page').disabled = data.page >= totalPages;
}

function backToBrowse() {
    state.mode = 'browse';
    document.getElementById('back-to-browse').style.display = 'none';
    document.getElementById('search-input').value = '';
    if (state.currentLayer) {
        selectLayer(state.currentLayer);
    } else if (state.layers.length > 0) {
        selectLayer(state.layers[0].name);
    }
}

// --- Transcript renderer ---

function renderTranscript(raw, metadata) {
    // Parse "user:" and "assistant:" turns from raw transcript text
    const turns = [];
    const lines = raw.split('\n');
    let currentRole = null;
    let currentLines = [];

    for (const line of lines) {
        // Detect turn boundaries
        if (/^user:\s*/i.test(line)) {
            if (currentRole) turns.push({ role: currentRole, text: currentLines.join('\n').trim() });
            currentRole = 'user';
            currentLines = [line.replace(/^user:\s*/i, '')];
        } else if (/^assistant:\s*/i.test(line)) {
            if (currentRole) turns.push({ role: currentRole, text: currentLines.join('\n').trim() });
            currentRole = 'assistant';
            currentLines = [line.replace(/^assistant:\s*/i, '')];
        } else {
            currentLines.push(line);
        }
    }
    if (currentRole) turns.push({ role: currentRole, text: currentLines.join('\n').trim() });

    if (turns.length === 0) {
        return `<div class="raw-content">${escapeHtml(raw)}</div>`;
    }

    const html = turns.map(t => {
        const cls = t.role === 'user' ? 'chat-user' : 'chat-assistant';
        const label = t.role === 'user' ? 'You' : 'AI';
        // Render assistant turns as markdown, user turns as plain text
        const body = t.role === 'assistant'
            ? marked.parse(t.text)
            : escapeHtml(t.text).replace(/\n/g, '<br>');
        return `<div class="chat-turn ${cls}">
            <div class="chat-role">${label}</div>
            <div class="chat-bubble">${body}</div>
        </div>`;
    }).join('');

    return `<div class="chat-transcript">${html}</div>`;
}

// --- Utility ---

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function escapeAttr(str) {
    if (!str) return '';
    return str.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
}

// --- Release switching ---

async function loadReleases() {
    try {
        const data = await api('/api/releases');
        if (data.releases.length > 0) {
            const box = document.getElementById('release-box');
            const sel = document.getElementById('release-select');
            box.style.display = 'block';
            sel.innerHTML = data.releases.map(name =>
                `<option value="${escapeHtml(name)}" ${name === data.current ? 'selected' : ''}>${escapeHtml(name)}</option>`
            ).join('');
            sel.addEventListener('change', switchRelease);
        }
    } catch (err) {
        console.error('Failed to load releases:', err);
    }
}

async function switchRelease() {
    const sel = document.getElementById('release-select');
    const name = sel.value;

    // Show loading state
    document.getElementById('reader-body').innerHTML = '<div class="empty-state">Switching release...</div>';
    document.getElementById('layer-nav').innerHTML = '<div style="padding:12px;color:var(--text-muted)">Loading...</div>';

    try {
        const result = await fetch('/api/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ release: name }),
        });
        if (!result.ok) {
            const err = await result.json();
            throw new Error(err.error || 'Switch failed');
        }
        // Reload everything
        await loadData();
    } catch (err) {
        console.error('Failed to switch release:', err);
        document.getElementById('reader-body').innerHTML =
            `<div class="empty-state" style="color:var(--neon)">Error: ${escapeHtml(err.message)}</div>`;
    }
}

// --- Boot ---

document.addEventListener('DOMContentLoaded', () => {
    marked.use({
        renderer: {
            html(token) {
                // Escape raw HTML blocks to prevent XSS
                return escapeHtml(typeof token === 'string' ? token : token.text);
            }
        }
    });
    init();
});
