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
    activeTab: 'browse',
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

    // Event delegation for clickable items (no inline onclick handlers)
    document.getElementById('list-items').addEventListener('click', (e) => {
        // Handle chunk group header toggles
        const groupHeader = e.target.closest('.chunk-group-header');
        if (groupHeader) {
            const groupId = groupHeader.dataset.groupId;
            const isExpanded = groupHeader.dataset.expanded === 'true';
            groupHeader.dataset.expanded = isExpanded ? 'false' : 'true';
            const items = groupHeader.parentElement.querySelector(`.chunk-group-items[data-group-id="${groupId}"]`);
            if (items) items.dataset.collapsed = isExpanded ? 'true' : 'false';
            return;
        }
        const item = e.target.closest('[data-label]');
        if (item) loadArtifact(item.dataset.label);
    });
    document.getElementById('lineage-panel').addEventListener('click', (e) => {
        // Handle collapsible section toggles
        const label = e.target.closest('.lineage-collapsible');
        if (label) { toggleLineageSection(label); return; }
        // Handle chip clicks
        const chip = e.target.closest('[data-label]');
        if (chip) loadArtifact(chip.dataset.label);
    });
    document.getElementById('layer-nav').addEventListener('click', (e) => {
        const btn = e.target.closest('[data-layer]');
        if (btn) selectLayer(btn.dataset.layer);
    });
    document.getElementById('prev-page').addEventListener('click', prevPage);
    document.getElementById('next-page').addEventListener('click', nextPage);
    document.getElementById('back-to-browse-btn').addEventListener('click', backToBrowse);
    document.getElementById('meta-toggle').addEventListener('click', toggleMeta);
    document.getElementById('copy-btn').addEventListener('click', copyContent);

    // Tab switching
    document.getElementById('tab-bar').addEventListener('click', (e) => {
        const btn = e.target.closest('.tab');
        if (!btn) return;
        switchTab(btn.dataset.tab);
    });

    // Hash routing
    window.addEventListener('hashchange', handleHash);

    // Load data from server (lazy — no preloading needed)
    try {
        const status = await api('/api/status');
        if (status.loaded) {
            state.loaded = true;
            if (status.workspace) {
                document.getElementById('logo').textContent = status.workspace;
                document.title = status.workspace + ' — ' + status.title;
            } else {
                document.getElementById('logo').textContent = status.title;
                document.title = status.title;
            }
            await loadReleases();
            await loadData();
            return;
        }
    } catch (err) {
        console.error('Failed to check status:', err);
    }
    showError('Failed to connect to server');
}

function showError(message) {
    document.getElementById('layer-nav').innerHTML = '';
    document.getElementById('list-items').innerHTML = '';
    document.getElementById('list-pagination').style.display = 'none';
    document.getElementById('reader-body').innerHTML = `
        <div class="empty-state" style="color:var(--accent-error)">${escapeHtml(message)}</div>
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
        <button class="layer-btn" data-layer="${escapeHtml(layer.name)}">
            <span style="display:flex;align-items:center">
                <span class="layer-dot" style="background:var(--layer-${layer.level})"></span>
                ${escapeHtml(layer.name)}
            </span>
            <span class="count">${layer.count}</span>
        </button>
    `).join('');
}

function populateSearchFilter() {
    const sel = document.getElementById('search-layer-filter');
    sel.innerHTML = '<option value="">All layers</option>' +
        state.layers.map(l => `<option value="${escapeHtml(l.name)}">${escapeHtml(l.name)}</option>`).join('');
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

function switchTab(tabName) {
    // Update tab button styles
    document.querySelectorAll('#tab-bar .tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    // Browse tab shows list-panel + reader-panel, hides other panels
    const listPanel = document.getElementById('list-panel');
    const readerPanel = document.getElementById('reader-panel');
    const metaPanel = document.getElementById('meta-panel');
    const pipelinePanel = document.getElementById('pipeline-panel');
    const promptsPanel = document.getElementById('prompts-panel');

    // Hide all tab panels
    listPanel.style.display = 'none';
    readerPanel.style.display = 'none';
    metaPanel.style.display = 'none';
    if (pipelinePanel) pipelinePanel.style.display = 'none';
    if (promptsPanel) promptsPanel.style.display = 'none';

    if (tabName === 'browse') {
        listPanel.style.display = '';
        readerPanel.style.display = '';
        // Restore meta panel visibility based on previous toggle state
        metaPanel.style.display = '';
    } else if (tabName === 'pipeline') {
        if (pipelinePanel) pipelinePanel.style.display = '';
    } else if (tabName === 'prompts') {
        if (promptsPanel) promptsPanel.style.display = '';
    }

    state.activeTab = tabName;

    // Stop any existing pipeline refresh interval
    if (pipelineRefreshInterval) {
        clearInterval(pipelineRefreshInterval);
        pipelineRefreshInterval = null;
    }

    if (tabName === 'pipeline') {
        loadPipelineTab();
        pipelineRefreshInterval = setInterval(refreshBuildStatus, 3000);
    } else if (tabName === 'prompts') {
        loadPromptsTab();
    }
}

// --- Metadata badge rendering ---

function formatMetaValue(v) {
    if (v === null || v === undefined) return '';
    if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') return String(v);
    if (Array.isArray(v)) return v.length === 0 ? '[]' : `[${v.length} items]`;
    if (typeof v === 'object') {
        const keys = Object.keys(v);
        if (keys.length === 0) return '{}';
        const preview = keys.slice(0, 2).map(k => `${k}: ${String(v[k]).slice(0, 20)}`).join(', ');
        return keys.length <= 2 ? `{${preview}}` : `{${preview}, +${keys.length - 2}}`;
    }
    return String(v);
}

function renderMetaBadges(metadata) {
    const skip = new Set(['title', 'layer_name', 'layer_level', 'created_at']);
    return Object.entries(metadata || {})
        .filter(([k, v]) => !skip.has(k) && v != null && v !== '' && !k.startsWith('_'))
        .map(([k, v]) => `<span class="meta-badge">${escapeHtml(k)}: ${escapeHtml(formatMetaValue(v))}</span>`)
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
        // Detect if items have source_label metadata (chunk layers)
        const hasSourceLabel = data.items.some(item => item.metadata && item.metadata.source_label);

        if (hasSourceLabel) {
            // Group items by source_label
            const groups = new Map();
            for (const item of data.items) {
                const key = (item.metadata && item.metadata.source_label) || '(unknown source)';
                if (!groups.has(key)) groups.set(key, []);
                groups.get(key).push(item);
            }

            let html = '';
            for (const [sourceLabel, items] of groups) {
                const groupId = 'chunk-group-' + sourceLabel.replace(/[^a-zA-Z0-9]/g, '-');
                html += `<div class="chunk-group-header" data-expanded="true" data-group-id="${escapeHtml(groupId)}">
                    <span class="toggle-icon">&#9660;</span>
                    <span class="source-name">${escapeHtml(sourceLabel)}</span>
                    <span class="chunk-count">${items.length} chunks</span>
                </div>`;
                html += `<div class="chunk-group-items" data-group-id="${escapeHtml(groupId)}">`;
                html += items.map(item => {
                    const meta = item.metadata || {};
                    const chunkInfo = meta.chunk_index != null && meta.chunk_total != null
                        ? `${meta.chunk_index + 1}/${meta.chunk_total}`
                        : '';
                    const preview = item.title || '';
                    return `
                        <div class="list-item ${item.label === state.currentLabel ? 'active' : ''}"
                             data-label="${escapeHtml(item.label)}" title="${escapeHtml(item.title)}">
                            <div class="item-title">
                                ${chunkInfo ? `<span class="meta-badge">${escapeHtml(chunkInfo)}</span> ` : ''}${escapeHtml(preview)}
                            </div>
                            <div class="item-meta">
                                ${item.date ? `<span>${escapeHtml(item.date)}</span>` : ''}
                                ${renderMetaBadges(item.metadata)}
                            </div>
                        </div>
                    `;
                }).join('');
                html += '</div>';
            }
            listItems.innerHTML = html;
        } else {
            listItems.innerHTML = data.items.map(item => `
                <div class="list-item ${item.label === state.currentLabel ? 'active' : ''}"
                     data-label="${escapeHtml(item.label)}" title="${escapeHtml(item.title)}">
                    <div class="item-title">${escapeHtml(item.title)}</div>
                    <div class="item-meta">
                        ${item.date ? `<span>${escapeHtml(item.date)}</span>` : ''}
                        ${renderMetaBadges(item.metadata)}
                    </div>
                </div>
            `).join('');
        }
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
        el.classList.toggle('active', el.dataset.label === label);
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
                <div class="lineage-label ${collapsed ? 'lineage-collapsible' : ''}">
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
                <div class="lineage-label ${collapsed ? 'lineage-collapsible' : ''}">
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
                 data-label="${escapeHtml(item.label)}"
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
                <div class="list-item" data-label="${escapeHtml(item.label)}">
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

// --- Pipeline tab ---

let pipelineRefreshInterval = null;

function renderDAG(data) {
    const container = document.getElementById('pipeline-content');
    if (!data.nodes || data.nodes.length === 0) {
        container.innerHTML = '<div class="empty-state">No pipeline loaded</div>';
        return;
    }

    // Group nodes by level
    const levels = new Map();
    for (const node of data.nodes) {
        const level = node.level || 0;
        if (!levels.has(level)) levels.set(level, []);
        levels.get(level).push(node);
    }

    let html = '<div class="dag-container">';

    // Build edge lookup for highlighting connections
    const edgeMap = new Map(); // target -> [sources]
    for (const edge of (data.edges || [])) {
        if (!edgeMap.has(edge.target)) edgeMap.set(edge.target, []);
        edgeMap.get(edge.target).push(edge.source);
    }

    const sortedLevels = [...levels.keys()].sort((a, b) => a - b);
    for (const level of sortedLevels) {
        const nodes = levels.get(level);
        html += `<div class="dag-level">`;
        html += `<div class="dag-level-label">L${level}</div>`;
        html += `<div class="dag-level-nodes">`;
        for (const node of nodes) {
            const deps = edgeMap.get(node.id) || [];
            const depsStr = deps.length ? `← ${deps.join(', ')}` : '';
            html += `
                <div class="dag-node" data-type="${escapeHtml(node.type)}">
                    <div class="dag-node-name">${escapeHtml(node.id)}</div>
                    <div class="dag-node-meta">
                        <span class="dag-type-badge">${escapeHtml(node.type)}</span>
                        <span class="dag-count">${node.count} artifacts</span>
                    </div>
                    ${depsStr ? `<div class="dag-node-deps">${escapeHtml(depsStr)}</div>` : ''}
                </div>
            `;
        }
        html += `</div></div>`;
    }

    // Projections section
    if (data.projections && data.projections.length > 0) {
        html += '<div class="dag-level"><div class="dag-level-label">Out</div><div class="dag-level-nodes">';
        for (const proj of data.projections) {
            const sources = proj.sources || [];
            html += `
                <div class="dag-node dag-projection">
                    <div class="dag-node-name">${escapeHtml(proj.id)}</div>
                    <div class="dag-node-meta">
                        <span class="dag-type-badge">${escapeHtml(proj.type)}</span>
                    </div>
                    ${sources.length ? `<div class="dag-node-deps">← ${sources.map(s => escapeHtml(s)).join(', ')}</div>` : ''}
                </div>
            `;
        }
        html += '</div></div>';
    }

    html += '</div>';
    container.innerHTML = html;
}

function renderBuildStatus(data) {
    let html = '<div class="build-status-container">';
    html += '<h3>Build Queue</h3>';

    // Stats bar
    const stats = data.stats || {};
    html += `<div class="build-stats">`;
    html += `<span class="stat"><strong>${data.queue_depth || 0}</strong> pending</span>`;
    html += `<span class="stat"><strong>${stats.total_processed || 0}</strong> processed</span>`;
    if (stats.avg_build_time_seconds) {
        html += `<span class="stat"><strong>${stats.avg_build_time_seconds.toFixed(1)}s</strong> avg build</span>`;
    }
    html += `</div>`;

    // Recent documents
    const recent = data.recent || [];
    if (recent.length > 0) {
        html += '<div class="build-recent"><h4>Recent</h4>';
        html += '<div class="build-recent-list">';
        for (const doc of recent) {
            const statusClass = `status-${doc.status}`;
            html += `
                <div class="build-recent-item">
                    <span class="build-status-dot ${statusClass}"></span>
                    <span class="build-filename">${escapeHtml(doc.filename)}</span>
                    <span class="build-bucket">${escapeHtml(doc.bucket)}</span>
                    <span class="build-doc-status">${escapeHtml(doc.status)}</span>
                    ${doc.client_id ? `<span class="build-client">${escapeHtml(doc.client_id)}</span>` : ''}
                </div>
            `;
        }
        html += '</div></div>';
    } else {
        html += '<div class="empty-state">No recent builds</div>';
    }

    html += '</div>';
    return html;
}

// --- Build history & log detail ---

function formatTimestamp(isoStr) {
    if (!isoStr) return '';
    try {
        const d = new Date(isoStr);
        if (isNaN(d.getTime())) return isoStr;
        return d.toLocaleString(undefined, {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
    } catch { return isoStr; }
}

function shortRunId(runId) {
    if (!runId) return '';
    // Show first 15 chars (YYYYMMDDTHHMMSS) + last 8 (hex suffix)
    if (runId.length > 24) {
        return runId.slice(0, 15) + '...' + runId.slice(-8);
    }
    return runId;
}

function renderBuildHistory(logs) {
    if (!logs || logs.length === 0) {
        return '<div class="build-history"><h3>Build History</h3><div class="empty-state">No build logs found</div></div>';
    }

    let html = '<div class="build-history">';
    html += '<h3>Build History</h3>';
    html += '<div class="build-history-list">';
    for (const log of logs) {
        const ts = formatTimestamp(log.timestamp);
        const sizeKb = log.size_bytes ? (log.size_bytes / 1024).toFixed(1) + ' KB' : '';
        html += `
            <div class="build-run-item" data-run-id="${escapeHtml(log.run_id)}">
                <span class="build-run-id">${escapeHtml(shortRunId(log.run_id))}</span>
                <span class="build-run-ts">${escapeHtml(ts)}</span>
                <span class="build-run-size">${escapeHtml(sizeKb)}</span>
            </div>
        `;
    }
    html += '</div>';
    html += '<div id="build-log-detail"></div>';
    html += '</div>';
    return html;
}

function renderBuildLogDetail(log) {
    if (!log || log.error) {
        return `<div class="empty-state">${escapeHtml(log?.error || 'Failed to load build log')}</div>`;
    }

    let html = '<div class="build-log-detail">';

    // Header
    html += '<div class="build-log-header">';
    html += `<span class="build-log-pipeline">${escapeHtml(log.pipeline || 'pipeline')}</span>`;
    html += `<span class="build-log-time">${formatTimestamp(log.started_at)}</span>`;
    if (log.total_time) {
        html += `<span class="build-log-duration">${log.total_time.toFixed(1)}s total</span>`;
    }
    html += '</div>';

    // Layer timeline
    const layers = log.layers || [];
    if (layers.length > 0) {
        html += '<div class="build-log-layers">';
        for (const layer of layers) {
            const llmCalls = layer.llm_calls || [];
            const llmCount = llmCalls.length;
            const llmTokens = llmCalls.reduce((sum, c) => sum + (c.input_tokens || 0) + (c.output_tokens || 0), 0);
            const hasLlm = llmCount > 0;
            const expandId = 'llm-detail-' + layer.name.replace(/[^a-zA-Z0-9]/g, '-');

            html += `<div class="layer-row">`;
            html += `<span class="layer-level">L${layer.level}</span>`;
            html += `<span class="layer-name">${escapeHtml(layer.name)}</span>`;
            html += `<span class="layer-stats">${layer.built} built, ${layer.cached} cached</span>`;
            html += `<span class="layer-time">${layer.time_seconds != null ? layer.time_seconds.toFixed(2) + 's' : ''}</span>`;
            if (hasLlm) {
                html += `<span class="layer-llm layer-llm-toggle" data-expand="${expandId}">${llmCount} LLM call${llmCount !== 1 ? 's' : ''}, ${llmTokens} tok</span>`;
            }
            html += `</div>`;

            // Expandable LLM detail
            if (hasLlm) {
                html += `<div class="llm-detail" id="${expandId}" style="display:none;">`;
                for (const call of llmCalls) {
                    const callTokens = (call.input_tokens || 0) + (call.output_tokens || 0);
                    html += `<div class="llm-detail-row">`;
                    html += `<span class="llm-artifact">${escapeHtml(call.artifact || '')}</span>`;
                    html += `<span class="llm-duration">${call.duration != null ? call.duration.toFixed(1) + 's' : ''}</span>`;
                    html += `<span class="llm-tokens">${call.input_tokens || 0} in / ${call.output_tokens || 0} out (${callTokens} tok)</span>`;
                    if (call.model) {
                        html += `<span class="llm-model">${escapeHtml(call.model)}</span>`;
                    }
                    html += `</div>`;
                }
                html += '</div>';
            }
        }
        html += '</div>';
    }

    // Summary
    const summary = log.summary || {};
    html += '<div class="build-summary">';
    html += `<span class="stat"><strong>${summary.layers_count || 0}</strong> layers</span>`;
    html += `<span class="stat"><strong>${summary.total_llm_calls || 0}</strong> LLM calls</span>`;
    html += `<span class="stat"><strong>${summary.total_tokens || 0}</strong> tokens</span>`;
    if (log.total_time) {
        html += `<span class="stat"><strong>${log.total_time.toFixed(1)}s</strong> total time</span>`;
    }
    html += '</div>';

    html += '</div>';
    return html;
}

async function loadPipelineTab() {
    const container = document.getElementById('pipeline-content');
    container.innerHTML = '<div class="empty-state">Loading pipeline...</div>';

    try {
        const [dagRes, statusRes, logsRes] = await Promise.all([
            fetch('/api/dag').then(r => r.ok ? r.json() : null).catch(() => null),
            fetch('/api/build-status').then(r => r.ok ? r.json() : null).catch(() => null),
            fetch('/api/build-logs').then(r => r.ok ? r.json() : null).catch(() => null),
        ]);

        let html = '';
        if (dagRes) {
            // Render DAG into container, then capture the HTML
            renderDAG(dagRes);
            html = container.innerHTML;
        } else {
            html = '<div class="empty-state">No pipeline data available</div>';
        }

        if (statusRes) {
            html += renderBuildStatus(statusRes);
        }

        if (logsRes) {
            html += renderBuildHistory(logsRes.logs || []);
        }

        container.innerHTML = html;

        // Wire up click handlers for build history items and LLM detail toggles
        _wirePipelineEvents(container);

        // Auto-load the most recent build log detail
        const firstRunItem = container.querySelector('.build-run-item');
        if (firstRunItem) {
            firstRunItem.click();
        }
    } catch (err) {
        container.innerHTML = `<div class="empty-state">Error loading pipeline: ${escapeHtml(err.message)}</div>`;
    }
}

function _wirePipelineEvents(container) {
    container.addEventListener('click', async (e) => {
        // Build run item click — load detail
        const runItem = e.target.closest('.build-run-item');
        if (runItem) {
            const runId = runItem.dataset.runId;
            if (!runId) return;

            // Mark active
            container.querySelectorAll('.build-run-item').forEach(el => {
                el.classList.toggle('active', el === runItem);
            });

            const detailEl = document.getElementById('build-log-detail');
            if (detailEl) {
                detailEl.innerHTML = '<div class="empty-state">Loading build log...</div>';
                try {
                    const res = await fetch(`/api/build-log?run_id=${encodeURIComponent(runId)}`);
                    const data = await res.json();
                    detailEl.innerHTML = renderBuildLogDetail(data);
                    // Wire LLM toggle events within the detail
                    _wireLlmToggles(detailEl);
                } catch (err) {
                    detailEl.innerHTML = `<div class="empty-state">Error: ${escapeHtml(err.message)}</div>`;
                }
            }
            return;
        }

        // LLM detail toggle
        const llmToggle = e.target.closest('.layer-llm-toggle');
        if (llmToggle) {
            _handleLlmToggle(llmToggle);
        }
    });
}

function _wireLlmToggles(container) {
    container.querySelectorAll('.layer-llm-toggle').forEach(toggle => {
        toggle.addEventListener('click', () => _handleLlmToggle(toggle));
    });
}

function _handleLlmToggle(toggle) {
    const expandId = toggle.dataset.expand;
    if (!expandId) return;
    const detail = document.getElementById(expandId);
    if (!detail) return;
    const isVisible = detail.style.display !== 'none';
    detail.style.display = isVisible ? 'none' : 'block';
    toggle.classList.toggle('expanded', !isVisible);
}

async function refreshBuildStatus() {
    try {
        const res = await fetch('/api/build-status');
        if (!res.ok) return;
        const data = await res.json();
        // Update just the build status section
        const existing = document.querySelector('.build-status-container');
        if (existing) {
            const tmp = document.createElement('div');
            tmp.innerHTML = renderBuildStatus(data);
            existing.replaceWith(tmp.firstElementChild);
        }
    } catch (err) {
        console.warn('Failed to refresh build status:', err);
    }
}

// --- Prompts tab ---

let currentPromptKey = null;

async function loadPromptsTab() {
    const container = document.getElementById('prompts-content');
    container.innerHTML = '<div class="empty-state">Loading prompts...</div>';

    try {
        const res = await fetch('/api/prompts');
        if (!res.ok) throw new Error('Failed to load prompts');
        const data = await res.json();

        const prompts = data.prompts || [];
        if (prompts.length === 0) {
            container.innerHTML = '<div class="empty-state">No prompts configured</div>';
            return;
        }

        let html = '<div class="prompts-layout">';

        // Prompt list (left sidebar)
        html += '<div class="prompts-list">';
        for (const p of prompts) {
            html += `
                <div class="prompt-item" data-key="${escapeHtml(p.key)}">
                    <div class="prompt-key">${escapeHtml(p.key)}</div>
                    <div class="prompt-meta">v${p.version || p.versions_count || 1}</div>
                </div>
            `;
        }
        html += '</div>';

        // Editor (right panel)
        html += '<div class="prompt-editor" id="prompt-editor">';
        html += '<div class="empty-state">Select a prompt to edit</div>';
        html += '</div>';

        html += '</div>';
        container.innerHTML = html;

        // Event delegation for prompt list, buttons, and history entries
        // Guard against stacking listeners on repeated tab switches
        if (container.dataset.delegated) return;
        container.dataset.delegated = 'true';
        container.addEventListener('click', (e) => {
            const item = e.target.closest('.prompt-item');
            if (item) {
                loadPrompt(item.dataset.key);
                return;
            }
            const saveBtn = e.target.closest('.btn-save');
            if (saveBtn && currentPromptKey) {
                savePrompt(currentPromptKey);
                return;
            }
            const histBtn = e.target.closest('.btn-history');
            if (histBtn && currentPromptKey) {
                loadPromptHistory(currentPromptKey);
                return;
            }
            const histEntry = e.target.closest('.history-entry');
            if (histEntry) {
                const version = histEntry.dataset.version;
                const key = histEntry.dataset.key;
                if (version && key) loadPromptVersion(key, parseInt(version));
                return;
            }
        });

    } catch (err) {
        container.innerHTML = `<div class="empty-state">Error: ${escapeHtml(err.message)}</div>`;
    }
}

async function loadPrompt(key) {
    currentPromptKey = key;
    const editor = document.getElementById('prompt-editor');
    if (!editor) return;

    // Highlight selected item
    document.querySelectorAll('.prompt-item').forEach(el => {
        el.classList.toggle('active', el.dataset.key === key);
    });

    try {
        const res = await fetch(`/api/prompts/${encodeURIComponent(key)}`);
        if (!res.ok) throw new Error('Failed to load prompt');
        const data = await res.json();

        let html = `
            <div class="prompt-editor-header">
                <h3>${escapeHtml(key)}</h3>
                <span class="prompt-version">v${data.version} — ${escapeHtml(data.content_hash)}</span>
            </div>
            <textarea id="prompt-textarea" class="prompt-textarea"></textarea>
            <div class="prompt-actions">
                <button class="btn-save">Save</button>
                <button class="btn-history">History</button>
            </div>
            <div id="prompt-history-panel"></div>
        `;
        editor.innerHTML = html;
        const textarea = document.getElementById('prompt-textarea');
        if (textarea) textarea.value = data.content;
    } catch (err) {
        editor.innerHTML = `<div class="empty-state">Error: ${escapeHtml(err.message)}</div>`;
    }
}

async function savePrompt(key) {
    const textarea = document.getElementById('prompt-textarea');
    if (!textarea) return;

    try {
        const res = await fetch(`/api/prompts/${encodeURIComponent(key)}`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({content: textarea.value}),
        });
        if (!res.ok) throw new Error('Save failed');
        const data = await res.json();

        // Update version display
        const versionEl = document.querySelector('.prompt-version');
        if (versionEl) {
            versionEl.textContent = `v${data.version} — ${data.content_hash}`;
        }

        // Flash save confirmation
        const btn = document.querySelector('.btn-save');
        if (btn) {
            btn.textContent = 'Saved!';
            setTimeout(() => { btn.textContent = 'Save'; }, 1500);
        }
    } catch (err) {
        alert('Failed to save: ' + err.message);
    }
}

async function loadPromptHistory(key) {
    const panel = document.getElementById('prompt-history-panel');
    if (!panel) return;

    try {
        const res = await fetch(`/api/prompts/${encodeURIComponent(key)}/history`);
        if (!res.ok) throw new Error('Failed to load history');
        const data = await res.json();

        const versions = data.versions || [];
        if (versions.length === 0) {
            panel.innerHTML = '<div class="empty-state">No version history</div>';
            return;
        }

        let html = '<div class="prompt-history"><h4>Version History</h4>';
        for (const v of versions) {
            html += `
                <div class="history-entry" data-key="${escapeHtml(key)}" data-version="${v.version}">
                    <span class="history-version">v${v.version}</span>
                    <span class="history-date">${escapeHtml(v.created_at)}</span>
                    <span class="history-hash">${escapeHtml(v.content_hash)}</span>
                </div>
            `;
        }
        html += '</div>';
        panel.innerHTML = html;
    } catch (err) {
        panel.innerHTML = `<div class="empty-state">Error: ${escapeHtml(err.message)}</div>`;
    }
}

async function loadPromptVersion(key, version) {
    try {
        const res = await fetch(`/api/prompts/${encodeURIComponent(key)}?version=${version}`);
        if (!res.ok) throw new Error('Failed to load version');
        const data = await res.json();
        const textarea = document.getElementById('prompt-textarea');
        if (textarea) textarea.value = data.content;
    } catch (err) {
        console.warn('Failed to load prompt version:', err);
    }
}

// --- Utility ---

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
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
            },
            link(token) {
                const href = token.href || '';
                // Block javascript:, vbscript:, data: protocols
                if (/^\s*(javascript|vbscript|data):/i.test(href)) {
                    return token.text || escapeHtml(href);
                }
                const title = token.title ? ` title="${escapeHtml(token.title)}"` : '';
                return `<a href="${escapeHtml(href)}"${title} rel="noopener noreferrer">${token.text || ''}</a>`;
            },
            image(token) {
                const src = token.href || '';
                if (/^\s*(javascript|vbscript|data):/i.test(src)) {
                    return escapeHtml(token.text || '');
                }
                const title = token.title ? ` title="${escapeHtml(token.title)}"` : '';
                return `<img src="${escapeHtml(src)}" alt="${escapeHtml(token.text || '')}"${title}>`;
            }
        }
    });
    init();
});
