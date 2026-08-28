/**
 * Everything the editor knows about mozo's HTTP API.
 *
 * One module, so that the shape of a request lives in one place rather than in whichever component
 * needed it. The node catalogue is passed through as the server sends it -- ordered lists of ports
 * and parameters, derived from the functions that implement them -- and only keyed by name, which
 * is a view rather than a second shape to keep in step.
 */

/** Build a URL, honouring a mount prefix if the page was served under one. */
export function apiUrl(path) {
    const config = (typeof window !== 'undefined' && window.APP_CONFIG) || {};
    return (config.apiBase || '') + path;
}

/**
 * Every node the server offers, keyed by name.
 *
 * The entries are exactly what `/workflow/nodes` returns: `{name, category, description, inputs,
 * outputs, parameters}`, where the last three are ordered lists. Nothing is renamed or reshaped --
 * the catalogue is derived server-side from the node functions, and a translation here would be
 * the second place for it to disagree from.
 */
export async function fetchCatalogue() {
    const response = await fetch(apiUrl('/workflow/nodes'));
    if (!response.ok) throw new Error(`could not load the node catalogue (${response.status})`);
    const { nodes } = await response.json();
    return Object.fromEntries(nodes.map(node => [node.name, node]));
}

/**
 * The editor's canvas as a workflow document.
 *
 * Svelte Flow carries every node as type `custom` with the real kind in `data.nodeType`, because
 * one component draws them all. mozo's format names the kind at the top level. This is the only
 * place that conversion happens, in each direction.
 */
export function toDocument(nodes, edges) {
    return {
        nodes: nodes.map(node => ({
            id: node.id,
            type: node.data.nodeType,
            position: node.position,
            data: { parameters: node.data.parameters || {} },
        })),
        edges: edges.map(edge => ({
            source: edge.source,
            sourceHandle: edge.sourceHandle,
            target: edge.target,
            targetHandle: edge.targetHandle,
        })),
    };
}

/** A workflow document as canvas nodes and edges. The inverse of {@link toDocument}. */
export function fromDocument(document, catalogue) {
    const nodes = (document.nodes || []).map(node => ({
        id: node.id,
        type: 'custom',
        position: node.position || { x: 0, y: 0 },
        data: {
            nodeType: node.type,
            label: catalogue?.[node.type]?.name || node.type,
            parameters: node.data?.parameters || {},
        },
    }));
    const edges = (document.edges || []).map((edge, index) => ({
        id: `${edge.source}-${edge.sourceHandle}-${edge.target}-${edge.targetHandle}-${index}`,
        source: edge.source,
        sourceHandle: edge.sourceHandle,
        target: edge.target,
        targetHandle: edge.targetHandle,
    }));
    return { nodes, edges };
}

/** The form body every run endpoint takes: the document, an optional file, optional overrides. */
function body(document, file, fields) {
    const form = new FormData();
    form.append('workflow', JSON.stringify(document));
    for (const [name, value] of Object.entries(fields || {})) form.append(name, value);
    if (file) form.append('file', file);
    return form;
}

/** Ask whether a document is a workflow. Returns `{valid, order, terminals}` or `{valid, error}`. */
export async function validate(document) {
    const response = await fetch(apiUrl('/workflow/validate'), {
        method: 'POST', body: body(document, null, {}),
    });
    return response.json();
}

/**
 * Read a server-sent event stream, calling `onEvent` for each one.
 *
 * Shared by both verbs because the framing is the same either way; what differs is which endpoint
 * and what the events say. `signal` is how a run is cancelled: aborting closes the connection,
 * which closes the generator on the server, which ends the run and closes what it opened.
 */
async function events(path, form, onEvent, signal) {
    const response = await fetch(apiUrl(path), { method: 'POST', body: form, signal });

    if (!response.ok) {
        let detail = `the server refused the workflow (${response.status})`;
        try { detail = (await response.json()).detail || detail; } catch { /* not JSON */ }
        throw new Error(detail);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
            if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)));
        }
    }
}

/**
 * Test a workflow: one item through the graph, with every node's output drawn on the canvas.
 *
 * `include: 'all'` because the editor draws every node's result -- which is the case the server's
 * default is not tuned for, and the reason it is a choice rather than a fixed rule.
 */
export function stream(document, file, onEvent) {
    return events('/workflow/stream', body(document, file, { include: 'all' }), onEvent);
}

/**
 * Run a workflow over its whole source, calling `onEvent` with `{item}` as each one finishes.
 *
 * No node outputs come back. One pass over a two-hour video through two nodes would be 1.3 TB of
 * canvas images; what arrives instead is a counter and, at most five times a second, one small
 * JPEG of `preview`. Pass an `AbortSignal` to cancel -- there is nothing else to call.
 */
export function process(document, file, { preview = '', signal } = {}, onEvent) {
    return events('/workflow/process', body(document, file, { preview }), onEvent, signal);
}
