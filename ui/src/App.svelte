<script>
    import { onMount, onDestroy, setContext } from 'svelte';
    import { writable, get } from 'svelte/store';
    import { SvelteFlow, Controls, ControlButton, Background, MiniMap, SvelteFlowProvider } from '@xyflow/svelte';

    import Toolbar from './lib/Toolbar.svelte';
    import CustomNode from './lib/CustomNode.svelte';
    import FlowDropZone from './lib/FlowDropZone.svelte';
    import DrawerSidebar from './lib/DrawerSidebar.svelte';
    import NodePalettePanel from './lib/NodePalettePanel.svelte';
    import RunProgress from './lib/RunProgress.svelte';
    import { generateNodeClasses } from './lib/utils.js';
    import { fetchCatalogue, fromDocument, process, stream, toDocument } from './lib/api.js';
    import { openSidebar, closeSidebar, pendingConnection, clearPendingConnection, chosenFile, running } from './lib/stores.js';

    // Every workflow starts here: pixels have to come from somewhere, and this is the node whose
    // path the runner overrides per image.
    const createLoadImageNode = () => ({
        id: 'read_media-1',
        type: 'default',
        position: { x: 100, y: 100 },
        data: { nodeType: 'read_media', parameters: {} },
        class: 'node-category-input'
    });

    let nodes = writable([createLoadImageNode()]);
    let edges = writable([]);
    let selectedNode = writable(null);
    let availableNodes = writable([]);
    let executionResults = writable(null);
    let isExecuting = writable(false);

    //: The abort handle of the run in progress, or null. Cancelling is aborting: the connection
    //: closes, the server's generator closes with it, and the run ends having closed its files.
    let controller = null;

    let svelteFlowInstance;
    //: What the last connection was refused for, shown beside the canvas rather than in an alert.
    //: A modal stops the editor to say a drag did not take, which the missing edge already said.
    let refusal = null;
    let refusalTimer;

    // Set context for child components - must be during initialization
    setContext('availableNodes', availableNodes);

    // Define custom node types for SvelteFlow
    const nodeTypes = {
        default: CustomNode
    };

    onMount(async () => {
        try {
            availableNodes.set(await fetchCatalogue());
        } catch (error) {
            say(`Could not load the node catalogue: ${error.message}`);
        }
    });

    /** Say something went wrong, where the person is looking, and take it back after a while. */
    function say(message) {
        refusal = message;
        clearTimeout(refusalTimer);
        refusalTimer = setTimeout(() => (refusal = null), 6000);
    }

    function initializeCanvas() {
        nodes.set([createLoadImageNode()]);
        edges.set([]);
    }

    function handleNodeDrop(event) {
        const { nodeType, position } = event.detail;

        const category = getNodeCategory(nodeType);
        const newNode = {
            id: `${nodeType}-${Date.now()}`,
            type: 'default',
            position,
            data: {
                nodeType: nodeType,
                parameters: getDefaultParameters()
            },
            class: generateNodeClasses(nodeType, category),
            origin: [0.5, 0.0]
        };

        nodes.update(n => [...n, newNode]);

        // Check if there's a pending connection to auto-connect
        const pending = $pendingConnection;
        if (pending) {
            // Create auto-connection between pending handle and new node
            createAutoConnection(pending, newNode);
            clearPendingConnection();
        }
    }

    /** A new node sets nothing. Unset means the node's own default, on both sides of the wire. */
    function getDefaultParameters() {
        return {};
    }

    function getNodeCategory(nodeType) {
        const currentNodes = $availableNodes || {};
        return currentNodes[nodeType]?.category || 'Other';
    }

    function onNodeClick(event) {
        const node = event.detail.node;

        selectedNode.set(node);
        // Auto-open properties panel when a node is selected
        openSidebar('properties');
    }

    function onPaneClick() {
        // Close drawer when clicking on empty canvas
        closeSidebar();
        // Clear any pending connections
        clearPendingConnection();
    }

    function createAutoConnection(pending, newNode) {
        const { nodeId: pendingNodeId, handleId: pendingHandleId, handleType: pendingHandleType } = pending;

        let sourceNode, targetNode, sourceHandle, targetHandle;

        if (pendingHandleType === 'source') {
            // Pending node has output handle, new node should provide input
            sourceNode = pendingNodeId;
            targetNode = newNode.id;
            sourceHandle = pendingHandleId;
            targetHandle = getDefaultInput(newNode.data.nodeType);
        } else {
            // Pending node has input handle, new node should provide output
            sourceNode = newNode.id;
            targetNode = pendingNodeId;
            sourceHandle = getDefaultOutput(newNode.data.nodeType);
            targetHandle = pendingHandleId;
        }

        // Validate connection before creating
        if (sourceHandle && targetHandle) {
            const sourceNodeType = pendingHandleType === 'source' ?
                $nodes.find(n => n.id === pendingNodeId)?.data?.nodeType :
                newNode.data.nodeType;
            const targetNodeType = pendingHandleType === 'target' ?
                $nodes.find(n => n.id === pendingNodeId)?.data?.nodeType :
                newNode.data.nodeType;

            if (portsAgree(sourceNodeType, sourceHandle, targetNodeType, targetHandle)) {
                const newEdge = {
                    id: `edge-${Date.now()}`,
                    source: sourceNode,
                    target: targetNode,
                    sourceHandle: sourceHandle,
                    targetHandle: targetHandle
                };

                edges.update(edges => [...edges, newEdge]);
            } else {
                say(`${sourceNodeType}.${sourceHandle} carries `
                    + `${typeOf($availableNodes?.[sourceNodeType]?.outputs, sourceHandle)}, but `
                    + `${targetNodeType}.${targetHandle} takes `
                    + `${typeOf($availableNodes?.[targetNodeType]?.inputs, targetHandle)}.`);
            }
        }
    }

    /**
     * Whether a connection may be made, asked by Svelte Flow while the wire is being dragged.
     *
     * Its own hook rather than a check on the `connect` event, because that event is a
     * notification: the library has already added the edge by the time it fires, and returning
     * early from it leaves the edge on the canvas. Asked here, an incompatible target simply does
     * not take the wire -- and it says so while the drag is still happening, by which handles
     * light up, rather than after the fact.
     */
    function isValidConnection(connection) {
        const source = $nodes.find(n => n.id === connection.source)?.data?.nodeType;
        const target = $nodes.find(n => n.id === connection.target)?.data?.nodeType;
        if (!source || !target) return false;

        // The handles the user actually dragged. Only fall back to the first port when a drag did
        // not name one -- picking the first regardless would make a multi-output node's second
        // port unreachable, and would send a handle nobody chose.
        return portsAgree(source, connection.sourceHandle || getDefaultOutput(source),
                          target, connection.targetHandle || getDefaultInput(target));
    }

    /** Whether what one port carries is what the other takes. */
    function portsAgree(sourceNodeType, sourceHandle, targetNodeType, targetHandle) {
        const sourceNodeInfo = $availableNodes?.[sourceNodeType];
        const targetNodeInfo = $availableNodes?.[targetNodeType];

        if (!sourceNodeInfo || !targetNodeInfo) return false;
        if (!sourceHandle || !targetHandle) return false;

        // Ports are ordered lists of {name, type}, so they are found by name rather than indexed.
        // Indexing an array by a handle name yields undefined; indexing it by position yields a
        // port object, and comparing two of those compares references, which are never equal.
        const source = sourceNodeInfo.outputs?.find(port => port.name === sourceHandle);
        const target = targetNodeInfo.inputs?.find(port => port.name === targetHandle);

        if (!source || !target) return false;
        return source.type === target.type;
    }

    /** What a named port carries, for saying why two of them would not join. */
    function typeOf(ports, name) {
        return ports?.find(port => port.name === name)?.type || 'nothing';
    }

    /** The name of a node's first output port, for a drag that did not name one. */
    function getDefaultOutput(nodeType) {
        return $availableNodes?.[nodeType]?.outputs?.[0]?.name || null;
    }

    /** The name of a node's first input port, for a drag that did not name one. */
    function getDefaultInput(nodeType) {
        return $availableNodes?.[nodeType]?.inputs?.[0]?.name || null;
    }

    function updateNodeParameters(nodeId, parameters) {
        nodes.update(n =>
            n.map(node =>
                node.id === nodeId
                    ? { ...node, data: { ...node.data, parameters } }
                    : node
            )
        );
        // And the selection, which the panel reads its current values from. Left stale, the panel
        // spreads the values it was opened with, so setting a second parameter discarded the
        // first.
        selectedNode.update(node =>
            node && node.id === nodeId
                ? { ...node, data: { ...node.data, parameters } }
                : node
        );
    }

    // Helper to strip execution state classes from a node's class string
    function stripExecutionClasses(classStr) {
        return classStr
            .replace(/\s*node-running/g, '')
            .replace(/\s*node-completed/g, '')
            .replace(/\s*node-error/g, '')
            .replace(/\s*node-has-output/g, '');
    }

    // Update a single node's class with execution state
    function updateNodeClass(nodeId, stateClass) {
        nodes.update(n => n.map(node => {
            if (node.id !== nodeId) return node;
            const baseClass = stripExecutionClasses(node.class);
            return { ...node, class: `${baseClass} ${stateClass}` };
        }));
    }

    async function testWorkflow() {
        isExecuting.set(true);
        executionResults.set({ success: true, results: {} });

        // Reset all node classes before starting
        nodes.update(n => n.map(node => ({
            ...node,
            class: stripExecutionClasses(node.class)
        })));

        try {
            await stream(toDocument($nodes, $edges), $chosenFile, (event) => {
                if (event.done) {
                    isExecuting.set(false);
                } else if (event.status === 'running') {
                    updateNodeClass(event.node, 'node-running');
                } else if (event.status === 'completed') {
                    updateNodeClass(event.node, 'node-completed node-has-output');
                    executionResults.update(r => ({
                        ...r,
                        results: { ...r.results, [event.node]: event.output }
                    }));
                } else if (event.status === 'failed') {
                    if (event.node) updateNodeClass(event.node, 'node-error');
                    executionResults.update(r => ({ ...r, success: false, error: event.error }));
                    say(event.error);
                    isExecuting.set(false);
                }
            });
        } catch (error) {
            executionResults.update(r => ({ ...r, success: false, error: error.message }));
            say(error.message);
        } finally {
            isExecuting.set(false);
        }
    }

    /**
     * Run over the whole source: every frame of a video, every file in a folder.
     *
     * No node outputs come back and none are drawn -- see `RunProgress`. What is watched is
     * whichever node is selected, falling back to a terminal, because the end of the graph is what
     * a person means by "show me" when they have not said.
     */
    async function runWorkflow() {
        const fed = new Set($edges.map(edge => edge.source));
        const watched = $selectedNode?.id || $nodes.filter(node => !fed.has(node.id)).pop()?.id || '';

        controller = new AbortController();
        running.set({ items: 0, seconds: 0, preview: null, node: watched, done: false });
        const began = performance.now();

        try {
            await process(toDocument($nodes, $edges), $chosenFile, { preview: watched, signal: controller.signal },
                          (event) => {
                if (event.status === 'failed') {
                    say(event.error);
                    running.set(null);
                    return;
                }
                if (event.done) {
                    running.update(r => r && { ...r, items: event.items,
                                               seconds: event.seconds, done: true });
                    return;
                }
                running.update(r => r && {
                    ...r,
                    items: event.item,
                    seconds: (performance.now() - began) / 1000,
                    preview: event.preview || r.preview,
                });
            });
        } catch (error) {
            // An abort is the Cancel button working, not a failure to report.
            if (error.name !== 'AbortError') {
                say(error.message);
                running.set(null);
            }
        } finally {
            controller = null;
        }
    }

    /** Hang up. That is the whole of cancelling: the server's generator closes with the socket. */
    function cancelRun() {
        controller?.abort();
        running.set(null);
    }

    function exportWorkflow() {
        const dataStr = JSON.stringify(toDocument($nodes, $edges), null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);

        const link = document.createElement('a');
        link.href = url;
        link.download = 'workflow.json';
        link.click();

        URL.revokeObjectURL(url);
    }

    function importWorkflow(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const canvas = fromDocument(JSON.parse(e.target.result), $availableNodes);
                nodes.set(canvas.nodes.map(node => ({
                    ...node, type: 'default',
                    class: generateNodeClasses(
                        node.data.nodeType, $availableNodes[node.data.nodeType]?.category || '')
                })));
                edges.set(canvas.edges);
            } catch (error) {
                say(`That file is not a workflow: ${error.message}`);
            }
        };
        reader.readAsText(file);
    }

    function clearWorkflow() {
        initializeCanvas();
        selectedNode.set(null);
        executionResults.set(null);
    }

    onDestroy(() => clearTimeout(refusalTimer));

</script>

<SvelteFlowProvider>
            <Toolbar
                    {testWorkflow}
                    {runWorkflow}
                    {cancelRun}
                    {exportWorkflow}
                    {importWorkflow}
                    isExecuting={$isExecuting}
                    isRunning={!!$running && !$running.done}
            />


<!-- Left-side node palette (always visible) -->
<NodePalettePanel {availableNodes} />

<!-- Main app layout without fixed sidebar -->
<div class="main-content">
    <div id="studio">
        <FlowDropZone on:nodedrop={handleNodeDrop}>
            <SvelteFlow
                    {nodes}
                    {edges}
                    {nodeTypes}
                    proOptions={{ hideAttribution: true }}
                    bind:this={svelteFlowInstance}
                    on:nodeclick={onNodeClick}
                    on:paneclick={onPaneClick}
                    {isValidConnection}
                    maxZoom={1}
                    fitView
                    fitViewOptions={{
                      maxZoom: 1,      // Prevents zooming in beyond 100%
                      padding: 0.2     // Adds some padding around nodes
                    }}
            >
<!--                <Controls />-->
                <Background variant="dots" />
            </SvelteFlow>
        </FlowDropZone>
    </div>
</div>

{#if refusal}
    <div class="refusal" role="status">
        {refusal}
        <button class="delete is-small" aria-label="dismiss" on:click={() => (refusal = null)}></button>
    </div>
{/if}

<!-- Drawer sidebar for context-aware panels -->
<DrawerSidebar
    {availableNodes}
    selectedNode={$selectedNode}
    {updateNodeParameters}
    executionResults={$executionResults}
/>

<RunProgress />

</SvelteFlowProvider>

<style>
    /* While a wire is being dragged, Svelte Flow marks the handle under the pointer `connectingto`
       and adds `valid` when `isValidConnection` said yes. Saying which is the whole point of being
       asked before the connection is made rather than after. */
    :global(.svelte-flow__handle.connectingto) {
        box-shadow: 0 0 0 3px rgba(214, 54, 56, 0.35);
    }

    :global(.svelte-flow__handle.connectingto.valid) {
        box-shadow: 0 0 0 3px rgba(72, 199, 116, 0.45);
    }

    /* Said at the bottom of the canvas, where a rejected drag ended, and dismissable. Not an
       alert(): a modal halts everything to report something the missing edge already showed. */
    .refusal {
        position: fixed;
        left: 50%;
        bottom: 1.5rem;
        transform: translateX(-50%);
        z-index: 40;
        max-width: min(40rem, 90vw);
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.6rem 0.9rem;
        border-radius: 6px;
        background: #2b2b2b;
        color: #fff;
        font-size: 0.85rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
</style>